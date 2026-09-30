import os

import cv2
import numpy as np


def _preload_cuda():
    """Đưa DLL CUDA của torch vào process trước khi onnxruntime mở GPU."""
    try:
        import torch
    except Exception:
        return
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(lib):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(lib)
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")


_preload_cuda()

from paddleocr import PaddleOCR

from ocr.process_ocr_text import PlateFormatter


class LicensePlateOCR:
    # Một kích thước cho mọi crop. 128 chia hết 32 và lớn hơn limit_side_len 64.
    DET_SIDE = 128
    FIXED_BATCH = 4
    REC_HEIGHT = 48
    REC_WIDTH = 320

    def __init__(self, use_gpu=True, min_confidence=0.6, batch_size=8):

        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.formatter = PlateFormatter()
        # PaddleOCR cần thư mục model (inference.yml + onnx), không nhận một file .onnx.
        # tiny đã có sẵn trong cache paddlex. small sẽ phải tải thêm.
        engine_config = None
        if use_gpu:
            # Shape đã khóa nên chỉ dò một lần lúc warmup.
            # DEFAULT rơi vào Fallback mode và chậm mỗi lần gọi.
            engine_config = {
                "provider_options": [
                    {"cudnn_conv_algo_search": "HEURISTIC"},
                    {},
                ]
            }
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",
            text_recognition_batch_size=batch_size,
            engine="onnxruntime",
            engine_config=engine_config,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            device="gpu" if use_gpu else "cpu",
        )
        inner = self.ocr.paddlex_pipeline._pipeline
        self._det = inner.text_det_model
        self._rec = inner.text_rec_model
        self._crop = inner._crop_by_polys
        self._det_params = inner.get_text_det_params(
            None, None, None, None, None, None)
        norm = self._det.pre_tfs["Normalize"]
        self._det_alpha = np.asarray(norm.alpha, dtype=np.float32)
        self._det_beta = np.asarray(norm.beta, dtype=np.float32)
        self._warmup()



    def read_batch(self, crop_images, min_confidence = None):

        threshold = min_confidence if min_confidence  is not None else self.min_confidence
        if not crop_images:
            return []

        valid_images = []
        valid_indices = []

        for i , image in enumerate(crop_images):
            if image is not None and image.size  > 0:
                valid_images.append(image)
                valid_indices.append(i)
        output = [{'text': '', 'confidence': 0.0, 'valid': False} for _ in crop_images]

        if not valid_images:
            return output

        raw_results = self._run_batch_inference(valid_images)

        for local_idx, ocr_result in enumerate(raw_results):
            original_idx = valid_indices[local_idx]
            text, confidence = self._parse_single_result(ocr_result)

            valid = bool(text) and confidence >= threshold
            output[original_idx] = {
                'text': text if valid else '',
                'confidence': confidence,
                'valid': valid
            }

        return output
    def _as_bgr(self, image):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def _letterbox(self, image):
        """Scale đều cho vừa 128×128, phần thiếu đệm trắng. Không kéo giãn chữ."""
        image = self._as_bgr(image)
        side = self.DET_SIDE
        height, width = image.shape[:2]
        canvas = np.full((side, side, 3), 255, dtype=np.uint8)
        if height <= 0 or width <= 0:
            return canvas
        scale = side / float(max(height, width))
        new_w = max(1, min(side, int(round(width * scale))))
        new_h = max(1, min(side, int(round(height * scale))))
        resized = cv2.resize(image, (new_w, new_h))
        y0 = (side - new_h) // 2
        x0 = (side - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def _blank_canvas(self):
        return np.full((self.DET_SIDE, self.DET_SIDE, 3), 255, dtype=np.uint8)

    def _warmup(self):
        """Chạy một lần đúng hai shape sẽ dùng suốt video."""
        self._infer_chunk([self._blank_canvas()])

    def _write_det_sample(self, blob, index, image):
        view = image.astype(np.float32, copy=False)
        alpha = self._det_alpha
        beta = self._det_beta
        for channel in range(3):
            blob[index, channel] = view[:, :, channel] * alpha[channel] + beta[channel]

    def _detect_polys(self, images):
        """Mọi ảnh đã là 128×128. Tensor detection luôn (N, 3, 128, 128)."""
        side = self.DET_SIDE
        blob = np.empty((len(images), 3, side, side), np.float32)
        for index, image in enumerate(images):
            self._write_det_sample(blob, index, image)

        preds = self._det.runner(x=blob)
        shape = np.array([side, side, 1.0, 1.0], dtype=np.float32)
        polys, _scores = self._det.post_op(
            preds,
            [shape] * len(images),
            thresh=self._det_params["thresh"],
            box_thresh=self._det_params["box_thresh"],
            unclip_ratio=self._det_params["unclip_ratio"],
        )
        return polys

    def _prepare_rec(self, image):
        """Cao 48, rộng tối đa 320. Dòng rộng hơn thì thu lại, không nới tensor."""
        image = self._as_bgr(image)
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return np.zeros((self.REC_HEIGHT, 1, 3), np.uint8)
        scale = self.REC_HEIGHT / float(height)
        new_w = max(1, int(round(width * scale)))
        new_h = self.REC_HEIGHT
        if new_w > self.REC_WIDTH:
            scale = self.REC_WIDTH / float(width)
            new_w = self.REC_WIDTH
            new_h = max(1, min(self.REC_HEIGHT, int(round(height * scale))))
        return cv2.resize(image, (new_w, new_h))

    def _recognize(self, crops):
        """Tensor recognition luôn (4, 3, 48, 320). Chỉ trả dòng thật."""
        prepared = [self._prepare_rec(crop) for crop in crops]
        texts = []
        scores = []
        for start in range(0, len(prepared), self.FIXED_BATCH):
            chunk = prepared[start:start + self.FIXED_BATCH]
            real = len(chunk)
            blob = np.zeros(
                (self.FIXED_BATCH, 3, self.REC_HEIGHT, self.REC_WIDTH), np.float32)
            for index, sample in enumerate(chunk):
                values = sample.astype(np.float32).transpose(2, 0, 1)
                values /= 255.0
                values -= 0.5
                values /= 0.5
                blob[index, :, :sample.shape[0], :sample.shape[1]] = values

            logits = self._rec.runner(x=blob)[0]
            indices = logits.argmax(axis=-1)
            probs = np.max(logits, axis=-1)
            decoded = self._rec.post_op.decode(
                indices, probs, is_remove_duplicate=True, return_word_box=False)
            for item in decoded[:real]:
                texts.append(item[0])
                scores.append(item[1])
        return texts, scores

    def _infer_chunk(self, images):
        real = len(images)
        canvases = [self._letterbox(image) for image in images]
        while len(canvases) < self.FIXED_BATCH:
            canvases.append(self._blank_canvas())
        polys_per_image = self._detect_polys(canvases)

        crops = []
        owners = []
        kept_per_image = []
        for index in range(real):
            boxes = polys_per_image[index]
            boxes = [] if boxes is None or len(boxes) == 0 else list(boxes)
            source = canvases[index]
            if not boxes:
                side = self.DET_SIDE
                boxes = [np.array(
                    [[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)]
            kept_boxes = []
            for crop, box in zip(self._crop(source, boxes), boxes):
                if crop is None or getattr(crop, "size", 0) == 0:
                    continue
                if crop.shape[0] == 0 or crop.shape[1] == 0:
                    continue
                crops.append(crop)
                owners.append(index)
                kept_boxes.append(np.asarray(box))
            kept_per_image.append(kept_boxes)

        pages = [{"rec_texts": [], "rec_scores": [], "rec_polys": polys}
                 for polys in kept_per_image]
        if not crops:
            return pages

        texts, scores = self._recognize(crops)
        for owner, text, score in zip(owners, texts, scores):
            pages[owner]["rec_texts"].append(str(text))
            pages[owner]["rec_scores"].append(float(score))
        return pages

    def _run_batch_inference(self, images):
        try:
            pages = []
            # Luôn cụm 4 để detection là (4, 3, 128, 128), không phụ thuộc số biển.
            for start in range(0, len(images), self.FIXED_BATCH):
                pages.extend(self._infer_chunk(images[start:start + self.FIXED_BATCH]))
            if len(pages) != len(images):
                raise ValueError("Batch call trả về số lượng kết quả không khớp input")
            return pages
        except Exception as e:
            print(f"[LicensePlateOCR] Lỗi khi chạy batch OCR: {e}, fallback loop.")
            return [self._safe_single_call(img) for img in images]

    def _safe_single_call(self, image):
        try:
            pages = list(self.ocr.predict([image]))
            return pages[0] if pages else None
        except Exception as e:
            print(f"[LicensePlateOCR] Lỗi OCR 1 ảnh trong fallback loop: {e}")
            return None

    # ==================================================================
    # PARSE KẾT QUẢ OCR (det + rec) -> text, confidence
    # ==================================================================
    def _parse_single_result(self, ocr_result):
        if not ocr_result:
            return "", 0.0

        page_result = ocr_result[0] if isinstance(ocr_result, list) and len(ocr_result) == 1 else ocr_result

        items = self._parse_items(page_result)
        if not items:
            return "", 0.0

        rows = self._group_into_rows(items)

        final_text_parts = []
        confidences = []
        for row in rows:
            row.sort(key=lambda x: x["cx"])
            row_text = "".join(x["text"] for x in row)
            final_text_parts.append(row_text)
            confidences.extend([x["confidence"] for x in row])

        combined_text = "".join(final_text_parts)
        formatted_text = self.formatter.format(combined_text)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return formatted_text, avg_confidence

    def _parse_items(self, page_result):
        items = []

        if isinstance(page_result, dict):
            rec_texts = page_result.get('rec_texts', [])
            rec_scores = page_result.get('rec_scores', [])
            rec_polys = page_result.get('rec_polys', [])

            for i in range(len(rec_texts)):
                box = rec_polys[i] if i < len(rec_polys) else None
                if box is None or len(box) == 0:
                    continue
                items.append(self._box_to_item(box, rec_texts[i], rec_scores[i]))

        elif isinstance(page_result, list):
            for det in page_result:
                box = det[0]
                text = det[1][0]
                conf = det[1][1]
                items.append(self._box_to_item(box, text, conf))

        return items

    def _box_to_item(self, box, text, confidence):
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        return {
            "cx": sum(xs) / len(xs),
            "cy": sum(ys) / len(ys),
            "height": max(ys) - min(ys),
            "text": text,
            "confidence": confidence
        }

    def _group_into_rows(self, items):
        items.sort(key=lambda x: x["cy"])

        rows = []
        current_row = [items[0]]
        avg_height = items[0]["height"] or 1

        for item in items[1:]:
            if abs(item["cy"] - current_row[-1]["cy"]) < (avg_height * 0.3):
                current_row.append(item)
                avg_height = sum(x["height"] for x in current_row) / len(current_row)
            else:
                rows.append(current_row)
                current_row = [item]
                avg_height = item["height"] or 1

        rows.append(current_row)
        return rows   

        
        


