
import os

def _preload_cuda():
    """Đưa DLL CUDA vào process trước khi onnxruntime mở GPU, tránh lỗi
    load DLL khi cùng máy có cài thêm torch."""
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

import numpy as np
import cv2
from fast_plate_ocr import LicensePlateRecognizer

from ocr.process_ocr_text import PlateFormatter


class LicensePlateOCR:
    def __init__(self, onnx_model_path, plate_config_path,
                 min_confidence=0.6, max_batch_cap=32, use_gpu=True):
        """
        Args:
            onnx_model_path: đường dẫn file .onnx của model.
            plate_config_path: đường dẫn file .yaml config đi kèm model.
            min_confidence: ngưỡng confidence tối thiểu để chấp nhận kết quả.
            max_batch_cap: giới hạn TRẦN an toàn cho 1 lần gọi run() — chỉ
                           chia nhỏ khi số ảnh vượt ngưỡng này (tránh tràn
                           VRAM lúc có quá nhiều xe trong 1 frame), không
                           phải nhịp gọi cố định mỗi lần.
        """
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if not isinstance(max_batch_cap, int) or max_batch_cap < 1:
            raise ValueError("max_batch_cap must be a positive integer")
        self.min_confidence = min_confidence
        self.max_batch_cap = max_batch_cap
        self.formatter = PlateFormatter()

        try:
            self.recognizer = LicensePlateRecognizer(
                onnx_model_path=onnx_model_path,
                plate_config_path=plate_config_path,
                providers=(["CUDAExecutionProvider", "CPUExecutionProvider"]
                           if use_gpu else ["CPUExecutionProvider"]),
            )
        except Exception as e:
            raise RuntimeError(
                f"Không khởi tạo được LicensePlateRecognizer. "
                f"Kiểm tra lại đường dẫn model/config: "
                f"{onnx_model_path}, {plate_config_path}. Lỗi gốc: {e}"
            )

        self._warmup()

    def _warmup(self):
        """Chạy thử 1 lần lúc khởi tạo để onnxruntime khóa shape ngay,
        tránh lần gọi đầu tiên trong vòng lặp thật bị chậm bất thường."""
        try:
            blank = np.full((48, 160, 3), 255, dtype=np.uint8)
            self.recognizer.run([blank], return_confidence=True)
        except Exception as e:
            print(f"[LicensePlateOCR] Cảnh báo: warmup thất bại ({e}), "
                  f"lần gọi OCR đầu tiên thực tế có thể chậm hơn bình thường.")

    # ==================================================================
    # BATCH API
    # ==================================================================
    def read_batch(self, crop_images, min_confidence=None, max_batch_cap=None):
        """
        Args:
            crop_images (list[np.ndarray]): ảnh crop biển số, mỗi ảnh (H, W, C).

        Returns:
            list[dict]: {'text', 'confidence', 'valid'}, đúng thứ tự crop_images.
            'text' đã được ép về format biển số Việt Nam.
        """
        threshold = min_confidence if min_confidence is not None else self.min_confidence
        cap = max_batch_cap if max_batch_cap is not None else self.max_batch_cap
        if not isinstance(cap, int) or cap < 1:
            raise ValueError("max_batch_cap must be a positive integer")
        if not 0 <= threshold <= 1:
            raise ValueError("min_confidence must be between 0 and 1")

        if not crop_images:
            return []

        valid_indices = []
        valid_images = []
        for i, img in enumerate(crop_images):
            if img is not None and img.size > 0:
                valid_indices.append(i)
                valid_images.append(self._prepare_image(img))

        output = [{'text': '', 'confidence': 0.0, 'valid': False} for _ in crop_images]

        if not valid_images:
            return output

        raw_predictions = self._run_batch_inference(valid_images, cap)

        for local_idx, pred in enumerate(raw_predictions):
            original_idx = valid_indices[local_idx]
            text, confidence = self._parse_single_result(pred)

            valid = self.formatter.is_valid(text) and confidence >= threshold
            output[original_idx] = {
                'text': text if valid else '',
                'confidence': confidence,
                'valid': valid
            }

        return output

    def _prepare_image(self, image):
        """OpenCV crops are BGR; recognizer arrays must match the model color mode."""
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            bgr = image
        else:
            raise ValueError(f"Unsupported crop shape: {image.shape}")
        if self.recognizer.config.image_color_mode == "grayscale":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _run_batch_inference(self, images, cap):
        """Gọi run() MỘT LẦN cho toàn bộ list khi trong giới hạn an toàn
        — đây là batch thật, gộp thành 1 tensor, 1 lần forward pass.
        Chỉ chia nhỏ khi vượt `cap` để tránh tràn VRAM."""
        try:
            if len(images) <= cap:
                predictions = self.recognizer.run(images, return_confidence=True)
            else:
                predictions = []
                for start in range(0, len(images), cap):
                    chunk = images[start:start + cap]
                    predictions.extend(self.recognizer.run(chunk, return_confidence=True))

            if len(predictions) != len(images):
                raise ValueError("Batch call trả về số lượng kết quả không khớp input")
            return predictions

        except Exception as e:
            print(f"[LicensePlateOCR] Lỗi khi chạy batch OCR: {e}, fallback loop.")
            return [self._safe_single_call(img) for img in images]

    def _safe_single_call(self, image):
        try:
            preds = self.recognizer.run([image], return_confidence=True)
            return preds[0] if preds else None
        except Exception as e:
            raise RuntimeError("OCR inference failed, including single-image fallback") from e

    # ==================================================================
    # PARSE KẾT QUẢ -> text (đã ép format VN), confidence
    # ==================================================================
    def _parse_single_result(self, pred):
        if pred is None:
            return "", 0.0

        raw_text = pred.plate
        if not raw_text:
            return "", 0.0

        # FastPlateOCR returns probabilities for all slots, including trailing padding.
        # Confidence is the mean over actual characters, excluding padding slots.
        probs = pred.char_probs
        if probs is None:
            return "", 0.0
        probs = np.asarray(probs, dtype=float).reshape(-1)[:len(raw_text)]
        if len(probs) != len(raw_text) or not np.isfinite(probs).all():
            return "", 0.0
        confidence = float(np.mean(probs))
        if not 0 <= confidence <= 1:
            return "", 0.0

        formatted_text = self.formatter.format(raw_text)
        return formatted_text, float(confidence)
