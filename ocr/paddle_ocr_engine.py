from ocr.license_plate_ocr import LicensePlateOCR


class PlateOCR:
    def __init__(self, onnx_model_path, plate_config_path,
                 min_confidence=0.7, max_batch_cap=32, use_gpu=True):
        self.engine = LicensePlateOCR(
            onnx_model_path=onnx_model_path,
            plate_config_path=plate_config_path,
            min_confidence=min_confidence,
            max_batch_cap=max_batch_cap,
            use_gpu=use_gpu,
        )

    def read_batch(self, crops):
        return self.engine.read_batch(crops)
