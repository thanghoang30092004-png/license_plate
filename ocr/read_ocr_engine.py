from ocr.license_plate_ocr import LicensePlateOCR
import cv2 


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

image = cv2.imread(r"D:\license_plate_data\data\image\c9e8434c-2c69-4d4c-aedc-c55257eeb285-plate.jpg")
results = PlateOCR(
    onnx_model_path=r"D:\license_plate_data\model\cct_s_v2_global.onnx",
    plate_config_path=r"D:\license_plate_data\model\cct_s_v2_global_plate_config.yaml",
    min_confidence=0.7,
    max_batch_cap=32,
    use_gpu=True
).read_batch([image])
print(results)