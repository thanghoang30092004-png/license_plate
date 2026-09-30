from ocr.license_plate_ocr import LicensePlateOCR



class PlateOcr:
    def __init__(self, use_gpu=True, min_confidence=0.6, batch_size=16):
        self.ocr = LicensePlateOCR(
            use_gpu=use_gpu,
            min_confidence=min_confidence,
            batch_size=batch_size)
    

    def read_batch(self,crop_images):
        return self.ocr.read_batch(crop_images)
