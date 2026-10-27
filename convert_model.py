from ultralytics import YOLO

# model detect xe
YOLO(r"D:\license_plate_data\model\last.pt").export(
    format="engine",
    half=True,        # FP16
    imgsz=640,        
    batch=1,
    simplify=True,
    device=0,
)
