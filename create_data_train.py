import cv2, os, sys

def extract(video_path, out_dir, every_n=5, blur_thresh=50.0):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            # bỏ frame mờ (motion blur) bằng phương sai Laplacian
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() >= blur_thresh:
                cv2.imwrite(f"{out_dir}/{name}_{idx:06d}.jpg", frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
        idx += 1
    cap.release()
    print(f"{video_path}: {saved} frame đã lưu / {idx} frame tổng")

if __name__ == "__main__":
    videos = [
        r"D:\license_plate_data\data\Camera Nhìn Bảng Số Xe Vào Ban Đêm - 0938112399.mp4",
        r"D:\license_plate_data\data\Camera Đọc Biển Số Xe, Giải pháp camera nhìn biển số xe của nhà imou.mp4",
    ]
    for v in videos:
        extract(v, "frames", every_n=6)