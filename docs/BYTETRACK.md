# Pipeline ByteTrack

Chạy trong môi trường `plate312` từ thư mục gốc dự án:

```powershell
python -m pipeline.run_realtime --no-display
```

Pipeline dùng `tracking/bytetrack_tracker.py`. DeepSORT được giữ lại cho các
script cũ, nhưng pipeline hiện tại không import hoặc khởi tạo DeepSORT.
Đã kiểm tra với Ultralytics **8.4.158**, lap **0.5.13**, Python **3.12**.
Validation dùng thêm motmetrics **1.4.0**; các gói này đã có trong môi trường
được kiểm thử. Lớp adapter dùng API `track_class` của phiên bản Ultralytics này.

## Cấu hình

`config/config.yaml` dùng `tracking.type: bytetrack` và các ngưỡng high/low/new,
`track_buffer`, `match_thresh`, `fuse_score`. Ngưỡng detector xe mặc định 0.1,
không được lớn hơn `track_low_thresh`, để giữ detection yếu cho bước ghép thứ hai.
Detection yếu có thể duy trì track hiện tại, nhưng không tạo track mới hoặc
khôi phục track đã Lost theo cách của ByteTrack đang dùng.

`track_buffer: 30` tính theo **30 lần update**, kể cả update không có detection;
không tính bằng số frame nguồn đã bị FrameReader bỏ qua. Lịch OCR vẫn tính theo
chỉ số frame nguồn như trước.

## Vòng đời cache và kết quả

- `update(detections, frame)` nhận N hàng `[x1,y1,x2,y2,score,class_id]` và trả
  các track hiện đang được quan sát, đã kích hoạt. Mỗi track có `track_id` dạng
  chuỗi, `to_ltrb()`, `confidence` của detection và `class_id`.
- `retained_track_ids()` gồm Tracked và Lost. Worker gọi `retain_tracks()` để
  giữ nguyên biển số và cờ snapshot khi track còn trong buffer; xóa khi tracker
  loại ID. Cache không dùng bộ đếm absent_grace riêng cho ByteTrack.
- Các đối tượng Removed còn sót trong danh sách Lost của thư viện bị loại khỏi
  association. Mỗi tracker có bộ đếm ID riêng để các phiên không reset lẫn nhau.
- Mỗi xe chỉ có một crop OCR/một phiếu trong một frame, chọn detection biển số
  có confidence cao nhất. Crop snapshot là toàn xe, lấy biển số theo đúng ID.

SnapshotSaver giữ nguyên queue giới hạn và chính sách bỏ ảnh chờ cũ nhất khi
đầy; có bộ đếm `dropped`. Khi đang ghi/chờ một ID, không xếp thêm ảnh cùng ID.
Ghi lỗi không gọi callback thành công. `stop()` đóng nhận việc và xử lý hết
queue; callback lỗi được ghi log và không giết thread. Main vẫn giới hạn chờ
thread 10 giây; nếu I/O treo quá lâu sẽ báo không dừng được thread.

Ảnh cùng biển số ở **ID mới** hoặc **phiên chạy mới** vẫn có thể được lưu.
Chưa áp dụng chống trùng biển số xuyên ID, vì xe quay lại và OCR trùng cần chính
sách riêng. Snapshot summary ghi số ảnh saved/failed/dropped/callback_errors.

## Kiểm thử

```powershell
python -m unittest discover -s tests -p test_bytetrack_pipeline.py -v
python -m unittest discover -s tests -p "test_snapshot*.py" -v
python tests/validate_bytetrack.py --config config/config.yaml
```

Lệnh cuối chạy các tình huống giả lập có nhãn và toàn bộ video với model thật.
Video, snapshot, log và `report.json` được ghi vào thư mục mới
`output/bytetrack-validation-...`; không ghi đè output cũ hay sửa config gốc.
Có thể dùng `--synthetic-only` để bỏ phần model/video thật.

Các ca kiểm thử gồm: confidence yếu, detection rỗng, bbox lỗi/NaN/Inf, mất dấu
ngắn/dài, sát biên buffer, ID độc lập giữa hai tracker, 80 đối tượng, che khuất
một phần, cache khi tái xuất hiện, nhiều bbox biển số một frame, lỗi ghi ảnh,
callback lỗi, queue đầy và bốn producer gửi 300 ảnh.

## Giới hạn đã đo

ByteTrack này ghép theo chuyển động/IoU, không có ReID và không phân tách
association theo class. Xe giao cắt hoặc đổi hướng trong lúc bị che có thể
đổi ID. Vượt buffer thì ID mới là hành vi dự kiến. Sai ID có thể làm cache biển
số đi theo sai xe; bộ test logic không chứng minh tránh được mọi tình huống đó.

Trong validation ngày 2026-10-06:

- Giao cắt chuyển động đều: 0 ID switch trong kịch bản giả lập.
- Che khuất 10 update: 0 ID switch; không xuất bbox trong lúc Lost.
- Che khuất 40 update: 2 ID switch do vượt buffer 30.
- Đổi hướng khi bị che: 2 ID switch, IDF1 khoảng 55.6%.
- 80 xe giả lập chuyển động đơn giản: 0 ID switch, khoảng 3.25 ms/update.
- Video thật 450 frame: inference 359 frame, lưu 57 ảnh, không ghi lỗi;
  tracking trung bình 1.28 ms, P95 1.76 ms sau bỏ 10 frame khởi động.

Các số giả lập không phải độ chính xác video thực tế. Chưa có nhãn bbox/ID
cho video thực để tính IDF1/MOTA đáng tin cậy. FrameReader vẫn bỏ frame khi
chậm; video xuất lặp frame để giữ thời lượng nguồn. Không so trực tiếp tốc độ
với lần chạy DeepSORT trước nếu model/detection/frame đầu vào khác nhau.

Test cũ `test_detect_plate_frame_log.py` hiện không import được vì file
`detect_plate_video.py` không còn trong workspace; đây là lỗi bộ test cũ độc lập
với ByteTrack, chưa sửa hoặc khôi phục script đó trong lần chuyển tracker này.
