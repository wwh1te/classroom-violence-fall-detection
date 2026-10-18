# YOLO Pose + Tracking và baseline té ngã

Pipeline độc lập tại `src/pose/pipeline.py` đọc video đã chuẩn hóa trong `data/processed` mặc định. Có thể truyền một video hoặc thư mục video khác qua `--source`. Pipeline không sửa video nguồn, manifest hay annotation hiện có.

## Chạy

```powershell
python -m pip install -r requirements-pose.txt
python -m src.pose.pipeline
```

Để chạy trên video/thư mục riêng hoặc GPU:

```powershell
python -m src.pose.pipeline --source data/processed/fall --device 0
python -m src.pose.pipeline --source data/raw/fall/example.mp4 --device cpu
```

Mô hình mặc định `yolo11n-pose.pt` và tracker `bytetrack.yaml`; Ultralytics sẽ tải checkpoint nếu chưa có. Đổi model/tracker, confidence hoặc cấu hình baseline bằng các cờ `--model`, `--tracker`, `--confidence`, `--min-fall-frames`, `--fall-keypoint-confidence`, `--max-torso-angle`, `--min-box-aspect-ratio`. Chọn `--device cpu` nếu không dùng GPU CUDA.

## Kết quả và định dạng

Mỗi lần chạy tạo thư mục mới có tên duy nhất dưới `outputs/pose_tracking/`. Kết quả không ghi đè lần chạy trước. `run.json` ghi cấu hình, video, số frame, số detection, trạng thái chạy và tổng số person-frame bị baseline gắn cờ. Mỗi video có một `frames.jsonl`, một record cho mỗi frame:

- `frame_index`, `timestamp_seconds`, kích thước frame.
- `people`: người phát hiện được, `track_id` (có thể null nếu tracker chưa cấp ID), bbox pixel `xyxy` và bbox chuẩn hóa trong `[0, 1]`.
- `keypoints`: 17 điểm COCO theo thứ tự chuẩn `nose`, mắt, tai, vai, khuỷu, cổ tay, hông, gối, cổ chân. `x/y` chuẩn hóa theo chiều rộng/cao của frame; `confidence` là confidence keypoint. Giá trị tọa độ có thể null nếu mô hình không trả điểm tương ứng.
- `fall_baseline`: ứng viên, trạng thái sau khi đủ số frame liên tiếp, độ dài streak và góc thân người.

Nếu lỗi xảy ra, lệnh trả mã khác 0 và `run.json` được đánh dấu `failed`; thư mục chạy dang dở được giữ lại để chẩn đoán, không bị xóa âm thầm.

## Baseline té ngã và giới hạn

Đây là heuristic, không phải bộ phân loại học từ nhãn: xác định điểm vai/hông đủ confidence, góc thân (đường trung điểm vai tới hông) gần ngang, bbox rộng hơn cao, rồi yêu cầu tư thế tồn tại liên tiếp mặc định 5 frame trên cùng Track ID. Ngưỡng angle mặc định 35 độ, aspect ratio tối thiểu 1.0 và confidence keypoint tối thiểu 0.35. Tùy chọn có thể đổi qua CLI.

Baseline có thể nhầm người cúi, nằm có chủ đích, góc camera thấp hoặc tracker đổi ID; cũng có thể bỏ sót té ngã bị che khuất. `fall_detected` chỉ là cảnh báo để xem lại, không phải ground truth. Dữ liệu hiện có cung cấp nhãn cấp clip, chưa có thời điểm té ngã theo frame/người, vì vậy pipeline chỉ trích xuất và tạo dự đoán; chưa tính precision/recall hay khẳng định chất lượng. Cần annotation sự kiện được duyệt độc lập để đánh giá.
