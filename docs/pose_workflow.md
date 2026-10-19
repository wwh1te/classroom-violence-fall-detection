# YOLO Pose và baseline té ngã

## Cài đặt

Trong môi trường ảo của project, cài các dependency dùng cho pipeline video và YOLO:

```powershell
python -m pip install -r requirements-pose.txt
```

`requirements-pose.txt` cài Ultralytics cùng OpenCV/NumPy. PyTorch được cài theo dependency của Ultralytics; để dùng NVIDIA GPU, hãy cài phiên bản PyTorch/CUDA phù hợp trước theo hướng dẫn chính thức của PyTorch. Kiểm tra:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

Project không lưu weight vào thư mục dữ liệu. Mặc định Ultralytics dùng `yolo11n-pose.pt` và tải weight khi chạy lần đầu nếu máy chưa có cache; có thể trỏ `--model` đến file weight đã được cấp phép/lưu cục bộ.

## Chạy một video

```powershell
python -m src.pose.pipeline --source data\processed\fall\<sample_id>.mp4
```

Video nguồn là chỉ đọc. Mỗi lần chạy tạo một thư mục mới trong `outputs/pose/`, gồm:

- `poses.jsonl`: một dòng mỗi frame, danh sách người, track ID, bounding box theo pixel và tọa độ chuẩn hóa theo frame, keypoint COCO theo pixel/frame/bbox, cùng kết quả baseline.
- `annotated.mp4`: video minh họa bounding box, skeleton, ID và cảnh báo `FALL?`.
- `summary.json`: thông số chạy, số frame và số phát hiện.

Tên thư mục dùng ID video, hash đường dẫn và timestamp UTC; file kết quả không ghi vào `data/raw`, `data/processed` hoặc metadata. Có thể chọn `--output-dir` khác. Nếu dùng `--run-id`, ID phải mới; chạy trùng thư mục sẽ báo lỗi thay vì ghi đè.

## Dùng manifest của pipeline dữ liệu

Sau khi hoàn tất `python -m src.data.pipeline download` và `prepare`, pipeline có thể xử lý video chuẩn hóa từ manifest; tùy chọn `--split` lọc dữ liệu:

```powershell
python -m src.pose.pipeline --manifest data\metadata\local\manifest.jsonl --split test
```

Bỏ `--split` để chạy toàn bộ manifest. Dữ liệu chỉ có metadata trong repository ban đầu; cần tải và chuẩn bị video trước khi chạy lệnh trên. Để xử lý trực tiếp video khác, dùng `--source`.

Tham số mặc định nằm trong `configs/pose.json`; các tùy chọn bổ sung gồm `--model`, `--device cpu` hoặc `--device 0`, `--root`, `--output-dir`, `--config`. Không chạy lên cả train/val/test để báo cáo đánh giá nếu đang dùng cùng video đó cho huấn luyện hoặc lựa chọn ngưỡng.

## Cách diễn giải baseline

YOLO Pose COCO cung cấp tối đa 17 keypoint/người; ByteTrack cấp ID theo chuỗi frame. Keypoint được lưu đồng thời ở tọa độ pixel, tỷ lệ khung hình và tỷ lệ trong bounding box. Góc thân được tính từ trung điểm vai tới trung điểm hông theo tọa độ pixel để tránh làm méo góc khi bbox không vuông.

Baseline gắn cờ một frame khi bbox đủ ngang (`width/height >= 1.25`) và thân nghiêng ít nhất 50° so với phương thẳng đứng; bốn keypoint vai/hông phải đạt confidence tối thiểu 0.25. Cờ chỉ được xác nhận sau 3 frame liên tiếp của cùng track. Có thể điều chỉnh các ngưỡng trong `configs/pose.json`.

Đây là luật hình học để tạo baseline/candidate cần con người xem lại, không phải confidence xác suất, bộ phân loại đã huấn luyện, ground truth hay cảnh báo an toàn đã được kiểm định. Tư thế nằm/ngồi, khuất người, đổi ID hoặc góc camera có thể gây bỏ sót/báo nhầm. Nhãn hiện có trong manifest là nhãn cấp clip, không chỉ ra thời điểm hay người bị ngã; vì vậy không thể dùng nó để tuyên bố độ chính xác theo frame/người. Cần gán nhãn sự kiện và đánh giá trên split độc lập trước khi dùng cho nghiên cứu hoặc triển khai.
