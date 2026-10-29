# Thực hiện issue #1 — Dữ liệu

## Chuẩn hóa video có sẵn trong `data/raw`

Nếu đã có video trong `data/raw` và chỉ cần chuẩn hóa để YOLO xử lý, không cần catalog hoặc manifest. Từ thư mục gốc repository, chạy:

```powershell
python -m src.data.pipeline normalize-raw
```

Pipeline quét đệ quy video `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.wmv` và `.webm`; đầu ra MP4 theo cấu hình `configs/data.json` được lưu dưới `data/processed/raw/`, giữ nguyên cấu trúc thư mục tương đối. Mặc định là 640×360, 15 FPS, letterbox nền đen và không có âm thanh. Video nguồn chỉ được đọc, không bị thay đổi. Kết quả và lỗi lần lượt được ghi vào `data/metadata/local/raw_normalization.jsonl` và `raw_normalization_errors.jsonl`; có lỗi thì lệnh trả mã khác 0. Đầu ra đã tồn tại không bị ghi đè; dùng `--output-dir data\processed\raw_v2` để chạy vào thư mục mới.

Chế độ này không tự suy đoán nhãn từ tên file/thư mục và không tạo split hay manifest dataset. Dùng quy trình bên dưới nếu cần catalog, nhãn, checksum nguồn và chia train/val/test.

## Chạy từ thư mục gốc repository

Dùng Python 3.10/3.11 như tài liệu dự án và một môi trường ảo. Chỉ cài dependencies dữ liệu để không cần tải mô hình/GPU:

```powershell
python -m pip install -r requirements-data.txt
python -m src.data.pipeline download
python -m src.data.pipeline prepare
python -m src.data.pipeline audit
```

Download lấy 420 video từ catalog đã ghim, khoảng 1,2 GB video gốc; chừa thêm dung lượng cho bản xử lý. Có retry, file .part, kiểm checksum AIRTLab, cache và giới hạn mặc định 64 MiB/file. Có thể tải lần lượt --dataset airtlab và --dataset urfall; prepare chỉ nên chạy sau khi đủ catalog. --workers 2 giúp giảm tải máy; mặc định 4.

Không cần ffmpeg trên PATH: OpenCV giải mã và ghi MP4 (codec mp4v). Bản xử lý dùng 640×360, 15 fps, letterbox đen, không âm thanh. Không cắt clip theo cửa sổ cố định để giữ nguyên sự kiện và tránh gán nhãn sai phần trước/sau sự kiện. configs/data.json lưu cấu hình và seed 42. Giữ tọa độ skeleton theo khung hình đã letterbox trong issue #2.

## Kết quả

- data/raw/{normal,violence,fall}: video tải từ tác giả.
- data/processed/{normal,violence,fall}: video RGB chuẩn hóa.
- data/metadata/local/downloads.jsonl: kết quả tải, checksum nguồn.
- data/metadata/local/manifest.jsonl: đường dẫn, kích thước, FPS, thời lượng, số frame, checksum, nhãn, group, split.
- data/metadata/local/preparation_errors.jsonl: lỗi chuẩn hóa, phải rỗng trước khi nghiệm thu kỹ thuật.
- data/metadata/local/audit.json: lỗi rò rỉ, trùng byte, ứng viên gần trùng, thống kê và trạng thái duyệt nhãn.
- data/annotations/local/events.csv: template gán nhãn thời gian; pending không phải ground truth đã duyệt.
- docs/reports/: số liệu chạy thực tế và mô tả dùng cho báo cáo.

Lệnh trả exit code khác 0 khi lỗi tải/chuẩn hóa, checksum thay đổi, thiếu lớp hoặc rò rỉ xác định. Audit vẫn trả 0 khi chỉ có cảnh báo; phải đọc warnings và ready_for_final_evaluation. Cờ này kiểm tra nhãn cấp clip trong manifest; không thay thế nghiệm thu CSV sự kiện hoặc độc lập người/bối cảnh.

## Kiểm lỗi và chia dữ liệu

Giải mã tuần tự toàn video và đối chiếu số frame khai báo để phát hiện file rỗng, không mở được, ngắt giải mã và đổi kích thước giữa chừng. Kiểm lại bản đầu ra sau ghi. Pipeline lấy mẫu theo chỉ số frame dựa trên FPS nguồn; các nguồn lựa chọn là CFR. Nếu thêm video VFR, cần chuyển sang giải mã dựa trên timestamp trước khi dùng.

Connected components liên kết group_id, parent_video_id, SHA-256 gốc, SHA-256 bản xử lý và ứng viên gần trùng dHash. split_component_id ghi nhóm liên thông cuối để chia tập; group_id vẫn giữ sự kiện gốc. Gom ứng viên là biện pháp bảo thủ, không kết luận chúng là bản sao. Chia các nhóm theo từng nguồn/lớp, gần tỷ lệ 70/15/15; không chia frame. Cần ít nhất ba nhóm độc lập mỗi tầng. Đây là split theo recording, không phải benchmark độc lập người hoặc phòng.

Gần trùng dùng dHash của tối đa 5 frame phân bố đều, so cặp cùng độ dài trong sai số 0,25 giây, ngưỡng khoảng cách Hamming trung bình 4/64 bit. Đây là danh sách ứng viên phải xem lại: có thể nhầm do nền giống nhau và có thể bỏ sót video cắt, đổi tốc độ hoặc đổi góc. Không gọi thuật toán này là chứng minh không rò rỉ. Hai camera cùng sự kiện đã được nhóm bằng metadata nguồn, không dựa vào dHash.

## Kiểm thử

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

tests/test_data_pipeline.py tạo video tổng hợp tạm thời để kiểm crop, FPS, thời lượng, letterbox, video hỏng và checksum; kiểm riêng split, trùng, rò rỉ và bảo toàn nhãn do người nhập. Video tổng hợp kiểm thử không nằm trong dataset thực nghiệm.

## Điều kiện đóng issue

- Có nguồn và điều khoản kiểm tra được, trích dẫn đầy đủ.
- Tải đủ catalog, chuẩn hóa thành công, đủ ba lớp; lưu thống kê thực.
- Có hướng dẫn gán nhãn và template; nếu báo cáo dùng ground truth thời gian thì phải gán nhãn/duyệt thời gian trước.
- Không có lỗi rò rỉ xác định; xem lại ứng viên gần trùng xuyên split.
- Mô tả đúng hạn chế, không báo số liệu mục tiêu như kết quả đã đạt.


Checksum video đã xử lý có thể khác khi thay phiên bản OpenCV/codec hoặc hệ điều hành, dù nội dung frame tương đương. Đối chiếu checksum nguồn, catalog, cấu hình và môi trường trước khi so sánh kết quả tái lập.
