# Bàn giao issue #1 — Dữ liệu

Đã tạo pipeline và chạy trên 420 video thật: 350 AIRTLab + 70 UR Fall camera 0.
Ba lớp: normal 160, violence 230, fall 30.
Video RGB chuẩn hóa: 640×360, 15 fps; 35.629 frame, khoảng 39,59 phút tính theo video/góc quay.
227 nhóm chia tập cuối từ 245 sự kiện nguồn, đã gom bảo thủ 21 cặp ứng viên gần trùng.
Không có lỗi giải mã/checksum/rò rỉ xác định; không còn ứng viên gần trùng xuyên tập.
21 kiểm thử tự động đạt trên Python 3.12.14, OpenCV 5.0.0, NumPy 2.5.4. Chưa chạy ma trận Python 3.10/3.11.

Đọc theo thứ tự:
1. docs/data_workflow.md — cách chạy, các file kết quả, tiêu chí đóng issue.
2. docs/dataset_sources.md — nguồn chính thức, điều khoản và trích dẫn.
3. docs/annotation_guide.md — nhãn và cách xác định mốc sự kiện.
4. docs/reports/dataset_report.md — phần mô tả dataset dùng cho báo cáo.
5. docs/reports/dataset_statistics.json và audit.json — số liệu thực, kiểm tra gần trùng.
6. data/metadata/splits.csv — split, nhóm và checksum.

Bản đầy đủ cục bộ có video tại data/raw, data/processed; manifest và template nhãn tại data/metadata/local và data/annotations/local. Không commit các thư mục này. Gói issue-1-source.zip chỉ chứa mã, catalog, split và tài liệu; không chứa video.

Trạng thái: chưa đóng issue GitHub và chưa push thay đổi. Nhãn cấp clip kế thừa từ nguồn, toàn bộ đang chờ kiểm duyệt của nhóm; events.csv chưa có ground truth thời gian. Không báo bộ này là dữ liệu lớp học thật hoặc split độc lập người/bối cảnh. Trước khi đánh giá cuối, kiểm duyệt nhãn theo mục tiêu thực nghiệm và giải quyết các ứng viên gần trùng (hiện đã giữ chung tập). Nếu dùng nhãn thời gian, hoàn tất CSV và người thứ hai kiểm tra.
