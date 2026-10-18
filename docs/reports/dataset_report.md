# Mô tả dữ liệu và kết quả chuẩn bị

Ngày chạy: 17/10/2026. Số liệu dưới đây đo trên các file đã tải và giải mã thành công, không phải số lượng mục tiêu.

Bộ dữ liệu khởi đầu gồm **420 video**, thuộc **245 nhóm bản ghi** từ AIRTLab và UR Fall. Video được ánh xạ sang ba nhãn normal, violence, fall theo nhãn cấp clip do nguồn cung cấp. Nguồn và điều khoản được ghi tại [dataset_sources.md](../dataset_sources.md).

| Nguồn / lớp | Video |
| --- | ---: |
| airtlab/normal | 120 |
| airtlab/violence | 230 |
| urfall/fall | 30 |
| urfall/normal | 40 |

Dung lượng gốc: 1.116 GiB; dung lượng chuẩn hóa: 0.290 GiB. Tổng thời lượng bản xử lý: 39.59 phút, 35,629 frame. Thời lượng tính theo từng video/góc quay, không phải thời gian của các sự kiện duy nhất.

Video được giải mã tuần tự, kiểm số frame, rồi chuẩn hóa thành RGB 640×360 ở 15 fps bằng letterbox. Với UR Fall MP4 preview, chỉ lấy nửa RGB bên phải. Không đưa ảnh depth hoặc âm thanh vào đặc trưng. Bản MP4 đầu ra được đọc lại để kiểm số frame và kích thước.

## Chia tập và kiểm tra

Chia theo nhóm sự kiện, kết nối cả video gốc, bản trùng byte và toàn bộ ứng viên gần trùng bằng connected components, phân tầng theo nguồn/lớp, seed 42 và tỷ lệ mục tiêu 70/15/15. Hai camera AIRTLab cùng sự kiện luôn đi chung. Tỷ lệ thực tế có làm tròn theo nhóm.

| Tập | normal | violence | fall | Tổng |
| --- | ---: | ---: | ---: | ---: |
| train | 113 | 162 | 23 | 298 |
| val | 24 | 34 | 3 | 61 |
| test | 23 | 34 | 4 | 61 |

Audit ghi nhận 0 lỗi xác định; 21 cặp ứng viên gần trùng, trong đó 0 cặp xuyên tập. Chi tiết nằm trong data/metadata/local/audit.json. Ứng viên dHash cần xem lại; không xem số lỗi bằng 0 là bằng chứng độc lập người/bối cảnh.

## Giới hạn và trạng thái nghiệm thu

- Có 420 video chưa được người gán nhãn duyệt lại. Nhãn hiện tại là nhãn cấp clip của nguồn, không phải ground truth từng frame/người.
- events.csv là template pending. Chưa đo thời điểm sự kiện, đồng thuận giữa người gán nhãn hoặc độ chính xác nhãn thời gian. Chỉ có hướng dẫn và quy trình duyệt đã xây dựng.
- Chưa xác lập split độc lập người tham gia hoặc bối cảnh vì thiếu metadata phù hợp. Không dùng kết quả từ split này để tuyên bố tổng quát sang người hoặc lớp học mới.
- Dữ liệu không đại diện riêng lớp học. Nguồn và lớp có tương quan: violence đến từ AIRTLab, fall từ UR Fall; mô hình có thể học nền/nguồn thay vì hành động. Cần bổ sung dữ liệu cùng bối cảnh cho cả ba lớp và đánh giá trên lớp học được đồng thuận.
- UR Fall dùng video preview có nén. Không so trực tiếp với nghiên cứu dùng PNG RGB gốc hoặc toàn bộ camera mà không ghi khác biệt.
- ready_for_final_evaluation = false. Chưa thể coi bộ sự kiện là nhãn đánh giá cuối.

Mã nguồn, nguồn dữ liệu, tải/chuẩn hóa, hướng dẫn nhãn và kiểm tra tự động đã có. Trước khi đóng issue, nhóm phải xác nhận phạm vi nghiệm thu nhãn cấp clip, xem lại ứng viên gần trùng xuyên tập; nếu dùng nhãn thời gian trong báo cáo, hoàn tất gán nhãn và kiểm tra độc lập.

## Tái lập

Chạy download → prepare → audit theo docs/data_workflow.md, sau đó python -m src.data.report. data/metadata/splits.csv lưu split và checksum để đối chiếu phiên bản. dataset_statistics.json lưu số liệu và checksum manifest/catalog. Mã kiểm thử không dùng video thật làm fixture.

Trích dẫn dataset và điều khoản được ghi trong [dataset_sources.md](../dataset_sources.md).
