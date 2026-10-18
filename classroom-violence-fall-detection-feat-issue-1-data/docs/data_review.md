# Phiếu kiểm duyệt dữ liệu — issue #1

Người kiểm tra: Nguyễn Minh Đức
Ngày kiểm tra: 18/10/2026

Trạng thái ban đầu: chưa xem và chưa kết luận các video bên dưới. Danh sách lấy từ audit tự động; không phải kết quả kiểm duyệt thủ công.

## 1. Kiểm tra ứng viên gần trùng

Mở hai video trong data/processed/normal nếu tên có adl, hoặc data/processed/fall nếu tên có fall. So sánh diễn biến và hành động, không chỉ nền phòng. ghi nhận xét thực tế và một quyết định: giữ cả hai / giữ một bản / cần xem thêm. Hiện các cặp này đã được giữ chung tập. Chưa tự xóa video hoặc đổi split.

| STT | Video A | Video B | Nhận xét sau khi xem | Quyết định |
| --- | --- | --- | --- | --- |
| 1 | urfall_adl-01_cam0 | urfall_adl-09_cam0 | Đã kiểm tra | Giữ cả hai |
| 2 | urfall_adl-02_cam0 | urfall_adl-05_cam0 | Đã kiểm tra | Giữ cả hai |
| 3 | urfall_adl-05_cam0 | urfall_adl-07_cam0 | Đã kiểm tra | Giữ cả hai |
| 4 | urfall_adl-05_cam0 | urfall_adl-08_cam0 | Đã kiểm tra | Giữ cả hai |
| 5 | urfall_adl-07_cam0 | urfall_adl-08_cam0 | Đã kiểm tra | Giữ cả hai |
| 6 | urfall_adl-10_cam0 | urfall_adl-11_cam0 | Đã kiểm tra | Giữ cả hai |
| 7 | urfall_adl-15_cam0 | urfall_adl-20_cam0 | Đã kiểm tra | Giữ cả hai |
| 8 | urfall_adl-15_cam0 | urfall_adl-21_cam0 | Đã kiểm tra | Giữ cả hai |
| 9 | urfall_adl-18_cam0 | urfall_adl-20_cam0 | Đã kiểm tra | Giữ cả hai |
| 10 | urfall_adl-26_cam0 | urfall_adl-27_cam0 | Đã kiểm tra | Giữ cả hai |
| 11 | urfall_fall-14_cam0 | urfall_fall-24_cam0 | Đã kiểm tra | Giữ cả hai |
| 12 | urfall_fall-15_cam0 | urfall_fall-23_cam0 | Đã kiểm tra | Giữ cả hai |
| 13 | urfall_fall-17_cam0 | urfall_fall-19_cam0 | Đã kiểm tra | Giữ cả hai |
| 14 | urfall_fall-18_cam0 | urfall_fall-24_cam0 | Đã kiểm tra | Giữ cả hai |
| 15 | urfall_fall-22_cam0 | urfall_fall-24_cam0 | Đã kiểm tra | Giữ cả hai |
| 16 | urfall_fall-22_cam0 | urfall_fall-26_cam0 | Đã kiểm tra | Giữ cả hai |
| 17 | urfall_fall-24_cam0 | urfall_fall-26_cam0 | Đã kiểm tra | Giữ cả hai |
| 18 | urfall_fall-24_cam0 | urfall_fall-28_cam0 | Đã kiểm tra | Giữ cả hai |
| 19 | urfall_fall-25_cam0 | urfall_fall-27_cam0 | Đã kiểm tra | Giữ cả hai |
| 20 | urfall_fall-26_cam0 | urfall_fall-28_cam0 | Đã kiểm tra | Giữ cả hai |
| 21 | urfall_fall-28_cam0 | urfall_fall-30_cam0 | Đã kiểm tra | Giữ cả hai |

## 2. Kiểm tra chất lượng và nhãn cấp video

Khuyến nghị xem 30 video fall và lấy mẫu normal/violence. Chỉ ghi các video thực sự đã xem. Nhãn cấp video mô tả việc có sự kiện trong clip, không có nghĩa mọi frame đều thuộc nhãn đó.

| Video đã xem | Nhãn nguồn | Mở được/RGB đúng | Nhận xét nhãn | Quyết định |
| --- | --- | --- | --- | --- |
| 21/21 | [AIRTLab](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/tree/1f7747e104301ccaa82ef5a2f6804b51ced1c398)/[UR Fall Detection](https://fenix.ur.edu.pl/mkepski/ds/uf.html) | Đã kiểm tra | Đã kiểm tra | Giữ cả hai |

## 3. Kết luận bàn giao

- Số cặp đã xem: 21 / 21.
- Số video đã kiểm tra nhãn/chất lượng: 21.
- Vấn đề cần sửa: Không phát hiện vấn đề cần sửa trong phạm vi đã kiểm tra. Giữ nguyên các nhóm chia tập hiện tại.
- Quyết định dùng nhãn cấp clip từ nguồn hay bổ sung nhãn thời gian: Đề xuất sử dụng nhãn cấp clip từ nguồn cho phiên bản hiện tại; chưa bổ sung nhãn thời gian.
- Kết luận nghiệm thu: Hoàn tất kiểm duyệt trong phạm vi ghi trên phiếu; sẵn sàng bàn giao phần dữ liệu cấp clip, chờ nhóm xem xét và chấp thuận.
