# Nguồn dữ liệu và điều khoản sử dụng

Ngày kiểm tra: 17/10/2026. Phạm vi bản này: nghiên cứu/đồ án giáo dục, không thương mại. Điều khoản dữ liệu độc lập với giấy phép mã nguồn dự án.

| Nguồn chính thức | Quyết định | Điều khoản đã kiểm tra |
| --- | --- | --- |
| [AIRTLab](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/tree/1f7747e104301ccaa82ef5a2f6804b51ced1c398) | Dùng video non-violent → normal; violent → violence | Dataset Release Agreement cho phép nghiên cứu và giáo dục, yêu cầu trích dẫn. Không gán nhãn MIT/CC cho video; không suy ra quyền thương mại hay tái phân phối. |
| [UR Fall Detection](https://fenix.ur.edu.pl/mkepski/ds/uf.html) | Dùng preview MP4 camera 0; ADL → normal; fall → fall | CC BY-NC-SA 4.0, mục đích học thuật phi thương mại. Giữ nguồn, trích dẫn, nêu thay đổi; nếu chia sẻ bản phái sinh phải tuân thủ ShareAlike. |
| [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) | Không tải trong bản này | Nguồn tác giả hiện không cung cấp video vì quyền riêng tư; hạn chế sửa đổi/tái phân phối khi chưa được đồng ý. Không thay bằng bản mirror không rõ quyền. |

AIRTLab có 350 clip từ hai camera. Cùng nhãn và số clip ở hai camera là cùng sự kiện; danh mục ghép chúng bằng group_id. Nguồn có hành vi bình thường dễ nhầm với bạo lực. Bộ này được diễn trong một phòng, không phải video lớp học.

UR Fall có 30 chuỗi ngã và 40 ADL. Bản này chỉ lấy camera 0, không coi hai camera là hai sự kiện độc lập. MP4 preview gồm depth bên trái và RGB bên phải; pipeline cắt nửa RGB, bỏ âm thanh, letterbox và lấy mẫu lại thời gian. Preview có thể nén mất chi tiết; nếu cần RGB gốc, tải PNG RGB và đồng bộ thời gian từ cùng trang nguồn trong một phiên bản dữ liệu mới.

## Dấu vết nguồn

- data/metadata/catalog.jsonl lưu URL từng video, nhãn nguồn, camera, group_id, giấy phép.
- AIRTLab được ghim commit 1f7747e104301ccaa82ef5a2f6804b51ced1c398; kiểm tra kích thước và SHA-1 Git blob với nguồn trước khi chấp nhận tải.
- UR Fall không công bố checksum tại trang đã kiểm tra. SHA-256 tính sau tải xác định bản cục bộ, không chứng minh file khớp một checksum do tác giả ký.
- SHA-256 video gốc và đã xử lý có trong manifest. Giữ video, manifest và events.csv trong kho riêng; .gitignore loại video khỏi Git.
- Không đưa video AIRTLab vào gói mã nguồn công khai. Tải lại trực tiếp bằng catalog; việc tải công khai không mặc nhiên cấp mọi quyền sử dụng.

## Trích dẫn trong báo cáo

Bianculli, M., Falcionelli, N., Sernani, P., Tomassini, S., Contardo, P., Lombardi, M., & Dragoni, A. F. (2020). A dataset for automatic violence detection in videos. Data in Brief, 33, 106587. https://doi.org/10.1016/j.dib.2020.106587

Kwolek, B., & Kepski, M. (2014). Human fall detection on embedded platform using depth maps and wireless accelerometer. Computer Methods and Programs in Biomedicine, 117(3), 489–501. Theo yêu cầu tại trang UR Fall chính thức.

[Điều khoản CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
