# Hướng dẫn gán nhãn hành động — phiên bản 1

## Đơn vị và ba lớp

Đơn vị nguồn là video gốc; đơn vị gán nhãn chi tiết là một khoảng thời gian trong video. Giữ nguyên sample_id và group_id trong mọi bản cắt, augmentation và chuỗi keypoint. Nhãn do dataset cung cấp chỉ là nhãn cấp clip, không phải nhãn cho mọi frame hoặc mọi người trong clip.

| Nhãn | Quy tắc quan sát |
| --- | --- |
| normal | Hành động sinh hoạt/giao tiếp, không quan sát thấy bạo lực hoặc chuyển tiếp ngã: đi, ngồi, cúi, ôm, vỗ tay, giơ tay. Nằm có chủ đích và đã nằm sẵn không tự động là fall. |
| violence | Quan sát hành vi tấn công hoặc mô phỏng tấn công: đấm, đá, tát, xô đẩy, vật lộn. Không suy đoán ý định từ một chuyển động nhanh đơn lẻ. |
| fall | Quan sát chuyển tiếp mất tư thế nâng đỡ từ đứng/ngồi xuống sàn hoặc mặt thấp. Phân biệt ngồi nhanh, quỳ, nhặt đồ, nằm có chủ đích. |

Nhãn mơ hồ dùng status=uncertain và ghi lý do; không thêm uncertain thành lớp huấn luyện thứ tư. Nếu fall và violence xảy ra chồng nhau, tạo hai dòng sự kiện, ghi cùng người và thời gian phù hợp. Với bài toán clip ba lớp loại trừ nhau, đánh dấu đoạn chồng nhãn để loại hoặc xử lý riêng; không ép chọn lớp theo thứ tự ưu tiên tùy ý.

## CSV và mốc thời gian

Dùng data/annotations/local/events.csv được tạo tự động. Template không ghi đè file đã tồn tại. Khi thêm video sau này, bổ sung dòng mới từ sample_id còn thiếu.

Các cột: sample_id, start_s, end_s, label, person_ids, status, annotator, reviewer, notes.
- start_s/end_s là giây từ đầu video RGB đã chuẩn hóa, khoảng nửa mở [start_s, end_s). Phải hữu hạn, 0 ≤ start_s < end_s ≤ duration_s trong manifest.
- person_ids là ID người do người gán nhãn quy ước trong clip, ví dụ p1;p2; có thể để trống với nhãn chỉ ở cấp clip. Track ID YOLO thuộc issue #2, cần bảng nối riêng, không tự động đồng nhất với person_ids.
- status nhận pending, uncertain, reviewed hoặc excluded. Dòng reviewed phải có annotator và reviewer; người thứ hai kiểm tra bất đồng.
- Chỉ đánh dấu reviewed sau khi xem video. Không dùng nhãn violent/fall của tên file để điền một sự kiện kéo dài toàn clip.

Ví dụ minh họa, không phải nhãn thực nghiệm:

sample_id,start_s,end_s,label,person_ids,status,annotator,reviewer,notes
example_fall,1.20,2.10,fall,p1,reviewed,A,B,chuyển tiếp ngã

## Bắt đầu và kết thúc sự kiện

Bạo lực bắt đầu ở frame đầu thấy động tác tấn công rõ ràng và kết thúc sau động tác cuối của chuỗi liên tiếp; ghi riêng khoảng yên giữa hai sự kiện. Ngã bắt đầu ở frame đầu thấy mất nâng đỡ/chuyển động rơi, kết thúc khi thân chạm mặt thấp và quá trình rơi kết thúc. Giai đoạn nằm sau ngã được ghi trong notes hoặc một thuộc tính riêng, không kéo dài sự kiện fall tùy tiện.

Đoạn normal chỉ được gán sau khi xem toàn khoảng. Đừng suy ra mọi khoảng ngoài sự kiện đều normal: có thể bị che khuất hoặc mơ hồ.

## Quy trình kiểm tra

1. Người A xem toàn video tốc độ gốc, sau đó tua chậm quanh sự kiện. Kiểm tra RGB, không gán trên ảnh depth.
2. Ghi mốc thời gian, người liên quan, che khuất, chất lượng thấp và hành động dễ nhầm.
3. Người B kiểm tra toàn bộ fall/violence và các đoạn uncertain; lấy mẫu kiểm tra normal. Với bản nghiệm thu nhỏ, nên duyệt toàn bộ clip.
4. Đối chiếu bất đồng, ghi quyết định trong notes. Tính tỷ lệ đồng thuận nhãn và chênh lệch mốc; chưa đo thì để chưa đo trong báo cáo.
5. Không lấy đoạn chưa duyệt vào đánh giá cuối. Lưu phiên bản CSV và SHA-256 của video đi kèm.

## Chống rò rỉ và mở rộng dữ liệu lớp học

Chia train/val/test theo sự kiện trước khi cắt clip hoặc trích frame/keypoint. Mọi góc quay, đoạn cắt và augmentation của cùng nguồn phải theo split của parent_video_id. Sau duyệt gần trùng, ghép group hoặc loại bản trùng rồi chạy lại split và audit.

Split hiện tại chỉ độc lập theo bản ghi. Không có metadata người tham gia đáng tin cậy nên không tuyên bố độc lập người/bối cảnh. Khi tự quay, dùng ID ẩn danh người/phiên/phòng, tách các ID có chung người hoặc phiên theo connected components; không ghi tên học sinh. Cần thêm dữ liệu lớp học được đồng thuận, hành vi bình thường khó và điều kiện che khuất. Mô phỏng an toàn; không yêu cầu va chạm hoặc ngã thật.
