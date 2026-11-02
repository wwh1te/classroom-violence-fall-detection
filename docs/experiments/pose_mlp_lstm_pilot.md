\# Báo cáo thử nghiệm kỹ thuật Pose Extraction và MLP/LSTM



\## 1. Mục tiêu



Thử nghiệm này nhằm xác minh các thành phần kỹ thuật chính của dự án có thể hoạt động và kết nối với nhau:



1\. Đọc danh sách video từ manifest.

2\. Trích xuất tư thế người bằng YOLO11n-Pose.

3\. Theo dõi người được phát hiện bằng ByteTrack.

4\. Chuyển dữ liệu pose thành chuỗi đặc trưng có độ dài cố định.

5\. Nạp dữ liệu vào các mô hình MLP và LSTM.

6\. Kiểm tra khả năng thực thi của hai mô hình.

7\. Kiểm tra quá trình huấn luyện thử nghiệm và lưu checkpoint.



\*\*Phạm vi:\*\* Chỉ sử dụng 10 video mẫu. Đây là thử nghiệm tích hợp kỹ thuật, không phải đánh giá hiệu suất của mô hình trên toàn bộ bộ dữ liệu.



\## 2. Môi trường thực hiện



\- Hệ điều hành: Windows

\- Terminal: PowerShell

\- Python: 3.11.1

\- GPU: NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM

\- PyTorch: 2.15.0+cu132

\- CUDA khả dụng: True

\- Phiên bản CUDA: 13.2

\- Ultralytics: 8.4.198

\- OpenCV: 5.0.0

\- NumPy: 2.4.6



Dự án sử dụng môi trường ảo `.venv`.



Kích hoạt môi trường ảo tại thư mục gốc dự án:



```powershell

.venv\\Scripts\\Activate.ps1

```



\## 3. Manifest thử nghiệm



Đường dẫn:



`data/metadata/local/manifest\_pose\_pilot10.jsonl`



Manifest chứa 10 video thuộc tập train.



| Nhãn | Số lượng |

|---|---:|

| Normal | 4 |

| Violence | 4 |

| Fall | 2 |

| \*\*Tổng cộng\*\* | \*\*10\*\* |



Tất cả bản ghi đều có trạng thái:



`source\_label\_pending\_review`



Điều này có nghĩa nhãn hiện tại được kế thừa từ nguồn dữ liệu và chưa được xác minh thủ công.



Các video AIRTLab có hai góc camera cho mỗi sự kiện được chọn. Các video UR Fall sử dụng góc camera `cam0`.



\### Kết quả kiểm tra manifest



\- Có 10 bản ghi JSONL hợp lệ.

\- Có 10 sample ID khác nhau.

\- Tất cả bản ghi thuộc tập train.

\- Phân bố nhãn đúng với dự kiến.

\- Cả 10 nhãn đều đang chờ xem xét.



\## 4. Trích xuất Pose



File thực thi:



`src/pose/pipeline.py`



Cấu hình:

\- Mô hình: YOLO11n-Pose

\- Tracker: ByteTrack

\- Keypoints: 17 điểm theo chuẩn COCO



\### Lệnh thực thi



Chạy tại thư mục gốc dự án:



```powershell

python -m src.pose.pipeline `

&#x20;   --manifest data/metadata/local/manifest\_pose\_pilot10.jsonl `

&#x20;   --split train `

&#x20;   --output-dir outputs/pose/dataset\_pilot\_10 `

&#x20;   --run-id pilot-10-v1 `

&#x20;   --device 0

```



Thư mục kết quả:



`outputs/pose/dataset\_pilot\_10/`



\### Kết quả



| Chỉ số | Kết quả |

|---|---:|

| Video xử lý thành công | 10 |

| Tổng số frame | 871 |

| Số lượt phát hiện người | 2.770 |

| Frame được heuristic đánh dấu là ứng viên ngã | 13 |



\### Kết quả theo từng nhãn



| Nhãn | Video | Frame | Detection | Ứng viên ngã |

|---|---:|---:|---:|---:|

| Normal | 4 | 422 | 1.582 | 0 |

| Violence | 4 | 286 | 1.026 | 0 |

| Fall | 2 | 163 | 162 | 13 |

| \*\*Tổng\*\* | \*\*10\*\* | \*\*871\*\* | \*\*2.770\*\* | \*\*13\*\* |



Mỗi video được xử lý tạo ra:

\- `summary.json`

\- `poses.jsonl`

\- `annotated.mp4`



\*\*Lưu ý:\*\* 13 frame ứng viên ngã được xác định bằng heuristic. Đây không phải kết quả dự đoán từ một bộ phân loại Fall đã được huấn luyện.



\## 5. Kiểm tra Dataset Loader



File thực thi:



`src/models/dataset.py`



Kích thước đặc trưng:

\- Person features: 61

\- Interaction features: 9

\- Frame features: 131



Độ dài chuỗi: 16 frame.



\### Lệnh kiểm tra



```powershell

python -c "from collections import Counter; import torch; from src.models.dataset import load\_pose\_dataset; d=load\_pose\_dataset('outputs/pose/dataset\_pilot\_10', manifest\_path='data/metadata/local/manifest\_pose\_pilot10.jsonl', split='train', sequence\_length=16); xs=\[d\[i]\[0] for i in range(len(d))]; ys=\[int(d\[i]\[1]) for i in range(len(d))]; print('Total samples:',len(d)); print('Feature shapes:',Counter(tuple(x.shape) for x in xs)); print('Label indices:',Counter(ys)); print('All finite:',all(torch.isfinite(x).all().item() for x in xs)); print('Feature min:',min(x.min().item() for x in xs)); print('Feature max:',max(x.max().item() for x in xs)); print('Feature dimensions:',xs\[0].shape\[-1] if xs else None)"

```



\### Kết quả



| Kiểm tra | Kết quả |

|---|---|

| Số sample nạp được | 10 |

| Kích thước mỗi sample | `(16, 131)` |

| Normal | Nhãn 0, có 4 sample |

| Violence | Nhãn 1, có 4 sample |

| Fall | Nhãn 2, có 2 sample |

| Tất cả giá trị đặc trưng hữu hạn | True |

| Giá trị nhỏ nhất | -0.349588 |

| Giá trị lớn nhất | 2.233606 |



Các giá trị đặc trưng không bắt buộc nằm trong khoảng `\[0, 1]`, vì một số đặc trưng sử dụng tọa độ tương đối và tỷ lệ.



\## 6. Kiểm tra mô hình MLP và LSTM



Các file triển khai:



\- `src/models/mlp.py`

\- `src/models/lstm.py`



Cả hai mô hình đều vượt qua kiểm tra forward pass với tensor đầu vào có kích thước:



`\[4, 16, 131]`



Kích thước đầu ra mong đợi:



`\[4, 3]`



\### MLP



MLP sử dụng phép lấy trung bình theo chiều thời gian của chuỗi đầu vào. Vì vậy, mô hình không trực tiếp học thứ tự diễn biến giữa các frame.



\### LSTM



LSTM xử lý dữ liệu theo chuỗi thời gian, phù hợp hơn khi cần khai thác sự thay đổi tư thế giữa các frame.



Các kiểm tra này xác nhận mô hình có thể thực thi và trả về đúng kích thước đầu ra. Chúng không chứng minh độ chính xác phân loại.



\## 7. Kiểm tra quá trình huấn luyện



File triển khai:



`src/models/train.py`



Một bộ dữ liệu tổng hợp được sử dụng để kiểm tra chức năng huấn luyện:



\- 30 sample.

\- 16 time step mỗi sample.

\- 131 đặc trưng mỗi time step.

\- 3 lớp được gán nhãn ngẫu nhiên.

\- 3 epoch.

\- Chạy trên CPU.



Cùng một bộ dữ liệu tổng hợp được sử dụng cho cả train và validation. Đây chỉ là kiểm tra chức năng phần mềm, không phải quy trình đánh giá mô hình hợp lệ.



\### Kết quả



\- Quá trình huấn luyện hoàn thành 3 epoch.

\- Tạo thành công `best\_model.pt`.

\- Tạo thành công `last\_model.pt`.

\- Cả hai checkpoint đều chứa model weights và optimizer state.

\- Cả hai checkpoint được nạp lại thành công.



Đã chỉnh sửa cơ chế lưu checkpoint để:

\- `best\_model.pt` lưu trạng thái optimizer tương ứng với mô hình tốt nhất.

\- `last\_model.pt` lưu trạng thái optimizer tại epoch cuối cùng.



\## 8. Những hạn chế hiện tại



1\. Thử nghiệm chỉ sử dụng 10 video, chưa đại diện cho toàn bộ bộ dữ liệu.

2\. Nhãn nguồn chưa được xác minh thủ công.

3\. Các frame ứng viên ngã được tạo bằng heuristic, không phải dự đoán từ mô hình đã huấn luyện.

4\. Chưa thực hiện huấn luyện hoặc đánh giá hiệu suất mô hình trên dữ liệu thực trong phạm vi thử nghiệm này.

5\. MLP không trực tiếp mô hình hóa thứ tự thời gian.

6\. Cần xem xét lại cách LSTM xử lý các chuỗi được padding trước khi huấn luyện thực tế, bao gồm việc truyền độ dài chuỗi khi phù hợp.

7\. Kiểm tra huấn luyện sử dụng dữ liệu tổng hợp và cùng dữ liệu cho train và validation.

8\. Các kết quả đầu ra của pilot được lưu cục bộ và không đưa vào Git.



\## 9. Quy trình tái thực hiện



Từ thư mục gốc dự án:



1\. Kích hoạt môi trường ảo `.venv`.

2\. Đảm bảo 10 video nguồn tồn tại đúng đường dẫn trong manifest.

3\. Chạy lệnh trích xuất Pose tại Mục 4.

4\. Chạy lệnh kiểm tra Dataset Loader tại Mục 5.

5\. Chạy kiểm tra forward pass của MLP và LSTM.

6\. Chạy kiểm tra huấn luyện dữ liệu tổng hợp nếu cần xác minh chức năng lưu checkpoint.



Các video nguồn và kết quả pose không được đưa vào Git commit của pilot. Cần có sẵn các tệp này trên máy để tái thực hiện bước trích xuất và nạp dữ liệu.



\## 10. Kết luận



Thử nghiệm kỹ thuật trên 10 video đã xác minh thành công các thành phần:



\- Trích xuất pose.

\- Theo dõi người.

\- Chuyển dữ liệu pose thành chuỗi đặc trưng.

\- Nạp dữ liệu bằng Dataset Loader.

\- Thực thi forward pass của MLP và LSTM.

\- Kiểm tra quá trình huấn luyện tổng hợp và lưu checkpoint.



Các thành phần trên đã có thể kết nối và hoạt động trong phạm vi thử nghiệm nhỏ.



\*\*Tuy nhiên, kết quả này chưa chứng minh hệ thống có khả năng phát hiện chính xác hành vi bạo lực hoặc té ngã trong lớp học.\*\* Việc xác minh nhãn, huấn luyện trên dữ liệu thực và đánh giá mô hình vẫn là các công việc tiếp theo.
