# Báo cáo thử nghiệm kỹ thuật Pose Extraction và MLP/LSTM

## 1. Mục tiêu

Thử nghiệm này nhằm xác minh các thành phần kỹ thuật chính của dự án có thể hoạt động và kết nối với nhau:

1. Đọc danh sách video từ manifest.
2. Trích xuất tư thế người bằng YOLO11n-Pose.
3. Theo dõi người được phát hiện bằng ByteTrack.
4. Chuyển dữ liệu pose thành chuỗi đặc trưng có độ dài cố định.
5. Nạp dữ liệu vào các mô hình MLP và LSTM.
6. Kiểm tra khả năng thực thi của hai mô hình.
7. Kiểm tra quá trình huấn luyện thử nghiệm và lưu checkpoint.
8. Kiểm tra các chức năng đánh giá mô hình và phân tích lỗi.

**Phạm vi:** Chỉ sử dụng 10 video mẫu cho thử nghiệm Pose và Dataset Loader. Các kiểm thử huấn luyện, Evaluation và Error Analysis sử dụng dữ liệu tổng hợp hoặc mô hình chưa huấn luyện. Đây là thử nghiệm tích hợp kỹ thuật, không phải đánh giá hiệu suất của mô hình trên toàn bộ bộ dữ liệu.

## 2. Môi trường thực hiện

- Hệ điều hành: Windows
- Terminal: PowerShell
- Python: 3.11.1
- GPU: NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM
- PyTorch: 2.15.0+cu132
- CUDA khả dụng: True
- Phiên bản CUDA: 13.2
- Ultralytics: 8.4.198
- OpenCV: 5.0.0
- NumPy: 2.4.6
- scikit-learn: 1.9.1
- Matplotlib: 3.11.2
- pandas: 3.1.0

Dự án sử dụng môi trường ảo `.venv`.

Kích hoạt môi trường ảo tại thư mục gốc dự án:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Manifest thử nghiệm

Đường dẫn:

`data/metadata/local/manifest_pose_pilot10.jsonl`

Manifest chứa 10 video thuộc tập train.

| Nhãn | Số lượng |
|---|---:|
| Normal | 4 |
| Violence | 4 |
| Fall | 2 |
| **Tổng cộng** | **10** |

Tất cả bản ghi đều có trạng thái:

`source_label_pending_review`

Điều này có nghĩa nhãn hiện tại được kế thừa từ nguồn dữ liệu và chưa được xác minh thủ công.

Các video AIRTLab có hai góc camera cho mỗi sự kiện được chọn. Các video UR Fall sử dụng góc camera `cam0`.

### Kết quả kiểm tra manifest

- Có 10 bản ghi JSONL hợp lệ.
- Có 10 sample ID khác nhau.
- Tất cả bản ghi thuộc tập train.
- Phân bố nhãn đúng với dự kiến.
- Cả 10 nhãn đều đang chờ xem xét.

## 4. Trích xuất Pose

File thực thi:

`src/pose/pipeline.py`

Cấu hình:

- Mô hình: YOLO11n-Pose
- Tracker: ByteTrack
- Keypoints: 17 điểm theo chuẩn COCO

### Lệnh thực thi

Chạy tại thư mục gốc dự án:

```powershell
python -m src.pose.pipeline `
    --manifest data/metadata/local/manifest_pose_pilot10.jsonl `
    --split train `
    --output-dir outputs/pose/dataset_pilot_10 `
    --run-id pilot-10-v1 `
    --device 0
```

Thư mục kết quả:

`outputs/pose/dataset_pilot_10/`

### Kết quả

| Chỉ số | Kết quả |
|---|---:|
| Video xử lý thành công | 10 |
| Tổng số frame | 871 |
| Số lượt phát hiện người | 2.770 |
| Frame được heuristic đánh dấu là ứng viên ngã | 13 |

### Kết quả theo từng nhãn

| Nhãn | Video | Frame | Detection | Ứng viên ngã |
|---|---:|---:|---:|---:|
| Normal | 4 | 422 | 1.582 | 0 |
| Violence | 4 | 286 | 1.026 | 0 |
| Fall | 2 | 163 | 162 | 13 |
| **Tổng** | **10** | **871** | **2.770** | **13** |

Mỗi video được xử lý tạo ra:

- `summary.json`
- `poses.jsonl`
- `annotated.mp4`

**Lưu ý:** 13 frame ứng viên ngã được xác định bằng heuristic. Đây không phải kết quả dự đoán từ một bộ phân loại Fall đã được huấn luyện.

## 5. Kiểm tra Dataset Loader

File thực thi:

`src/models/dataset.py`

Kích thước đặc trưng:

- Person features: 61
- Interaction features: 9
- Frame features: 131

Độ dài chuỗi: 16 frame.

### Lệnh kiểm tra

```powershell
python -c "from collections import Counter; import torch; from src.models.dataset import load_pose_dataset; d=load_pose_dataset('outputs/pose/dataset_pilot_10', manifest_path='data/metadata/local/manifest_pose_pilot10.jsonl', split='train', sequence_length=16); xs=[d[i][0] for i in range(len(d))]; ys=[int(d[i][1]) for i in range(len(d))]; print('Total samples:',len(d)); print('Feature shapes:',Counter(tuple(x.shape) for x in xs)); print('Label indices:',Counter(ys)); print('All finite:',all(torch.isfinite(x).all().item() for x in xs)); print('Feature min:',min(x.min().item() for x in xs)); print('Feature max:',max(x.max().item() for x in xs)); print('Feature dimensions:',xs[0].shape[-1] if xs else None)"
```

### Kết quả

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

Các giá trị đặc trưng không bắt buộc nằm trong khoảng `[0, 1]`, vì một số đặc trưng sử dụng tọa độ tương đối và tỷ lệ.

## 6. Kiểm tra mô hình MLP và LSTM

Các file triển khai:

- `src/models/mlp.py`
- `src/models/lstm.py`

Cả hai mô hình đều vượt qua kiểm tra forward pass với tensor đầu vào có kích thước:

`[4, 16, 131]`

Kích thước đầu ra mong đợi:

`[4, 3]`

### MLP

MLP sử dụng phép lấy trung bình theo chiều thời gian của chuỗi đầu vào. Vì vậy, mô hình không trực tiếp học thứ tự diễn biến giữa các frame.

### LSTM

LSTM xử lý dữ liệu theo chuỗi thời gian, phù hợp hơn khi cần khai thác sự thay đổi tư thế giữa các frame.

Các kiểm tra này xác nhận mô hình có thể thực thi và trả về đúng kích thước đầu ra. Chúng không chứng minh độ chính xác phân loại.

## 7. Kiểm tra quá trình huấn luyện

File triển khai:

`src/models/train.py`

Một bộ dữ liệu tổng hợp được sử dụng để kiểm tra chức năng huấn luyện:

- 30 sample.
- 16 time step mỗi sample.
- 131 đặc trưng mỗi time step.
- 3 lớp được gán nhãn ngẫu nhiên.
- 3 epoch.
- Chạy trên CPU.

Cùng một bộ dữ liệu tổng hợp được sử dụng cho cả train và validation. Đây chỉ là kiểm tra chức năng phần mềm, không phải quy trình đánh giá mô hình hợp lệ.

### Kết quả

- Quá trình huấn luyện hoàn thành 3 epoch.
- Tạo thành công `best_model.pt`.
- Tạo thành công `last_model.pt`.
- Cả hai checkpoint đều chứa model weights và optimizer state.
- Cả hai checkpoint được nạp lại thành công.

Đã chỉnh sửa cơ chế lưu checkpoint để:

- `best_model.pt` lưu trạng thái optimizer tương ứng với mô hình tốt nhất.
- `last_model.pt` lưu trạng thái optimizer tại epoch cuối cùng.

## 8. Kiểm tra Evaluation và Error Analysis

### 8.1. Evaluation

Các file triển khai:

- `src/evaluation/evaluate.py`
- `src/evaluation/plots.py`

Các chức năng được kiểm tra:

- Tính accuracy, precision, recall và F1-score.
- Tính các metric theo từng lớp.
- Tạo confusion matrix.
- Xuất kết quả đánh giá thành JSON và CSV.
- Tạo confusion matrix dạng ảnh PNG.
- Ghi nhận sample ID thật hoặc tạo ID dự phòng khi Dataset không cung cấp ID.

### 8.2. Kiểm tra compute_metrics bằng dữ liệu tổng hợp

Sử dụng 10 nhãn thực tế giả lập và 10 nhãn dự đoán:

```python
y_true = [0,0,0,0,1,1,1,1,2,2]
y_pred = [0,0,1,0,1,1,0,1,2,1]
```

Kết quả:

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.7000 |
| Macro Precision | 0.7833 |
| Macro Recall | 0.6667 |
| Macro F1 | 0.6944 |
| Weighted F1 | 0.7000 |

Kết quả theo từng lớp:

| Lớp | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Normal | 0.75 | 0.75 | 0.75 | 4 |
| Violence | 0.60 | 0.75 | 0.6667 | 4 |
| Fall | 1.00 | 0.50 | 0.6667 | 2 |

Confusion matrix:

| Ground truth / Prediction | Normal | Violence | Fall |
|---|---:|---:|---:|
| Normal | 3 | 1 | 0 |
| Violence | 1 | 3 | 0 |
| Fall | 0 | 1 | 1 |

Đây là dữ liệu tổng hợp dùng để kiểm tra phép tính metric, không phải kết quả của mô hình được huấn luyện trên dữ liệu dự án.

### 8.3. Kiểm tra evaluate_model

Thử nghiệm sử dụng:

- TensorDataset gồm 10 sample tổng hợp.
- MLP chưa huấn luyện.
- Input shape `[10, 16, 131]`.
- 3 lớp đầu ra.
- Thiết bị CPU.

Kết quả:

- Evaluator xử lý thành công 10 sample.
- Tạo confusion matrix kích thước `3 × 3`.
- Xuất kết quả đánh giá thành công.
- Không xảy ra lỗi khi Dataset không có sample ID.

Trong lần kiểm thử này, cả 10 mẫu đều được MLP dự đoán thành lớp Fall. Đây là kết quả của một mô hình chưa huấn luyện với dữ liệu ngẫu nhiên, không có ý nghĩa đánh giá hiệu suất phân loại.

### 8.4. Kiểm tra sample ID

Đã kiểm tra hai trường hợp:

**Trường hợp 1: Dataset không có sample ID**

Sử dụng TensorDataset.

Evaluator tạo ID dự phòng:

- `sample_0`
- `sample_1`
- `sample_2`
- ...

**Trường hợp 2: Dataset có sample ID**

Sử dụng `PoseSequenceDataset` với pilot manifest 10 mẫu.

Evaluator lấy đúng ID từ `dataset.samples`.

Các ID được ghi vào `predictions.csv`:

- `airltab_normal_10_cam1`
- `airltab_normal_10_cam2`
- `airltab_normal_13_cam1`
- `airltab_normal_13_cam2`
- `airltab_violence_100_cam1`
- `airltab_violence_100_cam2`
- `airltab_violence_101_cam1`
- `airltab_violence_101_cam2`
- `urfall_fall-02_cam0`
- `urfall_fall-03_cam0`

Cả 10 ID đều khớp với pilot manifest.

### 8.5. Các file đầu ra Evaluation

Trong các smoke test, evaluator tạo thành công:

- `classification_report.json`
- `metrics.json`
- `predictions.csv`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`

Các thư mục kiểm thử:

- `outputs/evaluation/evaluator_smoke_test/`
- `outputs/evaluation/pose_dataset_smoke_test/`

Các file đầu ra được lưu cục bộ và không đưa vào Git.

### 8.6. Error Analysis

File triển khai:

`src/evaluation/error_analysis.py`

Chức năng được kiểm tra:

- Xác định các dự đoán sai.
- Thống kê số lỗi theo lớp.
- Thống kê các cặp nhầm lẫn giữa các lớp.
- Xuất kết quả thành CSV và Markdown.
- Kiểm tra dữ liệu đầu vào không hợp lệ.

Sử dụng 5 bản ghi dự đoán tổng hợp.

Kết quả:

| Chỉ số | Kết quả |
|---|---:|
| Tổng số lỗi | 3 |
| Lỗi lớp Violence | 1 |
| Lỗi lớp Fall | 2 |

Các cặp nhầm lẫn:

| Nhầm lẫn | Số lượng |
|---|---:|
| Violence → Normal | 1 |
| Fall → Violence | 1 |
| Fall → Normal | 1 |

Các kiểm tra bổ sung đều hoạt động đúng:

- Tuple không đủ 5 giá trị: phát hiện và báo lỗi.
- Confidence vượt khoảng `[0, 1]`: phát hiện và báo lỗi.
- Nhãn không thuộc danh sách cho phép: phát hiện và báo lỗi.

Các file được tạo:

- `error_analysis.csv`
- `error_summary.md`

Đây là kiểm thử chức năng với dữ liệu tổng hợp, chưa phải phân tích lỗi của mô hình thực tế.

## 9. Những hạn chế hiện tại

1. Thử nghiệm Pose và Dataset Loader chỉ sử dụng 10 video, chưa đại diện cho toàn bộ bộ dữ liệu.
2. Nhãn nguồn chưa được xác minh thủ công.
3. Các frame ứng viên ngã được tạo bằng heuristic, không phải dự đoán từ mô hình đã huấn luyện.
4. Chưa thực hiện huấn luyện hoặc đánh giá hiệu suất mô hình trên dữ liệu thực trong phạm vi thử nghiệm này.
5. MLP không trực tiếp mô hình hóa thứ tự thời gian.
6. Cần xem xét lại cách LSTM xử lý các chuỗi được padding trước khi huấn luyện thực tế, bao gồm việc truyền độ dài chuỗi khi phù hợp.
7. Kiểm tra huấn luyện sử dụng dữ liệu tổng hợp và cùng dữ liệu cho train và validation.
8. Các kiểm thử Evaluation và Error Analysis cũng sử dụng dữ liệu tổng hợp hoặc mô hình chưa huấn luyện.
9. Các kết quả đầu ra của pilot được lưu cục bộ và không đưa vào Git.

## 10. Quy trình tái thực hiện

Từ thư mục gốc dự án:

### 10.1. Chuẩn bị

1. Kích hoạt môi trường ảo `.venv`.
2. Đảm bảo 10 video nguồn tồn tại đúng đường dẫn trong manifest.
3. Đảm bảo các thư viện trong `requirements.txt` đã được cài đặt.

### 10.2. Chạy Pose Pipeline

```powershell
python -m src.pose.pipeline `
    --manifest data/metadata/local/manifest_pose_pilot10.jsonl `
    --split train `
    --output-dir outputs/pose/dataset_pilot_10 `
    --run-id pilot-10-v1 `
    --device 0
```

Lưu ý: Pipeline tạo thư mục run mới và không tự động ghi đè run đã tồn tại. Cần sử dụng run ID và thư mục đầu ra phù hợp nếu chạy lại.

### 10.3. Kiểm tra Dataset Loader

Chạy lệnh tại Mục 5.

### 10.4. Kiểm tra MLP và LSTM

Thực hiện forward pass với input `[4, 16, 131]` và xác nhận output `[4, 3]`.

### 10.5. Kiểm tra Training

Sử dụng bộ dữ liệu tổng hợp để kiểm tra quá trình huấn luyện và lưu checkpoint theo nội dung Mục 7.

### 10.6. Kiểm tra Evaluation

Có thể kiểm tra riêng hàm tính metric bằng lệnh:

```powershell
python -c "from src.evaluation.evaluate import compute_metrics; y_true=[0,0,0,0,1,1,1,1,2,2]; y_pred=[0,0,1,0,1,1,0,1,2,1]; print(compute_metrics(y_true,y_pred))"
```

Để kiểm tra evaluator với TensorDataset, MLP và dữ liệu tổng hợp, sử dụng quy trình smoke test tại Mục 8.

Để kiểm tra ID thật, cần có sẵn Pose output của pilot và manifest tương ứng.

### 10.7. Kiểm tra Error Analysis

Sử dụng dữ liệu dự đoán tổng hợp theo Mục 8.6 để xác minh thống kê lỗi và khả năng xuất báo cáo.

Các video nguồn, manifest pilot và kết quả pose cần có sẵn trên máy để tái thực hiện đầy đủ các bước tương ứng.

Các kết quả đầu ra thử nghiệm không được đưa vào Git commit.

## 11. Kết luận

Thử nghiệm kỹ thuật đã xác minh thành công các thành phần trong phạm vi kiểm thử:

- Trích xuất pose.
- Theo dõi người.
- Chuyển dữ liệu pose thành chuỗi đặc trưng.
- Nạp dữ liệu bằng Dataset Loader.
- Thực thi forward pass của MLP và LSTM.
- Kiểm tra quá trình huấn luyện tổng hợp và lưu checkpoint.
- Tính toán các metric đánh giá.
- Tạo và xuất confusion matrix.
- Hỗ trợ sample ID thật và ID dự phòng.
- Phân tích lỗi và xuất báo cáo.

Các thành phần trên đã có thể hoạt động trong phạm vi thử nghiệm nhỏ.

**Tuy nhiên, kết quả này chưa chứng minh hệ thống có khả năng phát hiện chính xác hành vi bạo lực hoặc té ngã trong lớp học.** Việc xác minh nhãn, huấn luyện trên dữ liệu thực và đánh giá mô hình vẫn là các công việc tiếp theo.
