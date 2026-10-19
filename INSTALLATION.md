# Hướng dẫn cài đặt

Tài liệu này ưu tiên Windows 10/11, PowerShell và Visual Studio Code.

## 1. Yêu cầu

- Git.
- Python 3.10 hoặc 3.11.
- Visual Studio Code.
- Dung lượng trống phù hợp với video và model.
- GPU NVIDIA là tùy chọn; hệ thống vẫn có thể chạy thử bằng CPU nhưng chậm hơn.

Kiểm tra:

```powershell
git --version
python --version
```

Nếu lệnh `python` không hoạt động trên Windows, thử:

```powershell
py --version
```

## 2. Tải project

```powershell
git clone https://github.com/USERNAME/classroom-violence-fall-detection.git
cd classroom-violence-fall-detection
code .
```

Thay `USERNAME` bằng tên tài khoản hoặc tổ chức GitHub của nhóm.

## 3. Tạo môi trường ảo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell báo lỗi script policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Khi kích hoạt thành công, đầu dòng lệnh sẽ có `(.venv)`.

Trong VS Code, chọn Python Interpreter trỏ tới:

```text
.venv\Scripts\python.exe
```

## 4. Cài thư viện

Nâng cấp pip:

```powershell
python -m pip install --upgrade pip
```

### Phương án CPU hoặc cài đặt cơ bản

```powershell
pip install -r requirements.txt
```

### Phương án GPU NVIDIA

Cấu hình PyTorch phụ thuộc hệ điều hành và nền tảng CUDA. Truy cập trang cài đặt PyTorch chính thức, chọn Windows, Pip, Python và phiên bản CUDA phù hợp, sau đó chạy đúng lệnh được trang cung cấp. Sau khi cài PyTorch, chạy:

```powershell
pip install -r requirements.txt
```

Không tự chọn lệnh CUDA khi chưa kiểm tra cấu hình máy.

## 5. Kiểm tra môi trường

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Nếu dùng NVIDIA GPU:

```powershell
nvidia-smi
```

Kiểm tra OpenCV và Ultralytics:

```powershell
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from ultralytics import YOLO; print('Ultralytics imported successfully')"
```

## 6. Kiểm tra YOLO Pose

Cài dependency bổ sung cho YOLO Pose:

```powershell
python -m pip install -r requirements-pose.txt
```

Sau khi đã có video chuẩn hóa từ bước chuẩn bị dữ liệu, chạy:

```powershell
python -m src.pose.pipeline --source data\processed\fall\<sample_id>.mp4
```

Hoặc chạy các video được liệt kê trong manifest đã tạo:

```powershell
python -m src.pose.pipeline --manifest data\metadata\local\manifest.jsonl --split test
```

Mỗi lượt chạy tạo thư mục mới dưới `outputs\pose\` với `poses.jsonl`, `annotated.mp4` và `summary.json`. Video nguồn, manifest và metadata không bị sửa. Weight mặc định `yolo11n-pose.pt` được Ultralytics tải/cached khi chạy lần đầu; dùng `--model <path>` để chỉ định weight khác. Chi tiết định dạng và giới hạn baseline: [docs/pose_workflow.md](docs/pose_workflow.md).

## 7. Chạy giao diện

Khi file `app/app.py` đã được xây dựng:

```powershell
streamlit run app/app.py
```

Dừng chương trình bằng:

```text
Ctrl + C
```

## 8. Cập nhật project

```powershell
git checkout develop
git pull origin develop
pip install -r requirements.txt
```

Nếu dependencies thay đổi, chạy lại lệnh cài requirements.

## 9. Lỗi thường gặp

### `python is not recognized`

- Cài Python.
- Tích chọn thêm Python vào PATH khi cài.
- Mở lại PowerShell hoặc VS Code.
- Thử dùng `py` thay cho `python`.

### PowerShell không kích hoạt `.venv`

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Thiết lập này chỉ áp dụng cho phiên PowerShell hiện tại.

### `ModuleNotFoundError`

Kiểm tra đã kích hoạt đúng môi trường:

```powershell
where.exe python
python -m pip list
```

Sau đó cài lại:

```powershell
python -m pip install -r requirements.txt
```

### `CUDA available: False`

- Kiểm tra `nvidia-smi`.
- Kiểm tra máy có GPU NVIDIA hay không.
- Cài đúng bản PyTorch từ trang cài đặt chính thức.
- Nếu chưa xử lý được, chạy bằng CPU để kiểm tra chức năng.

### Không đọc được video

- Kiểm tra đúng đường dẫn.
- Ưu tiên video MP4 với codec phổ biến.
- Kiểm tra trực tiếp bằng OpenCV:

```powershell
python -c "import cv2; c=cv2.VideoCapture('data/raw/sample.mp4'); print(c.isOpened())"
```

### Git đẩy nhầm video hoặc model lớn

Kiểm tra `.gitignore`, sau đó bỏ file khỏi vùng theo dõi:

```powershell
git rm -r --cached data/raw data/processed models outputs/videos
git add .gitignore
git commit -m "fix: remove generated files from git tracking"
```

## 10. Tắt môi trường

```powershell
deactivate
```
