# 🖐 Hand Motion Detection

Điều khiển máy tính bằng cử chỉ tay qua webcam. **MediaPipe** phát hiện 21 điểm khớp bàn tay, vẽ lại thành ảnh khung xương trên nền đen, rồi một mạng **CNN** (PyTorch) phân loại ảnh đó thành 1 trong 9 cử chỉ và ánh xạ sang hành động (play/pause, âm lượng, di chuột, chuyển slide…).

Project còn kèm một bản cài đặt lại CNN bằng **NumPy thuần** (forward + backward) để minh hoạ cách mạng hoạt động bên trong.

## Pipeline

```
Webcam → MediaPipe HandLandmarker → Vẽ khung xương (nền đen) → Crop 64×64
       → CNN → Softmax → Cử chỉ (+ phát hiện swipe theo chuyển động cổ tay)
       → GestureController → Phím tắt / chuột / âm lượng
```

Dùng ảnh khung xương thay vì ảnh thật giúp model không phụ thuộc vào màu da, ánh sáng hay phông nền.

## Cử chỉ và hành động

| Cử chỉ        | Hành động                         |
|---------------|-----------------------------------|
| `fist`        | Play / Pause media                |
| `open_palm`   | Mute / Unmute                     |
| `thumbs_up`   | Tăng âm lượng 5%                  |
| `thumbs_down` | Giảm âm lượng 5%                  |
| `peace`       | Chụp màn hình (`Win+Shift+S`)     |
| `point`       | Di chuột theo đầu ngón trỏ        |
| `pinch`       | Click chuột trái                  |
| `swipe_left`  | Slide trước (phím ←)              |
| `swipe_right` | Slide tiếp (phím →)               |

`swipe_left` / `swipe_right` được nhận diện theo chuyển động của cổ tay trong 0.5s, không qua CNN.

## Cấu trúc thư mục

```
Hand_motion_detection/
├── main.py                    # Entry point (CLI)
├── requirements.txt
├── hand_gesture/
│   ├── config.py              # Đường dẫn, danh sách cử chỉ, hyperparameters
│   ├── landmarks.py           # MediaPipe: khởi tạo, vẽ khung xương, crop tay
│   ├── collect_data.py        # Thu thập ảnh từ webcam
│   ├── dataset.py             # Dataset + augmentation + DataLoader
│   ├── model.py               # Kiến trúc HandGestureCNN
│   ├── train.py               # Huấn luyện + early stopping
│   ├── predict.py             # Nhận diện real-time
│   ├── gesture_controller.py  # Ánh xạ cử chỉ → hành động
│   ├── gui.py                 # Giao diện Tkinter
│   └── numpy_cnn.py           # CNN cài đặt bằng NumPy (forward + backward)
├── dataset/                   # 9 thư mục × 200 ảnh khung xương 64×64
└── models/
    ├── best_model.pth         # Trọng số CNN đã train
    └── hand_landmarker.task   # Model MediaPipe Hand Landmarker
```

## Cài đặt

Yêu cầu Python 3.10+ và webcam.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Chế độ GUI cần Tkinter. Trên macOS + Homebrew Python: `brew install python-tk`.

## Sử dụng

```bash
python main.py                  # Mở GUI (mặc định)
python main.py --mode collect   # Thu thập dữ liệu
python main.py --mode train     # Huấn luyện model
python main.py --mode predict   # Demo nhận diện (không điều khiển máy)
python main.py --mode control   # Điều khiển máy tính bằng cử chỉ
```

Repo đã có sẵn dataset và `models/best_model.pth`, nên có thể chạy `predict` / `control` ngay.

### 1. Thu thập dữ liệu

| Phím | Chức năng                    |
|------|------------------------------|
| `S`  | Bật / tắt chụp tự động       |
| `C`  | Chụp 1 ảnh                   |
| `N`  | Chuyển sang cử chỉ tiếp theo |
| `Q`  | Thoát                        |

Ảnh được lưu vào `dataset/<gesture>/`. Mặc định thu 200 ảnh mỗi cử chỉ; nên thay đổi góc tay liên tục khi chụp.

### 2. Huấn luyện

Chia train/val 80/20, dùng Adam + `ReduceLROnPlateau`, dừng sớm sau 7 epoch nếu `val_loss` không giảm. Model tốt nhất được lưu vào `models/best_model.pth`.

### 3. Điều khiển

Nhấn `Q` trong cửa sổ camera để thoát. `pyautogui` có failsafe: đưa chuột vào góc màn hình để dừng khẩn cấp.

## Kiến trúc CNN

```
Input 3×64×64
├─ Conv3×3(32)  → BatchNorm → ReLU → MaxPool2   → 32×32×32
├─ Conv3×3(64)  → BatchNorm → ReLU → MaxPool2   → 64×16×16
├─ Conv3×3(128) → BatchNorm → ReLU → MaxPool2   → 128×8×8
└─ Flatten → FC 512 → ReLU → Dropout(0.5)
           → FC 128 → ReLU → Dropout(0.3) → FC 9
```

Khoảng 4.36 triệu tham số. Các hyperparameter đều nằm trong [hand_gesture/config.py](hand_gesture/config.py).

## CNN bằng NumPy

[hand_gesture/numpy_cnn.py](hand_gesture/numpy_cnn.py) cài đặt lại đúng kiến trúc trên bằng NumPy:

- `forward_with_cache`: forward cho batch `(N, C, H, W)`, có BatchNorm (train/eval, running stats) và inverted dropout
- `backward`: lan truyền ngược qua toàn mạng, trả về gradient cùng key với `params`

Kết quả đã được kiểm chứng: forward khớp `best_model.pth` (sai số ~1e-14), gradient khớp PyTorch autograd và finite-difference.

## Lưu ý theo hệ điều hành

- **Windows:** hỗ trợ đầy đủ. Âm lượng điều khiển trực tiếp qua `pycaw`.
- **macOS / Linux:** âm lượng dùng phím media qua `pyautogui`. Phím tắt chụp màn hình `Win+Shift+S` chỉ có trên Windows. Trên macOS cần cấp quyền Camera và Accessibility cho Terminal / IDE.
