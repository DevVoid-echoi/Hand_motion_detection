# 🖐 Hand Motion Detection

Control your computer with hand gestures through a webcam. **MediaPipe** detects 21 hand landmarks, which are redrawn as a skeleton image on a black background; a **CNN** (PyTorch) then classifies that image into one of 9 gestures and maps it to an action (play/pause, volume, mouse movement, slide navigation…).

The project also includes a from-scratch re-implementation of the CNN in **pure NumPy** (forward + backward) to illustrate how the network works internally.

## Pipeline

```
Webcam → MediaPipe HandLandmarker → Draw skeleton (black background) → Crop 64×64
       → CNN → Softmax → Gesture (+ swipe detection from wrist motion)
       → GestureController → Hotkeys / mouse / volume
```

Using skeleton images instead of raw camera frames makes the model independent of skin color, lighting and background.

## Gestures and actions

| Gesture       | Action                            |
|---------------|-----------------------------------|
| `fist`        | Play / Pause media                |
| `open_palm`   | Mute / Unmute                     |
| `thumbs_up`   | Volume up 5%                      |
| `thumbs_down` | Volume down 5%                    |
| `peace`       | Screenshot (`Win+Shift+S`)        |
| `point`       | Move cursor with index fingertip  |
| `pinch`       | Left click                        |
| `swipe_left`  | Previous slide (← key)            |
| `swipe_right` | Next slide (→ key)                |

`swipe_left` / `swipe_right` are detected from wrist movement within 0.5s, not by the CNN.

## Project structure

```
Hand_motion_detection/
├── main.py                    # Entry point (CLI)
├── requirements.txt
├── hand_gesture/
│   ├── config.py              # Paths, gesture list, hyperparameters
│   ├── landmarks.py           # MediaPipe: setup, skeleton drawing, hand cropping
│   ├── collect_data.py        # Collect images from the webcam
│   ├── dataset.py             # Dataset + augmentation + DataLoader
│   ├── model.py               # HandGestureCNN architecture
│   ├── train.py               # Training + early stopping
│   ├── predict.py             # Real-time recognition
│   ├── gesture_controller.py  # Gesture → action mapping
│   └── gui.py                 # Tkinter GUI
├── implement/
│   └── cnn_numpy_implement.py # CNN implemented in NumPy (forward + backward)
├── dataset/                   # 9 folders × 200 skeleton images (64×64)
└── models/
    ├── best_model.pth         # Trained CNN weights
    └── hand_landmarker.task   # MediaPipe Hand Landmarker model
```

## Installation

Requires Python 3.10+ and a webcam.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> GUI mode requires Tkinter. On macOS with Homebrew Python: `brew install python-tk`.

## Usage

```bash
python main.py                  # Open the GUI (default)
python main.py --mode collect   # Collect data
python main.py --mode train     # Train the model
python main.py --mode predict   # Recognition demo (no computer control)
python main.py --mode control   # Control the computer with gestures
```

The repo already ships with the dataset and `models/best_model.pth`, so you can run `predict` / `control` right away.

### 1. Data collection

| Key  | Function                     |
|------|------------------------------|
| `S`  | Toggle auto capture          |
| `C`  | Capture a single image       |
| `N`  | Move to the next gesture     |
| `Q`  | Quit                         |

Images are saved to `dataset/<gesture>/`. By default 200 images are collected per gesture; keep changing your hand angle while capturing.

### 2. Training

80/20 train/val split, Adam + `ReduceLROnPlateau`, early stopping after 7 epochs without `val_loss` improvement. The best model is saved to `models/best_model.pth`.

### 3. Control

Press `Q` in the camera window to quit. `pyautogui` has a failsafe: move the mouse into a screen corner for an emergency stop.

## CNN architecture

```
Input 3×64×64
├─ Conv3×3(32)  → BatchNorm → ReLU → MaxPool2   → 32×32×32
├─ Conv3×3(64)  → BatchNorm → ReLU → MaxPool2   → 64×16×16
├─ Conv3×3(128) → BatchNorm → ReLU → MaxPool2   → 128×8×8
└─ Flatten → FC 512 → ReLU → Dropout(0.5)
           → FC 128 → ReLU → Dropout(0.3) → FC 9
```

About 4.36 million parameters. All hyperparameters live in [hand_gesture/config.py](hand_gesture/config.py).

## NumPy CNN

[implement/cnn_numpy_implement.py](implement/cnn_numpy_implement.py) re-implements the exact architecture above in NumPy:

- `forward_with_cache`: forward pass for a `(N, C, H, W)` batch, with BatchNorm (train/eval, running stats) and inverted dropout
- `backward`: backpropagation through the whole network, returning gradients with the same keys as `params`

Verified results: the forward pass matches `best_model.pth` (error ~1e-14), and gradients match PyTorch autograd and finite differences.

## OS notes

- **Windows:** fully supported. Volume is controlled directly via `pycaw`.
- **macOS / Linux:** volume uses media keys via `pyautogui`. The `Win+Shift+S` screenshot shortcut only exists on Windows. On macOS, grant Camera and Accessibility permissions to your Terminal / IDE.
