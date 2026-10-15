"""
main.py - Entry point cho project Hand Gesture Recognition

Ví dụ:
    python main.py                  # Mở giao diện GUI (mặc định)
    python main.py --mode collect   # Thu thập dữ liệu từ webcam
    python main.py --mode train     # Huấn luyện CNN
    python main.py --mode predict   # Demo nhận diện real-time
    python main.py --mode control   # Điều khiển máy tính bằng cử chỉ
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Điều khiển máy tính bằng cử chỉ tay (CNN + MediaPipe)")
    parser.add_argument(
        "--mode",
        choices=["gui", "collect", "train", "predict", "control"],
        default="gui",
        help="Chế độ chạy (mặc định: gui)",
    )
    args = parser.parse_args()

    # Import trễ để chỉ load thư viện cần cho từng chế độ
    if args.mode == "gui":
        from hand_gesture.gui import main as run
    elif args.mode == "collect":
        from hand_gesture.collect_data import collect_gesture_data as run
    elif args.mode == "train":
        from hand_gesture.train import train_model as run
    elif args.mode == "predict":
        from hand_gesture.predict import run_prediction_demo as run
    else:
        from hand_gesture.predict import run_realtime_controller as run

    run()


if __name__ == "__main__":
    main()
