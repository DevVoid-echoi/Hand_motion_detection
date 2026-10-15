"""
main.py - Entry point for the Hand Gesture Recognition project

Examples:
    python main.py                  # Open the GUI (default)
    python main.py --mode collect   # Collect data from the webcam
    python main.py --mode train     # Train the CNN
    python main.py --mode predict   # Real-time recognition demo
    python main.py --mode control   # Control the computer with gestures
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Control your computer with hand gestures (CNN + MediaPipe)")
    parser.add_argument(
        "--mode",
        choices=["gui", "collect", "train", "predict", "control"],
        default="gui",
        help="Run mode (default: gui)",
    )
    args = parser.parse_args()

    # Lazy imports so each mode only loads the libraries it needs
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
