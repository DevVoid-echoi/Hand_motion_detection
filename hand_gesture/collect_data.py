"""
collect_data.py - Collect hand gesture data from the webcam
Uses the MediaPipe Tasks API (v1.0+) to detect the hand, crop the hand region, and save images into per-label folders.
"""
import os
import cv2
import time
import numpy as np
import mediapipe as mp
from .config import DATASET_DIR, GESTURES, CAMERA_INDEX, COLLECT_NUM_SAMPLES, IMG_SIZE
from .landmarks import create_hand_landmarker, draw_landmarks, crop_hand


def safe_imwrite(filepath, img):
    """
    Write an image to file, supporting Unicode paths on Windows.
    cv2.imwrite() fails silently on paths containing non-ASCII characters,
    so use cv2.imencode() + write the bytes with Python instead.
    """
    success, encoded = cv2.imencode('.jpg', img)
    if success:
        with open(filepath, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


def create_dataset_dirs():
    """Create a directory for each gesture."""
    for gesture in GESTURES:
        gesture_dir = os.path.join(DATASET_DIR, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
    print(f"✓ Created {len(GESTURES)} directories in {DATASET_DIR}")


def collect_gesture_data():
    """
    Run the data collection loop.

    Controls:
        - Press 's' to start/pause auto capture
        - Press 'c' to capture a single image
        - Press 'n' to move to the next gesture
        - Press 'q' to quit
    """
    create_dataset_dirs()

    hand_landmarker = create_hand_landmarker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("✗ Could not open webcam!")
        return

    current_gesture_idx = 0
    count = 0
    auto_capture = False
    last_capture_time = 0
    capture_interval = 0.15  # Auto capture every 0.15s

    print("\n" + "=" * 60)
    print("  HAND GESTURE DATA COLLECTION")
    print("=" * 60)
    print("  [S] Toggle auto capture")
    print("  [C] Capture one image")
    print("  [N] Next gesture")
    print("  [Q] Quit")
    print("=" * 60)

    while current_gesture_idx < len(GESTURES):
        gesture = GESTURES[current_gesture_idx]
        gesture_dir = os.path.join(DATASET_DIR, gesture)

        # Count existing images
        existing = len([f for f in os.listdir(gesture_dir)
                       if f.endswith(('.jpg', '.png'))])
        count = existing

        print(f"\n→ Collecting: [{gesture}] ({count}/{COLLECT_NUM_SAMPLES})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror

            # Convert to RGB and create a MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect hand
            detection_result = hand_landmarker.detect(mp_image)

            hand_crop = None

            if detection_result.hand_landmarks:
                for landmarks in detection_result.hand_landmarks:
                    # Draw landmarks on the real frame (for the user to see)
                    draw_landmarks(frame, landmarks)

                    # CREATE A SKELETON IMAGE ON A BLACK BACKGROUND (for the AI to learn)
                    black_canvas = np.zeros_like(frame)
                    draw_landmarks(black_canvas, landmarks)

                    # Crop the hand region from the skeleton image instead of the real frame
                    hand_crop = crop_hand(black_canvas, landmarks, size=IMG_SIZE)

            # Display info
            status_text = f"Gesture: {gesture} | Count: {count}/{COLLECT_NUM_SAMPLES}"
            mode_text = "AUTO CAPTURE" if auto_capture else "MANUAL"

            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, mode_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 0, 255) if auto_capture else (200, 200, 200), 2)

            if hand_crop is not None:
                # Show crop preview
                preview = cv2.resize(hand_crop, (128, 128))
                frame[10:138, frame.shape[1]-138:frame.shape[1]-10] = preview

            # Auto capture
            if auto_capture and hand_crop is not None:
                current_time = time.time()
                if current_time - last_capture_time >= capture_interval:
                    filename = f"{gesture}_{count:04d}.jpg"
                    filepath = os.path.join(gesture_dir, filename)
                    safe_imwrite(filepath, hand_crop)
                    count += 1
                    last_capture_time = current_time

                    if count >= COLLECT_NUM_SAMPLES:
                        print(f"  ✓ Collected {COLLECT_NUM_SAMPLES} images for [{gesture}]!")
                        auto_capture = False
                        break

            cv2.imshow("Collect Data - Hand Gesture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                hand_landmarker.close()
                print("\n✓ Exited data collection.")
                return
            elif key == ord('s'):
                auto_capture = not auto_capture
                print(f"  {'▶ Started' if auto_capture else '⏸ Paused'} auto capture")
            elif key == ord('c') and hand_crop is not None:
                filename = f"{gesture}_{count:04d}.jpg"
                filepath = os.path.join(gesture_dir, filename)
                safe_imwrite(filepath, hand_crop)
                count += 1
                print(f"  📷 Captured: {filename} ({count}/{COLLECT_NUM_SAMPLES})")
            elif key == ord('n'):
                break

        current_gesture_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    print("\n✓ Data collection complete!")


if __name__ == "__main__":
    collect_gesture_data()

