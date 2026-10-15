"""
predict.py - Real-time prediction from the webcam
Uses the MediaPipe Tasks API (v1.0+) to detect the hand → crop → CNN predict → return gesture + confidence
"""
import time
import cv2
import torch
import numpy as np
import mediapipe as mp
from PIL import Image
from .model import load_model
from .dataset import get_inference_transform
from .config import (
    MODEL_SAVE_PATH, GESTURES, CAMERA_INDEX, CONFIDENCE_THRESHOLD,
    SWIPE_DISTANCE_THRESHOLD, SWIPE_TIME_WINDOW
)
from .landmarks import (
    create_hand_landmarker, draw_landmarks, crop_hand,
    WRIST_IDX, INDEX_FINGER_TIP_IDX
)


class HandGesturePredictor:
    """Predict hand gestures in real time from the webcam."""

    def __init__(self, model_path=MODEL_SAVE_PATH):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CNN model
        self.model = load_model(model_path, device=self.device)
        self.transform = get_inference_transform()

        # MediaPipe HandLandmarker (Tasks API v1.0+)
        self.hand_landmarker = create_hand_landmarker()

        # Swipe tracking
        self._wrist_history = []  # list of (x, timestamp)

        print(f"✓ Model loaded on {self.device}")

    def _detect_swipe(self, landmarks, frame_width):
        """
        Detect left/right swipes from the wrist position over time.

        Returns:
            "swipe_left", "swipe_right", or None
        """
        wrist = landmarks[WRIST_IDX]
        current_x = wrist.x * frame_width
        current_time = time.time()

        self._wrist_history.append((current_x, current_time))

        # Drop old history
        self._wrist_history = [
            (x, t) for x, t in self._wrist_history
            if current_time - t <= SWIPE_TIME_WINDOW
        ]

        if len(self._wrist_history) >= 5:
            start_x = self._wrist_history[0][0]
            delta_x = current_x - start_x

            if abs(delta_x) >= SWIPE_DISTANCE_THRESHOLD:
                self._wrist_history.clear()
                if delta_x > 0:
                    return "swipe_right"
                else:
                    return "swipe_left"

        return None

    def predict_frame(self, frame):
        """
        Predict the gesture from a single frame.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            (gesture_name, confidence, extra_data, annotated_frame)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.hand_landmarker.detect(mp_image)

        gesture_name = None
        confidence = 0.0
        extra_data = None

        if detection_result.hand_landmarks:
            for landmarks in detection_result.hand_landmarks:
                # Draw landmarks
                draw_landmarks(frame, landmarks)

                # Extract index fingertip coordinates
                index_tip = landmarks[INDEX_FINGER_TIP_IDX]
                extra_data = {"x": index_tip.x, "y": index_tip.y}

                # Check for a swipe first
                swipe = self._detect_swipe(landmarks, frame.shape[1])
                if swipe:
                    gesture_name = swipe
                    confidence = 1.0
                    break

                # Not a swipe → use the CNN
                # CREATE A SKELETON IMAGE ON A BLACK BACKGROUND
                black_canvas = np.zeros_like(frame)
                draw_landmarks(black_canvas, landmarks)

                # Crop the skeleton region for the CNN
                hand_crop = crop_hand(black_canvas, landmarks)
                if hand_crop is not None:
                    # Preprocess
                    pil_image = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
                    input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

                    # Predict
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        conf, predicted = torch.max(probabilities, 1)

                        confidence = conf.item()
                        if confidence >= CONFIDENCE_THRESHOLD:
                            pred_name = GESTURES[predicted.item()]
                            # Don't let the CNN trigger motion gestures (swipes) while the hand is still
                            # Ignore them entirely to avoid accidentally triggering other actions
                            if pred_name in ["swipe_left", "swipe_right"]:
                                gesture_name = None
                            else:
                                gesture_name = pred_name

        # Draw the result on the frame
        if gesture_name:
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
            text = f"{gesture_name} ({confidence:.0%})"
            cv2.putText(frame, text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        else:
            cv2.putText(frame, "No gesture detected", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        return gesture_name, confidence, extra_data, frame

    def close(self):
        """Release resources."""
        self.hand_landmarker.close()


def run_prediction_demo():
    """Run the real-time recognition demo (prediction only, no control)."""
    predictor = HandGesturePredictor()
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("✗ Could not open webcam!")
        return

    print("\n  Gesture recognition demo - Press [Q] to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gesture, confidence, extra, annotated = predictor.predict_frame(frame)

        if gesture:
            print(f"  → {gesture} ({confidence:.0%})")

        cv2.imshow("Hand Gesture Prediction", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    predictor.close()


def run_realtime_controller():
    """Run the actual computer control mode."""
    from .gesture_controller import GestureController

    predictor = HandGesturePredictor()
    controller = GestureController()
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("✗ Could not open webcam!")
        return

    print("\n  Controller running - Press [Q] to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gesture, confidence, extra_data, annotated = predictor.predict_frame(frame)

        # Always call execute (even when gesture is None) so the controller resets its hand state
        action_result = controller.execute(gesture, extra_data)

        if gesture:
            if action_result:
                print(f"  → {gesture} ({confidence:.0%}) | Action: {action_result}")
                # Show the action on screen
                cv2.putText(annotated, action_result, (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cv2.imshow("Hand Gesture Controller", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    predictor.close()


if __name__ == "__main__":
    # Demo
    run_prediction_demo()
