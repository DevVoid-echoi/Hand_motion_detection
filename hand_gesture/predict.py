"""
predict.py - Real-time prediction từ webcam
Dùng MediaPipe Tasks API (v1.0+) detect tay → crop → CNN predict → trả về gesture + confidence
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
    """Dự đoán cử chỉ tay real-time từ webcam."""

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
        
        print(f"✓ Model loaded trên {self.device}")

    def _detect_swipe(self, landmarks, frame_width):
        """
        Detect vuốt trái/phải dựa trên vị trí cổ tay qua thời gian.
        
        Returns:
            "swipe_left", "swipe_right", hoặc None
        """
        wrist = landmarks[WRIST_IDX]
        current_x = wrist.x * frame_width
        current_time = time.time()

        self._wrist_history.append((current_x, current_time))

        # Xóa lịch sử cũ
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
        Dự đoán cử chỉ từ 1 frame.
        
        Args:
            frame: BGR frame từ OpenCV
            
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
                # Vẽ landmarks
                draw_landmarks(frame, landmarks)
                
                # Trích xuất tọa độ ngón trỏ
                index_tip = landmarks[INDEX_FINGER_TIP_IDX]
                extra_data = {"x": index_tip.x, "y": index_tip.y}

                # Kiểm tra swipe trước
                swipe = self._detect_swipe(landmarks, frame.shape[1])
                if swipe:
                    gesture_name = swipe
                    confidence = 1.0
                    break

                # Nếu không phải swipe → dùng CNN predict
                # TẠO ẢNH KHUNG XƯƠNG TRÊN NỀN ĐEN
                black_canvas = np.zeros_like(frame)
                draw_landmarks(black_canvas, landmarks)
                
                # Cắt vùng khung xương đưa cho CNN
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
                            # Không cho phép CNN trigger các hành động động (vuốt) khi đang đứng yên
                            # Bỏ qua hoàn toàn để không kích hoạt nhầm chức năng khác
                            if pred_name in ["swipe_left", "swipe_right"]:
                                gesture_name = None
                            else:
                                gesture_name = pred_name

        # Vẽ kết quả lên frame
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
        """Giải phóng resources."""
        self.hand_landmarker.close()


def run_prediction_demo():
    """Chạy demo nhận diện real-time (chỉ predict, không điều khiển)."""
    predictor = HandGesturePredictor()
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("✗ Không mở được webcam!")
        return

    print("\n  Demo nhận diện cử chỉ - Nhấn [Q] để thoát\n")

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
    """Chạy chế độ điều khiển máy tính thực tế."""
    from .gesture_controller import GestureController
    
    predictor = HandGesturePredictor()
    controller = GestureController()
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("✗ Không mở được webcam!")
        return

    print("\n  Đang chạy Controller - Nhấn [Q] để thoát\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gesture, confidence, extra_data, annotated = predictor.predict_frame(frame)

        # Luôn gọi execute (kể cả khi gesture là None) để Controller reset trạng thái tay
        action_result = controller.execute(gesture, extra_data)

        if gesture:
            if action_result:
                print(f"  → {gesture} ({confidence:.0%}) | Action: {action_result}")
                # Hiện action trên màn hình
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
