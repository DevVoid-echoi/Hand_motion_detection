"""
landmarks.py - Tiện ích MediaPipe dùng chung cho thu thập dữ liệu và predict

QUAN TRỌNG: collect_data.py và predict.py phải dùng chung các hàm này để ảnh
lúc training và lúc predict có cùng đặc trưng hình ảnh.
"""
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .config import HAND_LANDMARKER_PATH

# Index của WRIST và INDEX_FINGER_TIP trong MediaPipe Hands
WRIST_IDX = 0
INDEX_FINGER_TIP_IDX = 8

# Màu sắc riêng cho từng ngón tay (BGR format)
FINGER_COLORS = {
    "thumb":  (0, 128, 255),   # Cam - Ngón cái
    "index":  (0, 255, 0),     # Xanh lá - Ngón trỏ
    "middle": (255, 255, 0),   # Vàng - Ngón giữa
    "ring":   (255, 0, 255),   # Tím - Ngón áp út
    "pinky":  (0, 255, 255),   # Xanh cyan - Ngón út
    "palm":   (200, 200, 200), # Xám - Lòng bàn tay
}

# Kết nối cho từng ngón tay
FINGER_CONNECTIONS = {
    "thumb":  [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index":  [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
    "ring":   [(0, 13), (13, 14), (14, 15), (15, 16)],
    "pinky":  [(0, 17), (17, 18), (18, 19), (19, 20)],
    "palm":   [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)],
}

LINE_THICKNESS = 4   # Nét dày để rõ sau khi resize
JOINT_RADIUS   = 6   # Điểm khớp to


def create_hand_landmarker():
    """
    Khởi tạo MediaPipe HandLandmarker (Tasks API v1.0+).
    Đọc model dưới dạng bytes để tránh lỗi Unicode path trên Windows.
    """
    with open(HAND_LANDMARKER_PATH, "rb") as f:
        model_data = f.read()
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_landmarks(frame, landmarks):
    """
    Vẽ khung xương tay lên frame với màu sắc phân biệt từng ngón tay.
    Mỗi ngón tay có một màu riêng để CNN dễ học đặc trưng hình dạng hơn.
    Nét vẽ dày và điểm to để không bị mất chi tiết khi resize xuống 64x64.
    """
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Vẽ từng ngón tay với màu riêng
    for finger_name, connections in FINGER_CONNECTIONS.items():
        color = FINGER_COLORS[finger_name]
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx],
                         color, LINE_THICKNESS, cv2.LINE_AA)

    # Vẽ các điểm khớp
    for i, point in enumerate(points):
        if i <= 4:    color = FINGER_COLORS["thumb"]
        elif i <= 8:  color = FINGER_COLORS["index"]
        elif i <= 12: color = FINGER_COLORS["middle"]
        elif i <= 16: color = FINGER_COLORS["ring"]
        else:         color = FINGER_COLORS["pinky"]
        cv2.circle(frame, point, JOINT_RADIUS, color, -1, cv2.LINE_AA)
        # Viền trắng xung quanh điểm khớp để nổi bật hơn
        cv2.circle(frame, point, JOINT_RADIUS + 1, (255, 255, 255), 1, cv2.LINE_AA)


def crop_hand(frame, landmarks, padding=40, size=None):
    """
    Cắt vùng chứa bàn tay từ frame dựa trên landmarks.

    Args:
        frame: Ảnh gốc
        landmarks: MediaPipe NormalizedLandmark list
        padding: Pixel mở rộng xung quanh bounding box
        size: Nếu có, resize vùng tay về (size, size)

    Returns:
        Ảnh vùng tay đã crop, hoặc None nếu không valid
    """
    h, w, _ = frame.shape

    # Lấy bounding box từ landmarks
    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]

    x_min = int(max(0, min(x_coords) - padding))
    y_min = int(max(0, min(y_coords) - padding))
    x_max = int(min(w, max(x_coords) + padding))
    y_max = int(min(h, max(y_coords) + padding))

    if x_max - x_min < 10 or y_max - y_min < 10:
        return None

    hand_crop = frame[y_min:y_max, x_min:x_max]
    if size is not None:
        hand_crop = cv2.resize(hand_crop, (size, size))
    return hand_crop
