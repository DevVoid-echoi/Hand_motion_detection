"""
landmarks.py - MediaPipe utilities shared by data collection and prediction

IMPORTANT: collect_data.py and predict.py must use the same functions so that
training images and prediction images have the same visual features.
"""
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .config import HAND_LANDMARKER_PATH

# Indices of WRIST and INDEX_FINGER_TIP in MediaPipe Hands
WRIST_IDX = 0
INDEX_FINGER_TIP_IDX = 8

# Distinct color for each finger (BGR format)
FINGER_COLORS = {
    "thumb":  (0, 128, 255),   # Orange - Thumb
    "index":  (0, 255, 0),     # Green - Index finger
    "middle": (255, 255, 0),   # Yellow - Middle finger
    "ring":   (255, 0, 255),   # Purple - Ring finger
    "pinky":  (0, 255, 255),   # Cyan - Pinky
    "palm":   (200, 200, 200), # Gray - Palm
}

# Connections for each finger
FINGER_CONNECTIONS = {
    "thumb":  [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index":  [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
    "ring":   [(0, 13), (13, 14), (14, 15), (15, 16)],
    "pinky":  [(0, 17), (17, 18), (18, 19), (19, 20)],
    "palm":   [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)],
}

LINE_THICKNESS = 4   # Thick lines so they stay visible after resizing
JOINT_RADIUS   = 6   # Large joint dots


def create_hand_landmarker():
    """
    Create a MediaPipe HandLandmarker (Tasks API v1.0+).
    Reads the model as bytes to avoid Unicode path errors on Windows.
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
    Draw the hand skeleton on the frame with a distinct color per finger.
    Each finger has its own color so the CNN learns shape features more easily.
    Thick lines and large dots keep details from being lost when resized to 64x64.
    """
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # Draw each finger in its own color
    for finger_name, connections in FINGER_CONNECTIONS.items():
        color = FINGER_COLORS[finger_name]
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx],
                         color, LINE_THICKNESS, cv2.LINE_AA)

    # Draw the joints
    for i, point in enumerate(points):
        if i <= 4:    color = FINGER_COLORS["thumb"]
        elif i <= 8:  color = FINGER_COLORS["index"]
        elif i <= 12: color = FINGER_COLORS["middle"]
        elif i <= 16: color = FINGER_COLORS["ring"]
        else:         color = FINGER_COLORS["pinky"]
        cv2.circle(frame, point, JOINT_RADIUS, color, -1, cv2.LINE_AA)
        # White outline around each joint so it stands out
        cv2.circle(frame, point, JOINT_RADIUS + 1, (255, 255, 255), 1, cv2.LINE_AA)


def crop_hand(frame, landmarks, padding=40, size=None):
    """
    Crop the hand region from the frame based on the landmarks.

    Args:
        frame: Source image
        landmarks: MediaPipe NormalizedLandmark list
        padding: Pixels to expand around the bounding box
        size: If given, resize the hand region to (size, size)

    Returns:
        The cropped hand image, or None if invalid
    """
    h, w, _ = frame.shape

    # Bounding box from landmarks
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
