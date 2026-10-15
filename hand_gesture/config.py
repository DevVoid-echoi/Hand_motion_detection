"""
config.py - Shared configuration for the Hand Gesture Recognition project
"""
import os

# ==================== Paths ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
HAND_LANDMARKER_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

# ==================== Gestures ====================
GESTURES = [
    "fist",           # Closed fist
    "open_palm",      # Open palm
    "thumbs_up",      # Thumb up
    "thumbs_down",    # Thumb down
    "peace",          # Two fingers (V sign)
    "point",          # Pointing (index finger)
    "swipe_left",     # Swipe left
    "swipe_right",    # Swipe right
    "pinch",          # Pinch fingers (mouse click)
]

NUM_CLASSES = len(GESTURES)

# ==================== Image ====================
IMG_SIZE = 64           # Input image size (64x64)
IMG_CHANNELS = 3        # RGB

# ==================== Training Hyperparameters ====================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8       # 80% train, 20% validation
EARLY_STOP_PATIENCE = 7 # Stop early if val_loss doesn't improve for N epochs

# ==================== Data Augmentation ====================
AUGMENTATION = {
    "random_horizontal_flip": 0.5,
    "random_rotation": 15,        # Rotate up to 15 degrees
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
}

# ==================== Data Collection ====================
COLLECT_NUM_SAMPLES = 200   # Number of images to collect per gesture
CAMERA_INDEX = 0            # Camera index (0 = default webcam)

# ==================== Prediction ====================
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence required to execute an action
COOLDOWN_SECONDS = 1.0      # Wait time between two executions of an action

# ==================== Swipe Detection ====================
SWIPE_DISTANCE_THRESHOLD = 80   # Minimum pixels to count as a swipe
SWIPE_TIME_WINDOW = 0.5         # Seconds - maximum time to complete a swipe
