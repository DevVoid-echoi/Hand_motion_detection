"""
gesture_controller.py - Map hand gestures to computer control actions
"""
import time
import pyautogui
from .config import COOLDOWN_SECONDS

# Keep pyautogui failsafe on (move the mouse to a screen corner to abort)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # Set to 0 to remove the default 50ms delay entirely


class GestureController:
    """
    Control the computer based on hand gestures.

    Mapping:
        fist        → Play/Pause media
        open_palm   → Mute/Unmute
        thumbs_up   → Volume up
        thumbs_down → Volume down
        peace       → Take a screenshot
        point       → (Mouse movement - handled separately)
        swipe_left  → Previous slide (Left key)
        swipe_right → Next slide (Right key)
    """

    def __init__(self):
        self._last_action_time = {}  # gesture → last_executed_time
        self.last_gesture = None     # Track last gesture for edge-triggering
        self._volume_controller = None
        self._init_volume_control()

        # State for cursor smoothing (Exponential Moving Average)
        self.prev_x = 0
        self.prev_y = 0
        self.smoothing_factor = 0.6  # Raised to 0.6 so the cursor responds faster and follows the hand more closely

    def _init_volume_control(self):
        """Initialize pycaw for Windows volume control."""
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            self._volume_controller = cast(
                interface, POINTER(IAudioEndpointVolume)
            )
            print("✓ Volume controller initialized")
        except Exception as e:
            print(f"⚠ Could not initialize volume controller: {e}")
            print("  Falling back to media keys.")
            self._volume_controller = None

    def _is_cooldown(self, gesture):
        """Check whether the gesture is still in its cooldown period."""
        # Cursor movement ('point') needs continuous updates (60 FPS), so NO cooldown
        if gesture == "point":
            return False

        now = time.time()
        last_time = self._last_action_time.get(gesture, 0)

        # Volume should repeat continuously but faster (0.3s instead of 1s)
        cooldown_limit = 0.3 if gesture in ["thumbs_up", "thumbs_down"] else COOLDOWN_SECONDS

        if now - last_time < cooldown_limit:
            return True
        self._last_action_time[gesture] = now
        return False

    def execute(self, gesture, extra_data=None):
        """
        Execute the action for the given gesture.

        Args:
            gesture: Gesture name (string)
            extra_data: Additional data (e.g. fingertip coordinates for 'point')

        Returns:
            Description of the action performed, or None if on cooldown
        """
        # Hand lost or no gesture -> reset state
        if not gesture:
            self.last_gesture = None
            return None

        # Gestures that fire only once until the hand is lowered
        one_shot_gestures = ["fist", "open_palm", "peace", "pinch", "swipe_left", "swipe_right"]

        # If this is a one-shot gesture and IDENTICAL to the previous frame's gesture -> skip
        if gesture in one_shot_gestures and gesture == self.last_gesture:
            return None

        # Update the current gesture state
        self.last_gesture = gesture

        # Check cooldown (debounce against jitter)
        if self._is_cooldown(gesture):
            return None

        action_map = {
            "fist": self._play_pause,
            "open_palm": self._mute_unmute,
            "thumbs_up": self._volume_up,
            "thumbs_down": self._volume_down,
            "peace": self._screenshot,
            "point": self._move_cursor,
            "swipe_left": self._slide_previous,
            "swipe_right": self._slide_next,
            "pinch": self._left_click,
        }

        action_func = action_map.get(gesture)
        if action_func:
            return action_func(extra_data)
        return None

    # ==================== Actions ====================

    def _play_pause(self, _=None):
        """Play/Pause media."""
        pyautogui.press("playpause")
        return "▶⏸ Play/Pause"

    def _mute_unmute(self, _=None):
        """Mute/Unmute."""
        if self._volume_controller:
            muted = self._volume_controller.GetMute()
            self._volume_controller.SetMute(not muted, None)
            return "🔇 Mute" if not muted else "🔊 Unmute"
        else:
            pyautogui.press("volumemute")
            return "🔇 Toggle Mute"

    def _volume_up(self, _=None):
        """Increase volume by 5%."""
        if self._volume_controller:
            current = self._volume_controller.GetMasterVolumeLevelScalar()
            new_vol = min(1.0, current + 0.05)
            self._volume_controller.SetMasterVolumeLevelScalar(new_vol, None)
            return f"🔊 Volume: {new_vol:.0%}"
        else:
            pyautogui.press("volumeup")
            return "🔊 Volume Up"

    def _volume_down(self, _=None):
        """Decrease volume by 5%."""
        if self._volume_controller:
            current = self._volume_controller.GetMasterVolumeLevelScalar()
            new_vol = max(0.0, current - 0.05)
            self._volume_controller.SetMasterVolumeLevelScalar(new_vol, None)
            return f"🔉 Volume: {new_vol:.0%}"
        else:
            pyautogui.press("volumedown")
            return "🔉 Volume Down"

    def _screenshot(self, _=None):
        """Take a screenshot."""
        pyautogui.hotkey("win", "shift", "s")  # Windows Snipping Tool
        return "📸 Screenshot"

    def _move_cursor(self, extra_data=None):
        """Move the cursor to the index fingertip position (smoothed with EMA)."""
        if extra_data and "x" in extra_data and "y" in extra_data:
            screen_w, screen_h = pyautogui.size()
            # Define an "Active Region" in the middle of the camera view.
            # Drop a 20% margin on each side; the central 60% maps to the whole screen.
            margin_x = 0.2
            margin_y = 0.2
            active_w = 1.0 - 2 * margin_x
            active_h = 1.0 - 2 * margin_y

            # Compute the hand's relative position within the active region
            norm_x = (extra_data["x"] - margin_x) / active_w
            norm_y = (extra_data["y"] - margin_y) / active_h

            # Clamp to [0, 1] so the cursor doesn't fly past the screen edges
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))

            # Raw (amplified) coordinates
            raw_x = norm_x * screen_w
            raw_y = norm_y * screen_h

            # Initialize on the first movement
            if self.prev_x == 0 and self.prev_y == 0:
                self.prev_x, self.prev_y = raw_x, raw_y

            # Apply Exponential Moving Average (EMA)
            smooth_x = self.prev_x + self.smoothing_factor * (raw_x - self.prev_x)
            smooth_y = self.prev_y + self.smoothing_factor * (raw_y - self.prev_y)

            # Update history
            self.prev_x, self.prev_y = smooth_x, smooth_y

            # Actually move the cursor (duration=0 to avoid pyautogui delay)
            pyautogui.moveTo(int(smooth_x), int(smooth_y), duration=0)
            return f"🖱 Cursor → ({int(smooth_x)}, {int(smooth_y)})"
        return None

    def _slide_previous(self, _=None):
        """Go to the previous slide (Left Arrow key)."""
        pyautogui.press("left")
        return "⬅ Previous slide"

    def _slide_next(self, _=None):
        """Go to the next slide (Right Arrow key)."""
        pyautogui.press("right")
        return "➡ Next slide"

    def _left_click(self, _=None):
        """Left mouse click."""
        pyautogui.click()
        return "🖱 Left Click"


if __name__ == "__main__":
    print("Test GestureController:")
    controller = GestureController()

    # Test each gesture
    for gesture in ["fist", "thumbs_up", "thumbs_down", "swipe_left", "swipe_right"]:
        result = controller.execute(gesture)
        print(f"  {gesture}: {result}")
        time.sleep(0.2)
