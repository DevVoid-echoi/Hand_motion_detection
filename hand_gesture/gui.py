"""
gui.py - Tkinter GUI: data collection, training and running the controller
"""
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
import os

class RedirectText:
    def __init__(self, text_ctrl):
        self.output = text_ctrl

    def write(self, string):
        self.output.insert(tk.END, string)
        self.output.see(tk.END)

    def flush(self):
        pass

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Controller AI")
        self.root.geometry("750x650")
        self.root.configure(padx=20, pady=20)

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 11), padding=10)
        style.configure('Small.TButton', font=('Segoe UI', 9), padding=5)
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))

        # Title
        title_label = ttk.Label(root, text="🖐 Control Your Computer With Hand Gestures", style='Title.TLabel')
        title_label.pack(pady=(0, 10))

        # === Quick guide ===
        guide_frame = ttk.LabelFrame(root, text="📖 Quick guide", padding=10)
        guide_frame.pack(fill=tk.X, pady=(0, 10))

        guide_text = (
            "Step 1: Click 'Collect Data' → The camera window opens → "
            "Hold your hand in front of the camera → Press S to start auto capture → "
            "Press N to move to the next gesture → Press Q when done.\n"
            "Step 2: Click 'Check Dataset' to see whether you have enough images.\n"
            "Step 3: Click 'Train Model' → Wait until it finishes.\n"
            "Step 4: Click 'Run Controller' → Raise your hand to control the computer!"
        )
        ttk.Label(guide_frame, text=guide_text, wraplength=680, font=('Segoe UI', 9)).pack()

        # Frame for main buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill=tk.X, pady=5)

        # Main buttons
        self.btn_collect = ttk.Button(btn_frame, text="📸 1. Collect Data", command=self.run_collect)
        self.btn_collect.pack(side=tk.LEFT, expand=True, padx=3)

        self.btn_train = ttk.Button(btn_frame, text="🧠 2. Train Model", command=self.run_train)
        self.btn_train.pack(side=tk.LEFT, expand=True, padx=3)

        self.btn_run = ttk.Button(btn_frame, text="🚀 3. Run Controller", command=self.run_controller)
        self.btn_run.pack(side=tk.LEFT, expand=True, padx=3)

        # Frame for utility buttons
        util_frame = ttk.Frame(root)
        util_frame.pack(fill=tk.X, pady=(5, 0))

        self.btn_check = ttk.Button(util_frame, text="📊 Check Dataset", command=self.check_dataset, style='Small.TButton')
        self.btn_check.pack(side=tk.LEFT, padx=3)

        self.btn_clear = ttk.Button(util_frame, text="🗑 Clear Console", command=self.clear_console, style='Small.TButton')
        self.btn_clear.pack(side=tk.LEFT, padx=3)

        # Console output area
        lbl_console = ttk.Label(root, text="Logs & Output:", font=('Segoe UI', 10, 'bold'))
        lbl_console.pack(anchor=tk.W, pady=(10, 5))

        self.console = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15, font=('Consolas', 10), bg="#1e1e1e", fg="#00ff00")
        self.console.pack(fill=tk.BOTH, expand=True)

        # Redirect stdout and stderr
        redir = RedirectText(self.console)
        sys.stdout = redir
        sys.stderr = redir

        print("=" * 60)
        print("  Welcome! The system is ready.")
        print("=" * 60)
        print("")
        print("When the camera window opens (Step 1):")
        print("  [S] = Toggle auto capture (most important!)")
        print("  [C] = Capture one image manually")
        print("  [N] = Move to the next gesture")
        print("  [Q] = Quit")
        print("")

    def _run_script_in_thread(self, target_func, script_name):
        """Run the function in a thread so the GUI doesn't block."""
        def task():
            print(f"[{script_name}] Starting...")
            self.btn_collect.config(state=tk.DISABLED)
            self.btn_train.config(state=tk.DISABLED)
            self.btn_run.config(state=tk.DISABLED)

            try:
                target_func()
            except Exception as e:
                print(f"[ERROR] {e}")
            finally:
                print(f"[{script_name}] Finished or closed.\n")
                self.btn_collect.config(state=tk.NORMAL)
                self.btn_train.config(state=tk.NORMAL)
                self.btn_run.config(state=tk.NORMAL)

        threading.Thread(target=task, daemon=True).start()

    def check_dataset(self):
        """Check and print the number of collected images per gesture."""
        from .config import DATASET_DIR, GESTURES, COLLECT_NUM_SAMPLES

        print("=" * 50)
        print("  DATASET CHECK")
        print("=" * 50)

        total = 0
        all_ready = True

        for gesture in GESTURES:
            gesture_dir = os.path.join(DATASET_DIR, gesture)
            if os.path.exists(gesture_dir):
                count = len([f for f in os.listdir(gesture_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            else:
                count = 0

            total += count
            status = "✓" if count >= COLLECT_NUM_SAMPLES else "✗"
            if count < COLLECT_NUM_SAMPLES:
                all_ready = False

            bar_len = min(count, COLLECT_NUM_SAMPLES) * 20 // max(COLLECT_NUM_SAMPLES, 1)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {status} {gesture:15s} [{bar}] {count:4d}/{COLLECT_NUM_SAMPLES}")

        print(f"\n  Total: {total} images")

        if total == 0:
            print("\n  ⚠ NO IMAGES YET!")
            print("  → Click '📸 1. Collect Data'")
            print("  → When the camera opens, press S to start auto capture")
        elif all_ready:
            print("\n  ✓ Dataset is ready! You can move on to training.")
        else:
            print("\n  ⚠ Some gestures don't have enough images.")
            print("  → Click '📸 Collect Data' to capture more.")

        print("=" * 50 + "\n")

    def clear_console(self):
        """Clear the console."""
        self.console.delete(1.0, tk.END)

    def run_collect(self):
        print("─" * 50)
        print("  DATA COLLECTION GUIDE:")
        print("  1. The camera window will open")
        print("  2. Hold your hand in front of the camera")
        print("  3. Press S to START auto capture")
        print("  4. Keep changing your hand angle while capturing")
        print("  5. Press N to move to the next gesture")
        print("  6. Press Q when everything is done")
        print("─" * 50)
        from .collect_data import collect_gesture_data
        self._run_script_in_thread(collect_gesture_data, "Collect Data")

    def run_train(self):
        from .train import train_model
        self._run_script_in_thread(train_model, "Train Model")

    def run_controller(self):
        from .predict import run_realtime_controller
        self._run_script_in_thread(run_realtime_controller, "Run Controller")

def main():
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
