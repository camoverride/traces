import cv2
import logging
import mediapipe as mp
import numpy as np
import os
import random
import threading
import time
import yaml



# Set up logging,
logging.basicConfig(
    level=logging.INFO,
    force=True,
    format='%(levelname)s: %(message)s')

class ThreadedFaceBlender:
    """
    
    """
    def __init__(
            self,
            record_seconds=5,
            alpha=0.5,
            fps=30):
        self.record_seconds = record_seconds
        self.alpha = alpha
        self.fps = fps

        # Cv2 video capture.
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        # Mediapipe face detection.
        self.mp_face_detection = mp.solutions.face_detection  # type: ignore
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5)

        # Shared video frames.
        self.current_frames = []

        # Thread-safe access.
        self.lock = threading.Lock()

        self.running = True


    def detect_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        return results.detections is not None


    def blend(self, new_frames, masks):
        """
        Blend the current frames with new frames using a per-pixel mask.
        Only masked areas are updated. Preparation happens outside the lock to avoid pauses.
        """
        if self.current_frames:
            min_len = min(len(self.current_frames), len(new_frames))
            blended_frames = []
            for i in range(min_len):
                mask = masks[i][:, :, np.newaxis] if masks[i].ndim == 2 else masks[i]
                frame1 = self.current_frames[i].astype(np.float32)
                frame2 = new_frames[i].astype(np.float32)
                blended = frame1 * (1 - mask) + frame2 * mask
                blended_frames.append(np.clip(blended, 0, 255).astype(np.uint8))
        else:
            blended_frames = new_frames

        # Atomically swap frames under lock
        with self.lock:
            self.current_frames = blended_frames


    def record_new_video(self):
        """
        Detect faces and record new frames to blend.
        """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            if self.detect_face(frame):
                print("Face detected! Recording new video...")
                new_frames = []
                start_time = time.time()
                while time.time() - start_time < self.record_seconds:
                    ret, f = self.cap.read()
                    if not ret:
                        break
                    new_frames.append(f)
                    time.sleep(1 / self.fps)  # approximate frame timing

                # Assume all frames have same shape
                h, w, _ = new_frames[0].shape
                radius = min(h, w) // 4
                center = (random.randint(radius, w-radius), random.randint(radius, h-radius))

                masks = []
                for _ in new_frames:
                    mask = np.zeros((h, w), dtype=np.float32)
                    cv2.circle(mask, center, radius, 1, -1)  # circle with value 1 # type: ignore
                    masks.append(mask)

                # Now call the new blend function
                self.blend(new_frames, masks)
                print("Blending complete. Now looping the blended video.")

            time.sleep(0.01)


    def play_looped_video(self):
        """
        Continuously loop the current blended video in ping-pong style.
        """
        direction = 1  # 1 = forward, -1 = backward
        index = 0

        while self.running:
            frames_copy = None
            with self.lock:
                if self.current_frames:
                    frames_copy = self.current_frames  # Read atomically

            if frames_copy:
                # Ensure index is within bounds after video swap
                if index >= len(frames_copy):
                    index = len(frames_copy) - 1
                    direction = -1
                elif index < 0:
                    index = 0
                    direction = 1

                frame = frames_copy[index]
                cv2.imshow("Blended Video", frame)
                key = cv2.waitKey(int(1000 / self.fps))
                if key & 0xFF == ord('q'):
                    self.running = False
                    break

                index += direction
            else:
                time.sleep(0.05)


    def run(self):
        try:
            threading.Thread(
                target=self.record_new_video,
                daemon=True).start()
            self.play_looped_video()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.info("Setting up display")

    # Load the config.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set the display.
    os.environ["DISPLAY"] = ":0"

    # Rotate the screen.
    # NOTE: this works for Pi only.
    os.system(f"wlr-randr --output HDMI-A-1 --transform {config['rotation']}")

    # Hide the cursor.
    os.system("unclutter -idle 0 &")

    # Make the display fullscreen.
    cv2.namedWindow(
        "Blended Video",
        cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Blended Video",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN)

    # Initialize blender object.
    blender = ThreadedFaceBlender(
        record_seconds=config["recording_duration"],
        alpha=config["alpha"],
        fps=config["fps"])

    # Run!
    blender.run()
