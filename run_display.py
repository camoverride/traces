import cv2
import logging
import mediapipe as mp
import numpy as np
import os
import threading
import time
import yaml



class MaskSmoother:
    def __init__(self, blur_size=11, min_area=500, temporal_alpha=0.8):
        """
        Args:
            blur_size (int): Max Gaussian blur kernel size (odd number)
            min_area (int): Minimum area of connected components to keep
            temporal_alpha (float): EMA smoothing factor (0..1). Higher = smoother
        """
        self.blur_size = blur_size
        self.min_area = min_area
        self.temporal_alpha = temporal_alpha
        self.prev_mask = None

    def _get_valid_blur_size(self, mask):
        h, w = mask.shape
        size = min(self.blur_size, h // 2 * 2 + 1, w // 2 * 2 + 1)
        if size % 2 == 0:
            size += 1
        return size

    def smooth_single(self, mask):
        """
        Smooth a single mask: remove islands, blur edges, apply temporal smoothing.
        """
        # Ensure binary mask
        bin_mask = (mask > 0.5).astype(np.uint8)

        # Remove small islands
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        cleaned_mask = np.zeros_like(bin_mask)
        for i in range(1, nb_components):  # skip background
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                cleaned_mask[output == i] = 1

        # Smooth edges with Gaussian blur
        blur_size_valid = self._get_valid_blur_size(cleaned_mask)
        smoothed = cv2.GaussianBlur(cleaned_mask.astype(np.float32), (blur_size_valid, blur_size_valid), 0)
        smoothed = np.clip(smoothed, 0.0, 1.0)

        # Temporal smoothing (EMA)
        if self.prev_mask is None:
            final_mask = smoothed
        else:
            final_mask = self.temporal_alpha * self.prev_mask + (1 - self.temporal_alpha) * smoothed

        self.prev_mask = final_mask
        return final_mask.astype(np.float32)

    def smooth_masks(self, masks):
        """
        Smooth a list of masks.
        """
        smoothed_list = []
        for mask in masks:
            smoothed_list.append(self.smooth_single(mask))
        return smoothed_list




smoother = MaskSmoother(blur_size=11, min_area=500, temporal_alpha=0.8)



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

        # Mediapipe selfie segmentation
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation  # type: ignore
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)


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
                blended = frame1 * (1 - mask * self.alpha) + frame2 * (mask * self.alpha)

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
                masks = []
                start_time = time.time()
                while time.time() - start_time < self.record_seconds:
                    ret, f = self.cap.read()
                    if not ret:
                        break
                    # Get segmentation mask
                    rgb_f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    seg_results = self.selfie_segmentation.process(rgb_f)
                    mask = seg_results.segmentation_mask  # float32, 0..1
                    mask = (mask > 0.5).astype(np.float32)  # binary mask

                    new_frames.append(f)
                    masks.append(mask)

                    time.sleep(1 / self.fps)

                smoothed_masks = smoother.smooth_masks(masks)

                # Now call the new blend function
                self.blend(new_frames, smoothed_masks)
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
