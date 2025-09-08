import cv2
import mediapipe as mp
import numpy as np
import threading
import time



class MaskSmoother:
    def __init__(
        self,
        blur_size : int,
        min_area : int,
        temporal_alpha : float):
        """
        Smooths binary masks using spatial (Gaussian blur, small
        island removal) and temporal (exponential moving average)
        filtering.

        This is useful for segmentation masks in video sequences
        to remove noise and produce temporally stable masks.

        Parameters
        ----------
            blur_size : int)
                Max Gaussian blur kernel size
                NOTE: must be a positive odd number.
            min_area : int
                Minimum area of connected components to keep.
            temporal_alpha : float
                EMA smoothing factor (0,1). Higher = smoother.
        """
        self.blur_size = blur_size
        self.min_area = min_area
        self.temporal_alpha = temporal_alpha
        self.prev_mask = None


    def _get_valid_blur_size(self, mask):
        """
        Calculate a valid odd-sized Gaussian blur kernel for
        the given mask.

        Ensures the kernel size does not exceed mask dimensions
        and is odd as required by cv2.GaussianBlur.
        """
        h, w = mask.shape

        # Ensure kernel fits in mask.
        size = min(self.blur_size, h // 2 * 2 + 1, w // 2 * 2 + 1)

        # Make sure kernel is odd.
        if size % 2 == 0:
            size += 1

        return size


    def smooth_single(self, mask):
        """
        Smooth a single binary mask.

        Steps:
            1. Threshold mask to binary.
            2. Remove small connected components.
            3. Apply Gaussian blur to smooth edges.
            4. Apply temporal EMA smoothing with previous mask.
        """
        # Step 1: Convert mask to binary.
        bin_mask = (mask > 0.5).astype(np.uint8)

        # Step 2: Remove small islands.
        nb_components, output, stats, _ = \
            cv2.connectedComponentsWithStats(
                bin_mask,
                connectivity=8) # 8-connectivity considers diagonals for component detection.

        cleaned_mask = np.zeros_like(bin_mask)
        for i in range(1, nb_components):  # skip background label 0.
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:  # Keep only sufficiently large regions.
                cleaned_mask[output == i] = 1

        # Step 3: Smooth edges with Gaussian blur.
        blur_size_valid = self._get_valid_blur_size(cleaned_mask)
        smoothed = cv2.GaussianBlur(
            cleaned_mask.astype(np.float32), 
            (blur_size_valid, blur_size_valid),
            0)

        # Ensure mask stays in [0,1].
        smoothed = np.clip(smoothed, 0.0, 1.0)

        # Step 4: Apply temporal EMA smoothing.
        if self.prev_mask is None:
            final_mask = smoothed # First frame has no previous mask.

        else:
            # EMA: smooth previous and current frame masks.
            final_mask = self.temporal_alpha * self.prev_mask + (1 - self.temporal_alpha) * smoothed

        self.prev_mask = final_mask

        return final_mask.astype(np.float32)


    def smooth_masks(self, masks):
        """
        Smooth a list of masks sequentially.

        Applies smooth_single() to each mask in the list.
        Useful for processing video frames.
        """
        smoothed_list = []

        for mask in masks:
            smoothed_list.append(self.smooth_single(mask))

        return smoothed_list


class ThreadedFaceBlender:
    """
    Captures webcam video, detects faces, and blends new video
    segments into the existing video using segmentation masks.

    Runs two threads:
        1. record_new_video - Detect faces and record masked frames.
        2. play_looped_video - Play blended video in a loop.
    """
    def __init__(
            self,
            record_seconds,
            alpha,
            fps,
            blur_size,
            min_area,
            temporal_alpha):
        """
        Initialize the threaded face blender.

        Parameters
        ----------
        record_seconds : int
            Duration in seconds to record new video segments when a face is detected.
        alpha : float
            Blending strength (0-1) for new video frames.
        fps : int
            Frames per second for recording and playback.
        blur_size : int
            Gaussian blur kernel size for mask smoothing.
        min_area : int
            Minimum connected component area for mask cleaning.
        temporal_alpha : float
            Temporal EMA smoothing factor for masks.
        """
        # Add class attributes.
        self.record_seconds = record_seconds
        self.alpha = alpha
        self.fps = fps

        # Initialize smoother for mask processing.
        self.smoother = MaskSmoother(
            blur_size=blur_size,
            min_area=min_area,
            temporal_alpha=temporal_alpha)

        # Set up mediapipe selfie segmentation.
        self.mp_selfie_segmentation = \
            mp.solutions.selfie_segmentation  # type: ignore
        self.selfie_segmentation = \
            self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

        # cv2 video capture.
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        # Mediapipe face detection.
        self.mp_face_detection = mp.solutions.face_detection  # type: ignore
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5)

        # Thread-safe storage for video frames.
        self.current_frames = []

        # Lock for atomic access to current_frames.
        self.lock = threading.Lock()
        self.running = True


    def detect_face(self, frame):
        """
        Detect if a face exists in the given frame.

        Parameters
        ----------
        frame : ndarray
            BGR image frame from webcam.

        Returns
        -------
        bool
            True if at least one face is detected.
        """
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_detection.process(rgb_frame)

        return results.detections is not None


    def blend(
        self,
        new_frames : list[np.ndarray],
        masks: list[np.ndarray]):
        """
        Blend new frames into current frames using masks.

        Each pixel is blended as:
            blended = current * (1 - mask * alpha) + new * (mask * alpha)

        Parameters
        ----------
        new_frames : list[np.ndarray]
            A list of image frames.
        masks : list[np.ndarray]
            A list of image masks.
        """
        if self.current_frames:
            # Compute minimum length to avoid index errors if lists differ.
            min_len = min(len(self.current_frames), len(new_frames))
            blended_frames = []
    
            for i in range(min_len):
                # Ensure mask is 3D for broadcasting with RGB frames.
                mask = masks[i][:, :, np.newaxis] if masks[i].ndim == 2 else masks[i]

                # Convert frames to float32 for blending calculations.
                frame1 = self.current_frames[i].astype(np.float32)
                frame2 = new_frames[i].astype(np.float32)

                # Blend frames using mask and alpha.
                blended = frame1 * (1 - mask * self.alpha) + frame2 * (mask * self.alpha)

                # Clip result to 0-255 and convert back to uint8.
                blended_frames.append(np.clip(blended, 0, 255).astype(np.uint8))

        else:
            blended_frames = new_frames

        # Atomically swap blended frames into shared buffer.
        with self.lock:
            self.current_frames = blended_frames


    def record_new_video(self):
        """
        Continuously capture webcam frames.

            - Detect faces.
            - Record frames when face detected.
            - Generate segmentation masks.
            - Smooth masks.
            - Blend new video into current frames.

        Runs in a separate thread.
        """
        # Main loop: continuously read frames from webcam.
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Check if a face is detected.
            if self.detect_face(frame):
                print("Face detected! Recording new video...")
                new_frames = []
                masks = []
                start_time = time.time()

                # Record for self.record_seconds.
                while time.time() - start_time < self.record_seconds:
                    ret, f = self.cap.read()
                    if not ret:
                        break

                    # Convert frame to RGB for MediaPipe segmentation.
                    rgb_f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    seg_results = self.selfie_segmentation.process(rgb_f)

                    # Get mask and binarize.
                    mask = seg_results.segmentation_mask  # float32, 0..1
                    mask = (mask > 0.5).astype(np.float32)  # binary mask

                    new_frames.append(f)
                    masks.append(mask)

                    # Wait to maintain target FPS.
                    time.sleep(1 / self.fps)

                # Smooth recorded masks.
                smoothed_masks = self.smoother.smooth_masks(masks)

                # Blend new video segment into shared buffer.
                self.blend(new_frames, smoothed_masks)

                print("Blending complete. Now looping the blended video.")

            # Small sleep to avoid busy waiting.
            time.sleep(0.01)


    def play_looped_video(self):
        """
        Play the current blended video in a loop (ping-pong style).

        Displays frames with OpenCV and handles quitting with 'q'.
        """
        # Track direction: 1 = forward, -1 = backward.
        direction = 1
        index = 0

        while self.running:
            frames_copy = None

            # Copy frames atomically to avoid reading while blending.
            with self.lock:
                if self.current_frames:
                    frames_copy = self.current_frames  # Read atomically

            if frames_copy:
    
                # Ping-pong loop: reverse direction when reaching end or start.
                if index >= len(frames_copy):
                    index = len(frames_copy) - 1
                    direction = -1

                elif index < 0:
                    index = 0
                    direction = 1

                frame = frames_copy[index]

                # Display current frame.
                cv2.imshow("Blended Video", frame)

                # Wait for key press to maintain FPS; quit if 'q' pressed.
                key = cv2.waitKey(int(1000 / self.fps))
                if key & 0xFF == ord('q'):
                    self.running = False
                    break

                index += direction
            else:
                time.sleep(0.05)


    def run(self):
        """
        Start threaded video recording and playback.

        Spawns record_new_video in a daemon thread and plays looped video.
        Releases resources on exit.
        """
        try:
            # Start recording thread as daemon.
            threading.Thread(
                target=self.record_new_video,
                daemon=True).start()
            
            # Start main playback loop.
            self.play_looped_video()

        finally:
            # Cleanup resources when done,
            self.cap.release()
            cv2.destroyAllWindows()
