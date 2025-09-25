import cv2
import mediapipe as mp
import numpy as np
import random
import threading
import time
from typing import Optional



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
            monitor_width,
            monitor_height,
            frame_rotation,
            record_seconds,
            alpha,
            fps,
            blur_size,
            min_area,
            temporal_alpha,
            grid_height : Optional[int],
            grid_width : Optional[int]):
        """
        Initialize the threaded face blender.

        Parameters
        ----------
        monitor_width : int
            The width of the monitor (after physical rotation)
        monitor_height : int
            The height of the monitor (after physical rotation)
        frame_rotation : str
            The camera might be rotated. Rotate the frame accordingly.
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
        grid_height : Optional[int]
            Number of boxes on the height. If None, then grid is disabled.
        grid_width : Optional[int]
            Number of boxes on the width. If None, then grid is disabled.
        """
        # Add class attributes.
        self.monitor_height = monitor_height
        self.monitor_width = monitor_width
        self.frame_rotation = frame_rotation
        self.record_seconds = record_seconds
        self.alpha = alpha
        self.fps = fps
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_enabled = grid_height is not None and grid_width is not None
        self.used_grid_cells = set()

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
        
        # NOTE: hard-coded values for Logitech Brio.
        device_index = 0
        max_width = 4096
        max_height = 2160

        # cv2 video capture.
        self.cap = cv2.VideoCapture(device_index)

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)

        # Set MJPG codec
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # type: ignore
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        # Verify resolution.
        actual_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Requested 1920x1080, got {actual_width}x{actual_height}")

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
        
        If grid is enabled, place the entire new video in a random grid box.
        For first-time grid assignments, skip segmentation completely.
        """
        if self.current_frames:
            min_len = min(len(self.current_frames), len(new_frames))
            blended_frames = []
            
            # Choose random grid position once for the entire video segment
            if self.grid_enabled and self.grid_height and self.grid_width:
                grid_row = random.randint(0, self.grid_height - 1)
                grid_col = random.randint(0, self.grid_width - 1)
                
                # Check if this is the first time using this grid cell
                is_first_assignment = (grid_row, grid_col) not in self.used_grid_cells
                if is_first_assignment:
                    self.used_grid_cells.add((grid_row, grid_col))
                
                # Calculate grid box dimensions
                box_height = self.monitor_height // self.grid_height
                box_width = self.monitor_width // self.grid_width
                
                # Calculate coordinates for the grid box
                y_start = grid_row * box_height
                y_end = min((grid_row + 1) * box_height, self.monitor_height)
                x_start = grid_col * box_width
                x_end = min((grid_col + 1) * box_width, self.monitor_width)
                
                box_height_actual = y_end - y_start
                box_width_actual = x_end - x_start
            
            for i in range(min_len):
                if self.grid_enabled:
                    current_frame = self.current_frames[i].copy()
                    new_frame = new_frames[i]
                    
                    # Resize new frame to fit the grid box
                    resized_new_frame = cv2.resize(new_frame, (box_width_actual, box_height_actual))
                    
                    if is_first_assignment:
                        # First time assignment: place video directly without segmentation
                        current_frame[y_start:y_end, x_start:x_end] = resized_new_frame
                    else:
                        # Subsequent assignment: use segmentation mask for blending
                        mask = masks[i]
                        resized_mask = cv2.resize(mask, (box_width_actual, box_height_actual))
                        
                        # Ensure mask is 3D for broadcasting
                        if resized_mask.ndim == 2:
                            resized_mask = resized_mask[:, :, np.newaxis]
                        
                        # Blend using segmentation mask
                        current_box = current_frame[y_start:y_end, x_start:x_end].astype(np.float32)
                        new_box = resized_new_frame.astype(np.float32)
                        
                        blended_box = current_box * (1 - resized_mask * self.alpha) + new_box * (resized_mask * self.alpha)
                        blended_box = np.clip(blended_box, 0, 255).astype(np.uint8)
                        
                        current_frame[y_start:y_end, x_start:x_end] = blended_box
                    
                    blended_frames.append(current_frame)
                    
                else:
                    # Original blending logic (non-grid mode)
                    mask = masks[i][:, :, np.newaxis] if masks[i].ndim == 2 else masks[i]
                    frame1 = self.current_frames[i].astype(np.float32)
                    frame2 = new_frames[i].astype(np.float32)
                    blended = frame1 * (1 - mask * self.alpha) + frame2 * (mask * self.alpha)
                    blended_frames.append(np.clip(blended, 0, 255).astype(np.uint8))

        else:
            if self.grid_enabled and self.grid_height and self.grid_width:
                # For first frame in grid mode, choose random position
                grid_row = random.randint(0, self.grid_height - 1)
                grid_col = random.randint(0, self.grid_width - 1)
                self.used_grid_cells.add((grid_row, grid_col))
                
                box_height = self.monitor_height // self.grid_height
                box_width = self.monitor_width // self.grid_width
                
                y_start = grid_row * box_height
                y_end = min((grid_row + 1) * box_height, self.monitor_height)
                x_start = grid_col * box_width
                x_end = min((grid_col + 1) * box_width, self.monitor_width)
                
                box_height_actual = y_end - y_start
                box_width_actual = x_end - x_start
                
                blended_frames = []
                for new_frame in new_frames:
                    empty_frame = np.zeros((self.monitor_height, self.monitor_width, 3), dtype=np.uint8)
                    
                    # Resize and place new frame in grid box (no segmentation for first assignment)
                    resized_new_frame = cv2.resize(new_frame, (box_width_actual, box_height_actual))
                    empty_frame[y_start:y_end, x_start:x_end] = resized_new_frame
                    blended_frames.append(empty_frame)
                
            else:
                blended_frames = new_frames

        # Atomically swap blended frames into shared buffer
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
        failed_reads = 0
        MAX_FAILED_READS = 3000

        # Main loop: continuously read frames from webcam.
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                # Check to see if camera fails.
                failed_reads += 1
                print(f"Warning: Failed to read frame {failed_reads} times in a row")
                time.sleep(0.1)

                if failed_reads >= MAX_FAILED_READS:
                    print("Too many failed reads, attempting to reset camera")
                    self.cap.release()
                    failed_reads = 0
                continue

            failed_reads = 0

            # Rotate the image, as the camera itself might be rotated.
            if self.frame_rotation == "right":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.frame_rotation == "left":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif self.frame_rotation == "normal":
                pass

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

                    # Rotate the image, as the camera itself might be rotated.
                    if self.frame_rotation == "right":
                        f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
                    elif self.frame_rotation == "left":
                        f = cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif self.frame_rotation == "normal":
                        pass

                    # Resize for mediapipe processing, downscale to width=320.
                    mediapipe_width = 320
                    f_h, f_w = f.shape[:2]
                    scale = mediapipe_width / f_w
                    new_height = int(f_h * scale)
                    mediapipe_f = cv2.resize(f, (mediapipe_width, new_height))

                    # Convert frame to RGB for MediaPipe segmentation.
                    rgb_f = cv2.cvtColor(mediapipe_f, cv2.COLOR_BGR2RGB)

                    # Perform image segmentation.
                    seg_results = self.selfie_segmentation.process(rgb_f)

                    # Get mask and binarize.
                    mask = seg_results.segmentation_mask  # float32, 0..1
                    mask = (mask > 0.5).astype(np.float32)  # binary mask

                    # Resize display frame correct monitor dimensions.
                    f = cv2.resize(f, (self.monitor_width, self.monitor_height))

                    # Resize mask to correct monitor dimensions.
                    # mask = cv2.resize(mask, (f_w, f_h))
                    mask = cv2.resize(mask, (self.monitor_width, self.monitor_height))

                    # Append the results.
                    new_frames.append(f)
                    masks.append(mask)

                    # Wait to maintain target FPS.
                    time.sleep(1 / self.fps)

                # Smooth recorded masks.
                smoothed_masks = self.smoother.smooth_masks(masks)

                start_time = time.time()
                # Blend new video segment into shared buffer.
                self.blend(new_frames, smoothed_masks)
                end_time = time.time()

                print(f"Blending complete in {end_time - start_time:.3f}. Now looping the blended video.")

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
                cv2.imshow("Display Image", frame)

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
