import numpy as np
import cv2
import os
from datetime import datetime
import time as t
import yaml
from picamera2 import Picamera2
import mediapipe as mp
import psutil
from image_processing import process_image

# Global Debug Variable
SHOW_BB_DEBUG = True  # Set this to True to show bounding boxes

# Read data from the config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
WIDTH = config["WIDTH"]
HEIGHT = config["HEIGHT"]
PLAY_DIR = config["PLAY_DIR"]
CONFIDENCE_THRESHOLD = config["CONFIDENCE_THRESHOLD"]
CAPTURE_DURATION = 3  # Record for 3 seconds
FPS = 15  # 15 frames per second
ALPHA = config.get("ALPHA", 0.5)  # Alpha value for blending (default to 0.5 if not set in config)

# Initialize the picamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)}))
picam2.start()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def print_memory_usage(label):
    """
    Print memory use for debugging.
    NOTE: this code no longer has memory leak issues!
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{label}] Memory Usage: RSS = {mem_info.rss / (1024 * 1024):.2f} MB")


def capture_frames(frame_count):
    """
    Capture a series of frames.
    """
    frames = []
    for _ in range(frame_count):
        frame = picam2.capture_array()
        frames.append(frame)

    return np.array(frames)


def alpha_blend_frames(new_frames, current_frames, alpha):
    """
    Alpha blend two images.
    """
    blended_frames = []
    for i in range(new_frames.shape[0]):
        blended_frame = cv2.addWeighted(new_frames[i], alpha, current_frames[i], 1 - alpha, 0)
        blended_frames.append(blended_frame)

    return np.array(blended_frames)


def save_output_video(output_video_path, output_frames, fps):
    """
    Save the output video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (WIDTH, HEIGHT))
    for i, frame in enumerate(output_frames):
        out.write(frame)

    out.release()


def has_valid_detections(results, confidence_threshold):
    if results.detections:
        for detection in results.detections:
            if detection.score and detection.score[0] >= confidence_threshold:
                return True
    return False


# Initial capture to create the first video (not yet a composite)
frame_count = int(CAPTURE_DURATION * FPS)
current_composite_frames = capture_frames(frame_count)

# Generate a unique filename using a timestamp and save the video
initial_video_filename = os.path.join(PLAY_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
save_output_video(initial_video_filename, current_composite_frames, FPS)
print(f"Initial video saved as {initial_video_filename}\n\n\n")

with open("_completed_video.txt", "w") as f:
    f.write(initial_video_filename)


# Begin the main loop
with mp_face_detection.FaceDetection(min_detection_confidence=CONFIDENCE_THRESHOLD) as face_detection:
    while True:
        loop_start_time = t.time()

        # Capture two frames for face detection (temporal filtering)
        frame_1 = picam2.capture_array()
        processed_frame_1 = process_image(frame_1)
        t.sleep(0.5)
        frame_2 = picam2.capture_array()
        processed_frame_2 = process_image(frame_2)

        # Detect faces
        detection_start_time = t.time()
        results_1 = face_detection.process(processed_frame_1)
        results_2 = face_detection.process(processed_frame_2)
        detection_end_time = t.time()

        if has_valid_detections(results_1, CONFIDENCE_THRESHOLD) and has_valid_detections(results_2, CONFIDENCE_THRESHOLD):
            print(f"Time taken for face detection: {detection_end_time - detection_start_time:.4f} seconds")

            # Capture new frames
            capture_start_time = t.time()
            new_frames = capture_frames(frame_count)
            capture_end_time = t.time()
            print(f"Time for frame capture: {capture_end_time - capture_start_time:.4f}")

            # Overlay bounding boxes on all frames if debugging is enabled
            if SHOW_BB_DEBUG:
                for i, frame in enumerate(new_frames):
                    if results_1.detections:
                        for detection in results_1.detections:
                            mp_drawing.draw_detection(frame, detection)
                    if results_2.detections:
                        for detection in results_2.detections:
                            mp_drawing.draw_detection(frame, detection)
                    new_frames[i] = frame

            # Perform alpha blending on all frames
            blend_start_time = t.time()
            blended_frames = alpha_blend_frames(new_frames, current_composite_frames, ALPHA)
            blend_end_time = t.time()
            print(f"Time for face blending: {blend_end_time - blend_start_time:.4f} seconds")

            # Update the current composite frames
            current_composite_frames = blended_frames

            # Generate a unique filename using a timestamp
            video_save_start_time = t.time()
            new_video_filename = os.path.join(PLAY_DIR, f"_play_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            # Save the new composite as the play video
            save_output_video(new_video_filename, current_composite_frames, FPS)
            video_save_end_time = t.time()
            print(f"Time for saving video: {video_save_end_time - video_save_start_time:.4f}")
            print(f"Updated video saved as {new_video_filename}")

            with open("_completed_video.txt", "w") as f:
                f.write(new_video_filename)

            # Clean up old videos, keeping only the most recent two
            video_files = sorted([os.path.join(PLAY_DIR, f) for f in os.listdir(PLAY_DIR)], key=os.path.getmtime)
            if len(video_files) > 2:
                for f in video_files[:-2]:
                    os.remove(f)

            loop_end_time = t.time()
            print(f"Loop iteration completed in {loop_end_time - loop_start_time:.4f} seconds")
            print("--------------------------------------------")

        else:
            t.sleep(0.2)
