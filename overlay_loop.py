import numpy as np
import cv2
import os
from datetime import datetime
import time as t
import yaml
import mediapipe as mp
from image_processing import process_image
from picamera2 import Picamera2



# Initialize Picamera2 once
picam2 = Picamera2()

def capture_frames(picam2, frame_count, width, height):
    """
    Capture a series of frames.
    """
    picam2.configure(picam2.create_still_configuration(main={"size": (width, height), "format": "RGB888"}))
    picam2.start()
    frames = []
    for _ in range(frame_count):
        frame = picam2.capture_array()
        frames.append(frame)
    picam2.stop()

    return np.array(frames)

def alpha_blend_frames(new_frames, current_composite_frames, alpha):
    """
    Alpha blend two images.
    """
    blended_frames = []
    for i in range(new_frames.shape[0]):
        blended_frame = cv2.addWeighted(new_frames[i], alpha, current_composite_frames[i], 1 - alpha, 0)
        blended_frames.append(blended_frame)

    return np.array(blended_frames)

def save_output_video(output_video_path, output_frames, fps, width, height):
    """
    Save the output video.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in output_frames:
        out.write(frame)
    out.release()

if __name__ == "__main__":
    # Read data from the config
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    DEBUG = config["debug"]
    WIDTH = config["WIDTH"]
    HEIGHT = config["HEIGHT"]
    PLAY_DIR = config["PLAY_DIR"]
    CONFIDENCE_THRESHOLD = config["CONFIDENCE_THRESHOLD"]
    CAPTURE_DURATION = 3  # Record for 3 seconds
    FPS = 15  # 15 frames per second
    ALPHA = config.get("ALPHA", 0.5)  # Alpha value for blending (default to 0.5 if not set in config)
    frame_count = int(CAPTURE_DURATION * FPS)

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initial capture to create the first video (at higher resolution)
    current_composite_frames = capture_frames(picam2, frame_count, WIDTH, HEIGHT)

    # Generate a unique filename using a timestamp and save the video
    initial_video_filename = os.path.join(PLAY_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    save_output_video(initial_video_filename, current_composite_frames, FPS, WIDTH, HEIGHT)
    print(f"Initial video saved as {initial_video_filename}\n\n\n")

    with open("_completed_video.txt", "w") as f:
        f.write(initial_video_filename)

    # Begin the main loop
    while True:
        detection_score_1, detection_score_2 = 0, 0
        loop_start_time = t.time()

        # Capture frame_1 and frame_2 at lower resolution (640 x 480)
        picam2.configure(picam2.create_still_configuration(main={"size": (640, 480), "format": "RGB888"}))
        picam2.start()
        frame_1 = picam2.capture_array()
        t.sleep(0.5)
        frame_2 = picam2.capture_array()
        picam2.stop()

        # Detect faces
        results_1 = mp_face_detection.process(frame_1)
        results_2 = mp_face_detection.process(frame_2)

        # Get the detection scores
        if results_1.detections:
            detection_score_1 = results_1.detections[0].score[0]
            if DEBUG:
                mp_drawing.draw_detection(frame_1, results_1.detections[0])

        if results_2.detections:
            detection_score_2 = results_2.detections[0].score[0]
            if DEBUG:
                mp_drawing.draw_detection(frame_2, results_2.detections[0])

        # Display the images if DEBUG is enabled.
        if DEBUG:
            cv2.imshow("main debug", frame_1)
            cv2.waitKey(int(1000 / FPS))
            cv2.imshow("main debug", frame_2)
            cv2.waitKey(int(1000 / FPS))

            print(f"Results 1: {detection_score_1}")
            print(f"Results 2: {detection_score_2}")
            print("*********************")

        # If there are faces, overlay the frames.
        if (detection_score_1 > CONFIDENCE_THRESHOLD) and (detection_score_2 > CONFIDENCE_THRESHOLD):
            print("Face detected!")

            # Capture new frames at the original resolution (WIDTH x HEIGHT)
            new_frames = capture_frames(picam2, frame_count, WIDTH, HEIGHT)

            # Perform alpha blending on all frames
            blended_frames = alpha_blend_frames(new_frames, current_composite_frames, ALPHA)

            # Update the current composite frames
            current_composite_frames = blended_frames

            # Generate a unique filename using a timestamp
            new_video_filename = os.path.join(PLAY_DIR, f"_play_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            
            # Save the new composite as the play video
            save_output_video(new_video_filename, current_composite_frames, FPS, WIDTH, HEIGHT)
            print(f"  Updated video saved as {new_video_filename}")

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
            cv2.waitKey(int(1000 / FPS))
