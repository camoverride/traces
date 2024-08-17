import numpy as np
import cv2
import os
from datetime import datetime
import time as t
import yaml
from picamera2 import Picamera2
import mediapipe as mp
from pycoral.utils.edgetpu import make_interpreter
import psutil  # For memory usage

# Read data from the config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
WIDTH = config["WIDTH"]
HEIGHT = config["HEIGHT"]
PLAY_DIR = config["PLAY_DIR"]
CONFIDENCE_THRESHOLD = config["CONFIDENCE_THRESHOLD"]
CAPTURE_DURATION = 3  # Record for 3 seconds
FPS = 15  # 15 frames per second

# Initialize the picamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)}))
picam2.start()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the TFLite model for blending
interpreter = make_interpreter("alpha_blending_fixed_batch_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to print memory usage
def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{label}] Memory Usage: RSS = {mem_info.rss / (1024 * 1024):.2f} MB")

# Function to capture a series of frames into memory
def capture_frames(frame_count):
    frames = []
    for i in range(frame_count):
        frame = picam2.capture_array()
        frames.append(frame)
        if i % 10 == 0:
            print(f"Captured frame {i + 1}/{frame_count}")
            print_memory_usage(f"After Capturing Frame {i + 1}")
    return np.array(frames)

# Function to process and blend frames one by one using the TPU
def process_frames_individually(interpreter, input_details, output_details, new_frames, current_frames):
    blended_frames = []
    for i in range(new_frames.shape[0]):
        # Normalize and expand dimensions for the model
        new_frame_normalized = np.expand_dims(np.expand_dims(new_frames[i].astype(np.float32) / 255.0, axis=0), axis=0)
        current_frame_normalized = np.expand_dims(np.expand_dims(current_frames[i].astype(np.float32) / 255.0, axis=0), axis=0)

        # Set the tensors for processing
        interpreter.set_tensor(input_details[0]['index'], new_frame_normalized)
        interpreter.set_tensor(input_details[1]['index'], current_frame_normalized)

        # Run inference
        interpreter.invoke()

        # Get the blended output
        blended_frame = interpreter.get_tensor(output_details[0]['index'])[0][0]
        blended_frames.append((blended_frame * 255).astype(np.uint8))

    return np.array(blended_frames)

# Function to save output video
def save_output_video(output_video_path, output_frames, fps):
    print(f"Saving video to {output_video_path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (WIDTH, HEIGHT))
    for i, frame in enumerate(output_frames):
        out.write(frame)
        if i % 10 == 0:
            print(f"Saved frame {i + 1}/{len(output_frames)} to video.")
            print_memory_usage(f"After Saving Frame {i + 1}")
    out.release()
    print(f"Video saved successfully to {output_video_path}.")
    print_memory_usage("After Saving Video")

# Function to clean up old videos, keeping only the most recent couple
def clean_up_old_videos(play_dir, num_to_keep=2):
    video_files = sorted([os.path.join(play_dir, f) for f in os.listdir(play_dir) if f.endswith(".mp4")], reverse=True)
    if len(video_files) > num_to_keep:
        for old_video in video_files[num_to_keep:]:
            os.remove(old_video)
            print(f"Deleted old video: {old_video}")

# Initial capture to create the first composite video
frame_count = int(CAPTURE_DURATION * FPS)
print_memory_usage("Before Initial Capture")
print("Capturing initial 3-second video...")
current_composite_frames = capture_frames(frame_count)

# Generate a unique filename using a timestamp
initial_video_filename = os.path.join(PLAY_DIR, f"_play_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
save_output_video(initial_video_filename, current_composite_frames, FPS)
print(f"Initial video saved as {initial_video_filename}")
print_memory_usage("After Initial Capture and Save")

# Begin the main loop
with mp_face_detection.FaceDetection(min_detection_confidence=CONFIDENCE_THRESHOLD) as face_detection:
    while True:
        loop_start_time = t.time()
        print_memory_usage("Start of Loop")

        # Capture two frames for face detection
        frame_1 = picam2.capture_array()
        t.sleep(0.5)
        frame_2 = picam2.capture_array()

        # Detect faces
        detection_start_time = t.time()
        results_1 = face_detection.process(frame_1)
        results_2 = face_detection.process(frame_2)
        detection_end_time = t.time()
        print(f"Time taken for face detection: {detection_end_time - detection_start_time:.4f} seconds")

        if results_1.detections and results_2.detections:
            print("Face detected! Capturing and blending new frames...")
            print_memory_usage("Before Capturing New Frames")

            # Capture new frames
            new_frames = capture_frames(frame_count)

            print_memory_usage("Before Blending New Frames")
            # Blend all the frames one by one
            blended_frames = process_frames_individually(interpreter, input_details, output_details, new_frames, current_composite_frames)

            # Update the current composite frames
            current_composite_frames = blended_frames

            print_memory_usage("Before Saving Blended Video")
            # Generate a unique filename using a timestamp
            new_video_filename = os.path.join(PLAY_DIR, f"_play_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            # Save the new composite as the play video
            save_output_video(new_video_filename, current_composite_frames, FPS)
            print(f"Updated video saved as {new_video_filename}")

            # Clean up old videos, keeping only the most recent couple
            clean_up_old_videos(PLAY_DIR, num_to_keep=2)

        else:
            print(f"No face detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            t.sleep(1)

        loop_end_time = t.time()
        print(f"Loop iteration completed in {loop_end_time - loop_start_time:.4f} seconds")
        print("--------------------------------------------")

