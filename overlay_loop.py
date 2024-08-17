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
CAPTURE_DURATION = config["CAPTURE_DURATION"]
FPS = config["FPS"]

# Initialize the picamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)}))
picam2.start()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the TFLite model for batch blending
interpreter = make_interpreter("alpha_blending_batch_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to print memory usage
def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[{label}] Memory Usage: RSS = {mem_info.rss / (1024 * 1024):.2f} MB")

# Function to process frames using the TPU with batch processing
def process_frames_batch(interpreter, input_details, output_details, new_frames_batch, current_frames_batch):
    print(f"Processing batch of {len(new_frames_batch)} frames...")

    # Normalize and expand dimensions for the model
    new_frames_batch_normalized = new_frames_batch.astype(np.float32) / 255.0
    current_frames_batch_normalized = current_frames_batch.astype(np.float32) / 255.0

    # Set the tensors for batch processing
    interpreter.set_tensor(input_details[0]['index'], new_frames_batch_normalized)
    interpreter.set_tensor(input_details[1]['index'], current_frames_batch_normalized)

    # Run inference
    interpreter.invoke()

    # Get the blended output
    blended_frames_batch = interpreter.get_tensor(output_details[0]['index'])
    blended_frames_batch_uint8 = (blended_frames_batch * 255).astype(np.uint8)

    print(f"Finished processing batch of {len(new_frames_batch)} frames.")
    print_memory_usage("After Processing Batch")

    return blended_frames_batch_uint8

# Function to capture, process, and blend frames in smaller batches
def capture_and_blend_batches(frame_count, batch_size, current_composite_frames):
    blended_frames = []

    for start in range(0, frame_count, batch_size):
        end = min(start + batch_size, frame_count)
        print(f"Capturing and blending batch {start + 1} to {end}...")

        # Capture the batch of frames
        new_frames_batch = []
        for i in range(start, end):
            frame = picam2.capture_array()
            new_frames_batch.append(frame)
            if i % 10 == 0:
                print(f"Captured frame {i + 1}/{frame_count}")
                print_memory_usage(f"After Capturing Frame {i + 1}")
        new_frames_batch = np.array(new_frames_batch)

        # Process and blend the batch
        blended_frames_batch = process_frames_batch(interpreter, input_details, output_details, new_frames_batch, current_composite_frames[start:end])

        # Append the blended frames to the output and update current_composite_frames
        blended_frames.extend(blended_frames_batch)
        current_composite_frames[start:end] = blended_frames_batch

        # Release the memory used by the captured batch
        del new_frames_batch
        del blended_frames_batch

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

# Initial capture to create the first composite video
frame_count = int(CAPTURE_DURATION * FPS)
print_memory_usage("Before Initial Capture")
print("Capturing initial 5-second video...")
current_composite_frames = capture_and_blend_batches(frame_count, frame_count, np.zeros((frame_count, HEIGHT, WIDTH, 3), dtype=np.uint8))
save_output_video(os.path.join(PLAY_DIR, "_play_video.mp4"), current_composite_frames, FPS)
print("Initial video saved as _play_video.mp4")
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

            # Capture and blend frames in batches
            current_composite_frames = capture_and_blend_batches(frame_count, 50, current_composite_frames)

            print_memory_usage("Before Saving Blended Video")
            # Save the new composite as the play video
            save_output_video(os.path.join(PLAY_DIR, "_play_video.mp4"), current_composite_frames, FPS)
            print("Updated video saved as _play_video.mp4")

        else:
            print(f"No face detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            t.sleep(1)

        loop_end_time = t.time()
        print(f"Loop iteration completed in {loop_end_time - loop_start_time:.4f} seconds")
        print("--------------------------------------------")
