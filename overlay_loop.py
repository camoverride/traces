import numpy as np
import cv2
import os
from datetime import datetime
import time as t
import yaml
from picamera2 import Picamera2
import mediapipe as mp
from pycoral.utils.edgetpu import make_interpreter

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

# Function to capture frames into memory
def capture_frames(frame_count):
    frames = []
    for _ in range(frame_count):
        frame = picam2.capture_array()
        frames.append(frame)
    return np.array(frames)

# Function to save output video
def save_output_video(output_video_path, output_frames, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (WIDTH, HEIGHT))
    for frame in output_frames:
        out.write(frame)
    out.release()

# Function to process frames using the TPU with batch processing
def process_frames_batch(interpreter, input_details, output_details, new_frames, current_frames):
    # Normalize and expand dimensions for the model
    new_frames_normalized = new_frames.astype(np.float32) / 255.0
    current_frames_normalized = current_frames.astype(np.float32) / 255.0

    # Set the tensors for batch processing
    interpreter.set_tensor(input_details[0]['index'], new_frames_normalized)
    interpreter.set_tensor(input_details[1]['index'], current_frames_normalized)

    # Run inference
    interpreter.invoke()

    # Get the blended output
    output_frames = interpreter.get_tensor(output_details[0]['index'])
    output_frames_uint8 = (output_frames * 255).astype(np.uint8)

    return output_frames_uint8

# Initial capture to create the first composite video
frame_count = int(CAPTURE_DURATION * FPS)
print("Capturing initial 5-second video...")
current_composite_frames = capture_frames(frame_count)
save_output_video(os.path.join(PLAY_DIR, "_play_video.mp4"), current_composite_frames, FPS)
print("Initial video saved as _play_video.mp4")

# Begin the main loop
with mp_face_detection.FaceDetection(min_detection_confidence=CONFIDENCE_THRESHOLD) as face_detection:
    while True:
        loop_start_time = t.time()

        # Capture two frames for face detection
        frame_1 = picam2.capture_array()
        t.sleep(0.5)
        frame_2 = picam2.capture_array()

        # Detect faces
        results_1 = face_detection.process(frame_1)
        results_2 = face_detection.process(frame_2)

        if 1==1:#results_1.detections and results_2.detections:
            print("Face detected! Capturing and blending new frames...")

            # Capture new frames as a batch
            new_frames = capture_frames(frame_count)

            # Blend the batch of new frames with the current composite frames using the TPU
            blended_frames = process_frames_batch(interpreter, input_details, output_details, new_frames, current_composite_frames)

            # Update the current composite frames
            current_composite_frames = blended_frames

            # Save the new composite as the play video
            save_output_video(os.path.join(PLAY_DIR, "_play_video.mp4"), current_composite_frames, FPS)
            print("Updated video saved as _play_video.mp4")

        else:
            print(f"No face detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            t.sleep(1)

        loop_end_time = t.time()
        print(f"Loop iteration completed in {loop_end_time - loop_start_time:.4f} seconds")
        print("--------------------------------------------")
