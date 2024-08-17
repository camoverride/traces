import threading
import numpy as np
import cv2
import os
from datetime import datetime
import time as t
import yaml
from picamera2 import Picamera2
import mediapipe as mp
from pycoral.utils.edgetpu import make_interpreter

# Read data from the config.
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
WIDTH = config["WIDTH"]
HEIGHT = config["HEIGHT"]
PLAY_DIR = config["PLAY_DIR"]
CONFIDENCE_THRESHOLD = config["CONFIDENCE_THRESHOLD"]
NEW_IMAGES_MEMMAP_PATH = config["NEW_IMAGES_MEMMAP_PATH"]
CAPTURE_DURATION = config["CAPTURE_DURATION"]
FPS = config["FPS"]

# Initialize the picamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)}))
picam2.start()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the TFLite model for overlaying
interpreter = make_interpreter("alpha_blending_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preload frames into memory
def preload_frames(memmap_path, shape):
    print(f"Preloading frames from {memmap_path}...")
    start_time = t.time()
    memmap_data = np.memmap(memmap_path, dtype='uint8', mode='r', shape=shape)
    frames = np.array(memmap_data)
    del memmap_data  # Clean up memmap to release the file handle
    end_time = t.time()
    print(f"Preloading completed in {end_time - start_time:.4f} seconds")
    return frames

# Function to save output video asynchronously
def save_output_video(output_video_path, output_frames, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (WIDTH, HEIGHT))
    for frame in output_frames:
        out.write(frame)
    out.release()

# Function to process frames using the TPU
def process_frames(frame_count, HEIGHT, WIDTH, interpreter, input_details, output_details, new_frames, most_recent_frames):
    output_frames = np.empty_like(new_frames)
    for frame_num in range(frame_count):
        input_1 = np.expand_dims(new_frames[frame_num].astype(np.float32) / 255.0, axis=0)
        input_2 = np.expand_dims(most_recent_frames[frame_num].astype(np.float32) / 255.0, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_1)
        interpreter.set_tensor(input_details[1]['index'], input_2)

        interpreter.invoke()

        output_frame = interpreter.get_tensor(output_details[0]['index'])
        output_frames[frame_num] = (output_frame[0] * 255).astype(np.uint8)

    return output_frames

# Save initial videos to play folder
frame_count = int(CAPTURE_DURATION * FPS)
for _ in range(2):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{current_time}.dat"
    start_memmap_path = os.path.join(PLAY_DIR, filename)
    memmap_shape = (frame_count, HEIGHT, WIDTH, 3)
    start_memmap = np.memmap(start_memmap_path, dtype='uint8', mode='w+', shape=memmap_shape)

    for frame_num in range(frame_count):
        frame = picam2.capture_array()
        start_memmap[frame_num] = frame

    start_memmap.flush()
    del start_memmap

# Initialize threading for saving frames
save_thread = None

# Begin the main loop
with mp_face_detection.FaceDetection(min_detection_confidence=CONFIDENCE_THRESHOLD) as face_detection:
    while True:
        loop_start_time = t.time()

        # Ensure the previous save thread has finished
        if save_thread is not None:
            save_thread.join()

        # Capture frames for temporal filtering
        capture_start_time = t.time()
        frame_1 = picam2.capture_array()
        t.sleep(0.5)
        frame_2 = picam2.capture_array()
        capture_end_time = t.time()
        print(f"Time taken to capture frames: {capture_end_time - capture_start_time:.4f} seconds")

        # Detect faces
        detection_start_time = t.time()
        results_1 = face_detection.process(frame_1)
        results_2 = face_detection.process(frame_2)
        detection_end_time = t.time()
        print(f"Time taken for face detection: {detection_end_time - detection_start_time:.4f} seconds")

        # Get current time for filenames
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        if 1==1:#results_1.detections and results_2.detections:
            face_detection_time = t.time()
            print(f"Face detected! Processing frames...")

            # Capture a series of frames into memory
            capture_series_start_time = t.time()
            new_frames = []
            for frame_num in range(frame_count):
                frame = picam2.capture_array()
                new_frames.append(frame)
            new_frames = np.array(new_frames)
            capture_series_end_time = t.time()
            print(f"Time taken to capture series of frames: {capture_series_end_time - capture_series_start_time:.4f} seconds")

            # Load the most recent composite from memory
            load_start_time = t.time()
            composites_paths = sorted([os.path.join(PLAY_DIR, f) for f in os.listdir(PLAY_DIR)], reverse=True)
            most_recent_memmap_composite_path = composites_paths[0]
            most_recent_composite_memmap_shape = (frame_count, HEIGHT, WIDTH, 3)
            most_recent_composite_frames = preload_frames(most_recent_memmap_composite_path, most_recent_composite_memmap_shape)
            load_end_time = t.time()
            print(f"Time taken to load most recent composite: {load_end_time - load_start_time:.4f} seconds")

            # Process frames using TPU (in the main thread)
            blending_start_time = t.time()
            output_frames = process_frames(frame_count, HEIGHT, WIDTH, interpreter, input_details, output_details, new_frames, most_recent_composite_frames)
            blending_end_time = t.time()
            print(f"Time taken for blending frames: {blending_end_time - blending_start_time:.4f} seconds")

            # Start a new thread to save the output video
            output_video_path = os.path.join(PLAY_DIR, f"{current_time}.avi")
            save_thread = threading.Thread(target=save_output_video, args=(output_video_path, output_frames, FPS))
            save_thread.start()

            # Clean up old files if necessary
            cleanup_start_time = t.time()
            if len(composites_paths) > 5:
                for f in composites_paths[5:]:
                    os.remove(f)
            cleanup_end_time = t.time()
            print(f"Time taken for cleanup: {cleanup_end_time - cleanup_start_time:.4f} seconds")

        else:
            no_face_detected_time = t.time()
            print(f"No face detected: {current_time}")
            t.sleep(1)

        loop_end_time = t.time()
        total_loop_time = loop_end_time - loop_start_time
        print(f"Total time for loop iteration: {total_loop_time:.4f} seconds")
        print(f"Detailed breakdown:")
        print(f"- Capture time: {capture_end_time - capture_start_time:.4f} seconds")
        print(f"- Face detection time: {detection_end_time - detection_start_time:.4f} seconds")
        if results_1.detections and results_2.detections:
            print(f"- Time to capture series of frames: {capture_series_end_time - capture_series_start_time:.4f} seconds")
            print(f"- Time to load most recent composite: {load_end_time - load_start_time:.4f} seconds")
            print(f"- Time for blending: {blending_end_time - blending_start_time:.4f} seconds")
            print(f"- Time for cleanup: {cleanup_end_time - cleanup_start_time:.4f} seconds")
        print("--------------------------------------------")

# Ensure all threads finish before exiting
if save_thread is not None:
    save_thread.join()
