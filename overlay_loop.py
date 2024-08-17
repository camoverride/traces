from datetime import datetime
import os
import time as t  # Avoid conflict with 'time' function
import yaml

import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

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
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888",
                                                            "size": (WIDTH, HEIGHT)}))
picam2.start()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the TFLite model for overlaying
interpreter = make_interpreter("alpha_blending_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# When the program starts, save two new videos to the play folder.
frame_count = int(CAPTURE_DURATION * FPS)

for _ in range(2):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{current_time}.dat"
    start_memmap_path = os.path.join(PLAY_DIR, filename)
    print(f"Creating start file {start_memmap_path}")
    memmap_shape = (frame_count, HEIGHT, WIDTH, 3)
    start_memmap = np.memmap(start_memmap_path, dtype='uint8', mode='w+', shape=memmap_shape)

    for frame_num in range(frame_count):
        frame = picam2.capture_array()
        start_memmap[frame_num] = frame

    # Finalize the memmap file
    start_memmap.flush()

# Begin the main loop
with mp_face_detection.FaceDetection(min_detection_confidence=CONFIDENCE_THRESHOLD) as face_detection:
    while True:
        start_time = t.time()  # Start timing the loop
        
        # Snap two photos for temporal filtering to reduce the likelihood of false positives
        capture_start_time = t.time()
        frame_1 = picam2.capture_array()
        t.sleep(0.5)
        frame_2 = picam2.capture_array()
        capture_end_time = t.time()
        print(f"Time taken to capture frames: {capture_end_time - capture_start_time:.4f} seconds")
        
        # Process the frames and detect faces
        detection_start_time = t.time()
        results_1 = face_detection.process(frame_1)
        results_2 = face_detection.process(frame_2)
        detection_end_time = t.time()
        print(f"Time taken for face detection: {detection_end_time - detection_start_time:.4f} seconds")
        
        # Get the time for filenaming
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        if 1==1:#results_1.detections and results_2.detections:
            t_now = datetime.now().strftime("%H-%M-%S")
            print(f"{t_now} - Face detected! Saving frames to {NEW_IMAGES_MEMMAP_PATH}")

            frame_count = int(CAPTURE_DURATION * FPS)
            memmap_shape = (frame_count, HEIGHT, WIDTH, 3)
            new_images_memmap = np.memmap(NEW_IMAGES_MEMMAP_PATH, dtype='uint8', mode='w+', shape=memmap_shape)

            for frame_num in range(frame_count):
                frame = picam2.capture_array()
                new_images_memmap[frame_num] = frame

            new_images_memmap.flush()

            composites_paths = list(reversed(sorted([os.path.join(PLAY_DIR, f) for f in os.listdir(PLAY_DIR)])))
            most_recent_memmap_composite_path = composites_paths[0]
            most_recent_composite_memmap = np.memmap(most_recent_memmap_composite_path, dtype='uint8', mode='r', shape=memmap_shape)

            output_memmap_path = os.path.join(PLAY_DIR, f"{current_time}.dat")
            t_now = datetime.now().strftime("%H-%M-%S")
            print(f"{t_now} - Combining frames from {NEW_IMAGES_MEMMAP_PATH} and {most_recent_memmap_composite_path} to create {output_memmap_path}")

            output_memmap = np.memmap(output_memmap_path, dtype='uint8', mode='w+', shape=memmap_shape)

            blending_start_time = t.time()
            for frame_num in range(frame_count):
                # Prepare the input tensors
                input_1 = np.expand_dims(new_images_memmap[frame_num].astype(np.float32) / 255.0, axis=0)
                input_2 = np.expand_dims(most_recent_composite_memmap[frame_num].astype(np.float32) / 255.0, axis=0)

                interpreter.set_tensor(input_details[0]['index'], input_1)
                interpreter.set_tensor(input_details[1]['index'], input_2)

                # Run inference
                interpreter.invoke()

                # Get the result
                output_frame = interpreter.get_tensor(output_details[0]['index'])

                # Scale the output back to 0-255 range and convert to uint8
                output_memmap[frame_num] = (output_frame[0] * 255).astype(np.uint8)
            blending_end_time = t.time()
            print(f"Time taken for blending and saving frames: {blending_end_time - blending_start_time:.4f} seconds")

            output_memmap.flush()

            del new_images_memmap, most_recent_composite_memmap, output_memmap

            # Clean up old files from play dir if there are too many
            cleanup_start_time = t.time()
            if len(composites_paths) > 5:
                for f in composites_paths[5:]:
                    os.remove(f)
            cleanup_end_time = t.time()
            print(f"Time taken for cleanup: {cleanup_end_time - cleanup_start_time:.4f} seconds")
        else:
            print(f"No face detected: {current_time}")
            t.sleep(1)

        loop_end_time = t.time()
        print(f"Total time for loop iteration: {loop_end_time - start_time:.4f} seconds")
        print("--------------------------------------------")
