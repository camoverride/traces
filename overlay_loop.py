from datetime import datetime
import os
import time
import yaml

import cv2
import mediapipe as mp
import numpy as np
from picamera2 import Picamera2



# Read data from the config.
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
WIDTH = config["WIDTH"]
HEIGHT = config["HEIGHT"]
PLAY_DIR = config["PLAY_DIR"]
CONFIDENCE_THRESHOLD = config["CONFIDENCE_THRESHOLD"]
NEW_IMAGES_MEMMAP_PATH = config["NEW_IMAGES_MEMMAP_PATH"]
ALPHA = config["ALPHA"]
CAPTURE_DURATION = config["CAPTURE_DURATION"]
FPS = config["FPS"]


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize the picamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888",
                                                            "size": (WIDTH, HEIGHT)}))
picam2.start()

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
        # Snap two photos for temporal filtering to reduce the likelihood of false positives
        frame_1 = picam2.capture_array()
        time.sleep(0.5)
        frame_2 = picam2.capture_array()

        # Process the frame and detect faces
        results_1 = face_detection.process(frame_1)
        results_2 = face_detection.process(frame_2)

        # Get the time for filenaming
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # Check if any faces are detected
        if results_1.detections and results_2.detections:
            print(f"Face detected! Saving frames to {NEW_IMAGES_MEMMAP_PATH}")

            # Create a copy of the frame for debugging. TODO: eventually get rid of this.
            debug_frame_1 = frame_1.copy()
            debug_frame_2 = frame_2.copy()

            # Draw bounding boxes on the debug frame
            for detection in results_1.detections:
                mp_drawing.draw_detection(debug_frame_1, detection)
            for detection in results_2.detections:
                mp_drawing.draw_detection(debug_frame_2, detection)

            # Save the debug frames with bounding boxes
            cv2.imwrite(f"debug_frames/_debug_frame_1_{current_time}.jpg", debug_frame_1)
            cv2.imwrite(f"debug_frames/_debug_frame_2_{current_time}.jpg", debug_frame_2)

            # How many frames to record?
            frame_count = int(CAPTURE_DURATION * FPS)

            # Create a memory-mapped array to store the frames
            memmap_shape = (frame_count, HEIGHT, WIDTH, 3)
            new_images_memmap = np.memmap(NEW_IMAGES_MEMMAP_PATH, dtype='uint8', mode='w+', shape=memmap_shape)

            for frame_num in range(frame_count):
                frame = picam2.capture_array()
                # Store the frame in the correct index
                new_images_memmap[frame_num] = frame

            # Finalize the memmap file
            new_images_memmap.flush()

            # Get the most recent composite to add to the incoming frames to create the next composite.
            composites_paths = list(reversed(sorted([os.path.join(PLAY_DIR, f)
                                                    for f in os.listdir(PLAY_DIR)])))
            most_recent_memmap_composite_path = composites_paths[0]
            most_recent_composite_memmap = np.memmap(most_recent_memmap_composite_path,
                                                     dtype='uint8', mode='r', shape=memmap_shape)

            # Overlay the images frame by frame
            output_memmap_path = os.path.join(PLAY_DIR, f"{current_time}.dat")
            print(f"Combining frames from {NEW_IMAGES_MEMMAP_PATH} and {most_recent_memmap_composite_path} to create {output_memmap_path}")

            output_memmap = np.memmap(output_memmap_path, dtype='uint8', mode='w+', shape=memmap_shape)

            for frame_num in range(frame_count):
                output_memmap[frame_num] = (
                    ALPHA * new_images_memmap[frame_num] +
                    (1 - ALPHA) * most_recent_composite_memmap[frame_num])

            output_memmap.flush()

            del new_images_memmap, most_recent_composite_memmap, output_memmap

            # Clean up old files from play dir if there are too many
            if len(composites_paths) > 5:
                for f in composites_paths[5:]:
                    os.remove(f)

        else:
            print(f"No face detected: {current_time}")
            time.sleep(1)

        print("--------------------------------------------")
