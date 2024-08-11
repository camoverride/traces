from datetime import datetime
import os
import time

import cv2
import mediapipe as mp
import numpy as np
from picamera2 import Picamera2



WIDTH, HEIGHT = 1080, 1920
# WIDTH, HEIGHT = 1280, 720
PLAY_DIR = "play_files"
CONFIDENCE_THRESHOLD = 0.5
NEW_IMAGES_MEMMAP_PATH = "_current_frames.dat"
DURATION = 5
ALPHA=0.5
CAPTURE_DURATION=5 # seconds
FPS = 15

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize the picamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888",
                                                            "size": (WIDTH, HEIGHT)}))
picam2.start()

with mp_face_detection.FaceDetection(min_detection_confidence=CONFIDENCE_THRESHOLD) as face_detection:

    while True:
        # Snap a photo
        frame = picam2.capture_array()

        # Convert the image from BGR to RGB
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect faces
        results = face_detection.process(frame)

        # Check if any faces are detected
        if results.detections:
            print(f"Face detected! Saving frames to {NEW_IMAGES_MEMMAP_PATH}")

            # How many frames to record?
            frame_count = int(CAPTURE_DURATION * FPS)

            # Create a memory-mapped array to store the frames
            memmap_shape = (frame_count, HEIGHT, WIDTH, 3)
            new_images_memmap = np.memmap(NEW_IMAGES_MEMMAP_PATH, dtype='uint8', mode='w+', shape=memmap_shape)

            for frame_num in range(frame_count):
                time.sleep(0.04)
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

            # Get the time for filenaming
            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            # Overlay the images
            output_memmap_path = os.path.join(PLAY_DIR, f"{current_time}.dat")
            print(f"Combining frames from {NEW_IMAGES_MEMMAP_PATH} and {most_recent_memmap_composite_path} to create {output_memmap_path}")

            # Create output memmap
            output_memmap = np.memmap(output_memmap_path, dtype='uint8', mode='w+', shape=memmap_shape)

            output_memmap[:] = ALPHA * new_images_memmap[:] + (1 - ALPHA) * most_recent_composite_memmap[:]

            output_memmap.flush()

            del new_images_memmap, most_recent_composite_memmap, output_memmap

        else:
            print("No face detected!")
            time.sleep(1)

        # Clean up old files from play dir if there are too many
        if len(composites_paths) > 5:
            for f in composites_paths[5:]:
                os.remove(f)

        print("--------------------------------------------")
