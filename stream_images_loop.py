import os
import subprocess
import yaml
import cv2
import numpy as np



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


# Configure the screen properly
commands = ["WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90",
            "export DISPLAY=:0"]
for command in commands:
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Set to fullscreen.
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Get the frame count
frame_count = CAPTURE_DURATION * FPS


while True:
    # Get the paths to all the files in the play_dir, sorted from newest to oldest.
    file_paths = list(reversed(sorted([os.path.join(PLAY_DIR, f) for f in os.listdir(PLAY_DIR)])))

    # Instead of playing the newest file, play the second newest.
    # This is because if the newest file is being written to, there might be an error.
    play_file_path = file_paths[1]

    print(f"Streaming {play_file_path}")

    # Create the memmap with the correct shape
    memmap = np.memmap(play_file_path, dtype='uint8', mode='r', shape=(frame_count, HEIGHT, WIDTH, 3))

    for frame_num in range(frame_count):
        frame = memmap[frame_num]
        cv2.imshow("window", frame)

        # Wait for user input
        key = cv2.waitKey(40)  # Adjust this to match the desired FPS

        # Exit if 'q' is pressed
        if key == ord("q"):
            break
