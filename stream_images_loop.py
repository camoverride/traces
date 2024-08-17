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
CAPTURE_DURATION = config["CAPTURE_DURATION"]
FPS = config["FPS"]

# Set the DISPLAY environment variable for the current process
os.environ["DISPLAY"] = ":0"

# Set to fullscreen.
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Get the frame count
frame_count = CAPTURE_DURATION * FPS

# Configure the screen properly (run only once, ideally after the screen turns on)
subprocess.run("WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90",
               shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

while True:
    # Get the paths to all the files in the play_dir, sorted from newest to oldest.
    file_paths = list(reversed(sorted([os.path.join(PLAY_DIR, f) for f in os.listdir(PLAY_DIR)])))

    # Play the most recent video file (or second most recent if necessary)
    play_file_path = file_paths[0]

    print(f"Streaming {play_file_path}")

    # Open the video file
    cap = cv2.VideoCapture(play_file_path)

    while True:
        ret, frame = cap.read()

        if not ret:
            # Restart the video when it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Display the frame in fullscreen
        cv2.imshow("window", frame)

        # Wait for user input (loop at the desired FPS)
        key = cv2.waitKey(int(1000 / FPS))  # Adjust this to match the desired FPS

        # Exit if 'q' is pressed
        if key == ord("q"):
            break

    # Release the video capture object
    cap.release()

    # Exit if 'q' is pressed (for the outer loop)
    if key == ord("q"):
        break

# Destroy all OpenCV windows
cv2.destroyAllWindows()
