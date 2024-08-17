import os
import subprocess
import time
import yaml
import cv2
import numpy as np

# Read data from the config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
WIDTH = config["WIDTH"]
HEIGHT = config["HEIGHT"]
PLAY_DIR = config["PLAY_DIR"]
CAPTURE_DURATION = config["CAPTURE_DURATION"]
FPS = config["FPS"]

# Set the DISPLAY environment variable for the current process
os.environ["DISPLAY"] = ":0"

# Set to fullscreen
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def get_latest_video_path(play_dir):
    """Returns the path of the most recent video in the directory."""
    file_paths = list(reversed(sorted([os.path.join(play_dir, f) for f in os.listdir(play_dir)])))
    return file_paths[0] if file_paths else None

def play_video(video_path):
    """Play the video in a loop."""
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
            continue
        cv2.imshow("window", frame)
        key = cv2.waitKey(int(1000 / FPS))  # Wait for key press or continue playback
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)  # Exit the entire program

def main():
    last_video_path = None

    # Configure the screen properly (run only once)
    subprocess.run("WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90",
                   shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    while True:
        latest_video_path = get_latest_video_path(PLAY_DIR)

        # Check if a new video has been created
        if latest_video_path != last_video_path:
            last_video_path = latest_video_path
            print(f"Loading new video: {last_video_path}")
            play_video(last_video_path)  # This will loop until a new video is detected

        time.sleep(1)  # Sleep briefly before checking again

if __name__ == "__main__":
    main()
