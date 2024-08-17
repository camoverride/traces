import os
import subprocess
import yaml
import cv2

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

# Function to hide the mouse cursor
def hide_mouse(event, x, y, flags, param):
    cv2.setMouseCallback("window", lambda *args : None)  # No-op callback to hide cursor

# Apply the function to hide the cursor
cv2.setMouseCallback("window", hide_mouse)

def get_latest_videos(play_dir):
    """
    Returns the paths of the most recent and second most recent videos in the directory.
    """
    file_paths = list(reversed(sorted([os.path.join(play_dir, f) for f in os.listdir(play_dir) if f.endswith(('.mp4', '.avi', '.mov'))])))
    return (file_paths[0], file_paths[1]) if len(file_paths) > 1 else (file_paths[0], None) if file_paths else (None, None)

def is_file_complete(video_path):
    """
    Checks if a file is still being written to.
    Returns True if the file is complete, False if it's still being written to.
    """
    try:
        with open(video_path, 'ab') as f:
            pass
        return True
    except OSError:
        return False

def play_video(video_path):
    """
    Play the video in a loop until a new video is available.
    """
    cap = cv2.VideoCapture(video_path)
    frame_counter = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
            continue

        cv2.imshow("window", frame)
        key = cv2.waitKey(int(1000 / FPS))  # Adjust this to match the desired FPS

        # Check if a new video is available every few frames
        if frame_counter % FPS == 0:  # Check every second
            latest_video_path, second_latest_video_path = get_latest_videos(PLAY_DIR)
            if latest_video_path != video_path:
                print(f"New video detected: {latest_video_path}")
                cap.release()
                return latest_video_path

        frame_counter += 1

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
        latest_video_path, second_latest_video_path = get_latest_videos(PLAY_DIR)

        # Check if the most recent video is still being written to
        if latest_video_path and is_file_complete(latest_video_path):
            if latest_video_path != last_video_path:
                last_video_path = latest_video_path
                print(f"Loading new video: {last_video_path}")
        else:
            if second_latest_video_path and second_latest_video_path != last_video_path:
                last_video_path = second_latest_video_path
                print(f"Playing second most recent video: {last_video_path}")
        
        # Play the current video, checking for a new one
        last_video_path = play_video(last_video_path)

if __name__ == "__main__":
    main()
