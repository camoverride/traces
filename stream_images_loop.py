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


def get_second_latest_video_path(play_dir):
    """
    Returns the path of the second most recent video in the directory.
    """
    file_paths = list(reversed(sorted([os.path.join(play_dir, f) for f in os.listdir(play_dir)])))
    return file_paths[1] if len(file_paths) > 1 else None


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
            new_video_path = get_second_latest_video_path(PLAY_DIR)
            if new_video_path != video_path:
                print(f"New video detected: {new_video_path}")
                cap.release()
                return new_video_path

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
        latest_video_path = get_second_latest_video_path(PLAY_DIR)

        # Check if a new video has been created
        if latest_video_path != last_video_path:
            last_video_path = latest_video_path
            print(f"Loading new video: {last_video_path}")
        
        # Play the current video, checking for a new one
        last_video_path = play_video(last_video_path)



if __name__ == "__main__":
    main()
