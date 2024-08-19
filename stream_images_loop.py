import os
import cv2
import time



# Configuration
FPS = 10  # Adjust this to match the desired FPS
VIDEO_INFO_FILE = "./_completed_video.txt"


def get_latest_video_path():
    """
    Reads the video filename from the _completed_video.txt file.
    Returns the full path of the video if the file exists, otherwise None.
    """
    if os.path.exists(VIDEO_INFO_FILE):
        with open(VIDEO_INFO_FILE, "r") as file:
            video_filename = file.read().strip()
            if video_filename and os.path.exists(video_filename):
                return video_filename
    return None


def play_video(video_path):
    """
    Play the video in a loop until a new video is available.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
            continue

        cv2.imshow("window", frame)
        key = cv2.waitKey(int(1000 / FPS))  # Adjust this to match the desired FPS

        # Check for a new video after each frame
        new_video_path = get_latest_video_path()
        if new_video_path and new_video_path != video_path:
            print(f"New video detected: {new_video_path}")
            cap.release()
            return new_video_path

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)  # Exit the entire program


def main():
    last_video_path = None

    # Set to fullscreen
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        latest_video_path = get_latest_video_path()

        if latest_video_path and latest_video_path != last_video_path:
            last_video_path = latest_video_path
            print(f"Loading new video: {last_video_path}")
        
        if last_video_path:
            last_video_path = play_video(last_video_path)
        else:
            print("Waiting for a valid video to be listed...")
            time.sleep(1)  # Wait before checking again



if __name__ == "__main__":
    # Set the display orientation and configure the screen
    print("changing the screen orientation")
    os.environ["DISPLAY"] = ":0"
    os.system("WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90")
    
    time.sleep(2)

    print("starting the main event loop!!")
    main()
