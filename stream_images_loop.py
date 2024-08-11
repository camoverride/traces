import os
import subprocess
import cv2
import numpy as np
from frames import stream_memmap_frames
from overlay_loop import HEIGHT, WIDTH, CAPTURE_DURATION, FPS, PLAY_DIR



# Configure the screen properly
commands = ["WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90",
            "export DISPLAY=:0"]
for command in commands:
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Set to fullscreen.
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while True:
    # Get the paths to all the files in the play_dir, sorted from newest to oldest.
    file_paths = list(reversed(sorted([os.path.join(PLAY_DIR, f) for f in os.listdir(PLAY_DIR)])))

    # Instead of playing the newest file, play the second newest.
    # This is because if the newest file is being written to, there might be an error.
    play_file_path = file_paths[1]

    print(f"Streaming {play_file_path}")
    stream_memmap_frames(memmap_filename=play_file_path)

    # The shape is set to match (frame_count, height, width, channels)
    frame_count = CAPTURE_DURATION * FPS
    height = HEIGHT
    width = WIDTH
    channels = 3

    # Create the memmap with the correct shape
    memmap = np.memmap(play_file_path, dtype='uint8', mode='r', shape=(frame_count, height, width, channels))

    for frame_num in range(frame_count):
        frame = memmap[frame_num]
        cv2.imshow("window", frame)

        # Wait for user input
        key = cv2.waitKey(40)  # Adjust this to match the desired FPS

        # Exit if 'q' is pressed
        if key == ord("q"):
            break
