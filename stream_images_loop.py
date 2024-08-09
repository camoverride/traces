import os
import cv2
from frames import stream_memmap_frames



if __name__ == "__main__":
    # Set to fullscreen.
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    play_dir = "play_files"
    while True:
        # Get the paths to all the files in the play_dir, sorted from newest to oldest.
        file_paths = list(reversed(sorted([os.path.join(play_dir, f) for f in os.listdir(play_dir)])))

        # Instead of playing the newest file, play the second newest.
        # This is because if the newest file is being written to, there might be an error.
        play_file = file_paths[1]

        print(f"Streaming {play_file}")
        stream_memmap_frames(memmap_filename=play_file)
