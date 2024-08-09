import time
import os
import shutil
from frames import stream_images
import cv2



def copy_folder(src, dst):
    try:
        # Check if the source folder exists
        if not os.path.exists(src):
            print(f"Source folder '{src}' does not exist.")
            return

        # If the destination folder exists, remove it
        if os.path.exists(dst):
            shutil.rmtree(dst)
        
        # Copy the source folder to the destination
        shutil.copytree(src, dst)
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    capture_datetime = None

    # Set to fullscreen TODO: is this inherited?
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



    while True:
        # Get the date of the previous recordings
        with open("recording_time.txt", "r") as f:
            current_capture_datetime = str(f.read())

        # If there is new data, first copy it to the play folder so it can't be overwritten by the image wrting function
        if current_capture_datetime != capture_datetime:
            print(f"New imagesd detected. Copying them to the play_dir and streaming them...")
            copy_folder(src="overlay_dir", dst="play_dir")
            capture_datetime = current_capture_datetime

        # Stream images from ths folder!
        stream_images(data_dir="play_dir")


        # # If the data is unchanged, wait 1 second and check again.
        # else:
        #     time.sleep(1)
