import os
import shutil
import time
import cv2
from frames import stream_images



def copy_folder(src, dst):
    """
    Copies a folder from a `src` to a `dst`.
    """
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
    last_frame_capture_time = None

    # Set to fullscreen.
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    while True:
        # Get the current capture time of the images.
        current_frame_capture_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                   time.localtime(os.path.getmtime("frames/frame_0000.png")))

        # If there is new data, first copy it to the play folder so it can't be overwritten by the image wrting function
        if current_frame_capture_time != last_frame_capture_time:
            print("New images detected...")

            copy_folder(src="composites", dst="play_dir")

            last_frame_capture_time = current_frame_capture_time

        # If the data is unchanged, still stream.
        else:
            print("No new images detected...")

        # Stream images from ths folder!
        stream_images(data_dir="play_dir")
