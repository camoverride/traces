import os
import time
import cv2
from frames import stream_memmap_frames, copy_file



last_memmap_modification_time = None


def stream_new_images(memmap_file_path):
    global last_memmap_modification_time

    # Get the current capture time of the images.
    current_memmap_modification_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                time.localtime(os.path.getmtime(memmap_file_path)))

    # If there is new data, first copy it to the play folder
    # so it can't be overwritten by the image writing function
    if last_memmap_modification_time != current_memmap_modification_time:
        print("New images detected, copying files...")

        copy_file(src=memmap_file_path, dst="__play.dat")

        last_memmap_modification_time = current_memmap_modification_time

    # If the data is unchanged, still stream.
    else:
        print("No new images detected...")

    # Stream images from ths folder!
    print("Streaming images")
    stream_memmap_frames(memmap_filename="__play.dat")



if __name__ == "__main__":
    # Set to fullscreen.
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
       stream_new_images(memmap_file_path="composites.dat")
