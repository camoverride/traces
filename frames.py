import shutil
import subprocess
import time
import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2


WIDTH, HEIGHT = 1080, 1920
# WIDTH, HEIGHT = 1280, 720


def alpha_blend_images(image1, image2, alpha):
    """
    Blend two images with a given alpha value.
    """
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)


def overlay_frames_from_memmaps(memmap_filenames, output_memmap_filename, alpha):
    """
    Overlay frames from multiple memmaps and save the composite frames
    into an output memmap file.

    Parameters:
    - memmap_filenames: List of memmap filenames containing frames for each chunk.
    - output_memmap_filename: Filename where the composite frames will be saved.
    - alpha: Blending factor for overlaying frames.
    """
    # Correct the shape to match your frame dimensions
    frame_count = 15 * 5 #150 // should not be hard-coded!
    height = HEIGHT
    width = WIDTH
    channels = 3

    memmaps = [np.memmap(filename, dtype='uint8', mode='r', shape=(frame_count, height, width, channels)) for filename in memmap_filenames]

    # Create output memmap
    output_memmap = np.memmap(output_memmap_filename, dtype='uint8', mode='w+', shape=(frame_count, height, width, channels))

    for frame_num in range(frame_count):
        composite_frame = memmaps[0][frame_num]
        for memmap in memmaps[1:]:
            composite_frame = alpha_blend_images(composite_frame, memmap[frame_num], alpha)
        output_memmap[frame_num] = composite_frame

    output_memmap.flush()


def stream_memmap_frames(memmap_filename):
    # The shape is set to match (frame_count, height, width, channels)
    frame_count = 15*5  # Or whatever the correct number of frames is
    height = HEIGHT
    width = WIDTH
    channels = 3

    # Create the memmap with the correct shape
    memmap = np.memmap(memmap_filename, dtype='uint8', mode='r', shape=(frame_count, height, width, channels))

    for frame_num in range(frame_count):
        frame = memmap[frame_num]
        cv2.imshow("window", frame)

        # Wait for user input
        key = cv2.waitKey(40)  # Adjust this to match the desired FPS

        # Exit if 'q' is pressed
        if key == ord("q"):
            break


def copy_file(src, dst):
    """
    Copies a file from the source path to the destination path.

    Parameters:
    - src: The source file path.
    - dst: The destination file path.
    """
    try:
        shutil.copy2(src, dst)
        print(f"File copied successfully from {src} to {dst}")
    except FileNotFoundError:
        print(f"Source file not found: {src}")
    except PermissionError:
        print(f"Permission denied: Unable to write to {dst}")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":

    print("Saving first memmap")
    save_frames_to_memmap(duration=5, width=WIDTH, height=HEIGHT, memmap_filename="1.dat")

    print("Saving second memmap")
    save_frames_to_memmap(duration=5, width=WIDTH, height=HEIGHT, memmap_filename="2.dat")
    
    print("Overlaying")
    overlay_frames_from_memmaps(memmap_filenames=["1.dat", "2.dat"],
                                output_memmap_filename="composites.dat", alpha=0.5)
    
    print("Streaming the result")
    stream_memmap_frames(memmap_filename="composites.dat")
