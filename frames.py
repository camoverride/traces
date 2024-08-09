import shutil
import subprocess
import time
import cv2
import numpy as np
import mediapipe as mp



WIDTH, HEIGHT = 1080, 1920


def get_user():
    """
    Get the user as a string. Raspberry Pi's should have the username "pi"
    """
    # Run the `whoami` command
    result = subprocess.run(["whoami"], capture_output=True, text=True, check=True)
    # Return the output, stripped of any trailing newline
    return result.stdout.strip()

USER = get_user()

if USER == "pi":
    from picamera2 import Picamera2


def save_frames_to_memmap(duration, width, height, memmap_filename):
    """
    Saves `duration` (seconds) of frames to a numpy memmap object.
    The memmap file is named `memmap_filename`.
    """
    # Start the camera if it's a Raspberry Pi camera
    if USER == "pi":
        # Set up the camera
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888",
                                                                   "size": (width, height)}))
        picam2.start()
        fps = 30

    # Start the camera if it's on a MacBook
    else:
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)

    # How many frames to record?
    frame_count = int(duration * fps)

    # Create a memory-mapped array to store the frames
    memmap_shape = (frame_count, height, width, 3)  # Correctly shaping the memmap array
    memmap = np.memmap(memmap_filename, dtype='uint8', mode='w+', shape=memmap_shape)

    for frame_num in range(frame_count):

        if USER == "pi":
            frame = picam2.capture_array()
        else:
            time.sleep(0.1) # So first images aren't black/
            ret, frame = cap.read()
            if not ret:
                break

        memmap[frame_num] = frame  # Store the frame in the correct index

    # Finalize the memmap file
    memmap.flush()

    if USER == "pi":
        picam2.stop()
        picam2.close()

    else:
        cap.release()
        cv2.destroyAllWindows()


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
    frame_count = 150
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


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def face_detected_mp(width, height, confidence_threshold=0.5):
    # Get a picture

    # Start the camera if it's a pi camera
    if USER == "pi":
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888",
                                                                   "size": (width, height)}))
        picam2.start()

        frame = picam2.capture_array()
        picam2.stop()
        picam2.close()

    # Start the camera if it's on a macbook
    else:
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return False

        # Capture a single frame from the webcam
        time.sleep(1) # pause required or first image will be murky and dark
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            cap.release()  # Ensure release on error
            return False

        # Release the webcam
        cap.release()

    # Convert the image from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(min_detection_confidence=confidence_threshold) as face_detection:
        # Process the frame and detect faces
        results = face_detection.process(frame_rgb)

        # Check if any faces are detected
        if results.detections:
            return True
        else:
            return False


def stream_memmap_frames(memmap_filename):
    memmap = np.memmap(memmap_filename, dtype='uint8', mode='r')

    # Print the shape of the memmap to debug
    print(f"Memmap shape: {memmap.shape}")

    if memmap.ndim == 1:
        # If the shape is 1D, calculate the correct shape
        total_elements = memmap.size
        height = HEIGHT
        width = WIDTH
        channels = 3

        # Calculate the frame count
        frame_count = total_elements // (height * width * channels)

        # Reshape the memmap to the correct 4D shape
        memmap = memmap.reshape((frame_count, height, width, channels))

    frame_count, height, width, channels = memmap.shape

    for frame_num in range(frame_count):
        frame = memmap[frame_num]
        cv2.imshow("window", frame)

        # Wait for user input
        key = cv2.waitKey(20)  # Adjust this to match the desired FPS

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
    save_frames_to_memmap(duration=5, width=WIDTH, height=HEIGHT, memmap_filename="current_frames.dat")

    print("Saving second memmap")
    save_frames_to_memmap(duration=5, width=WIDTH, height=HEIGHT, memmap_filename="_composites.dat")
    
    print("Overlaying")
    overlay_frames_from_memmaps(memmap_filenames=["current_frames.dat", "_composites.dat"],
                                output_memmap_filename="composites.dat", alpha=0.5)
    
    print("Streaming the result")
    stream_memmap_frames(memmap_filename="composites.dat")
