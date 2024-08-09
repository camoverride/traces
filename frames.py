import cv2
import os
import subprocess
import time
import mediapipe as mp
from picamera2 import Picamera2



def get_user():
    # Run the `whoami` command
    result = subprocess.run(["whoami"], capture_output=True, text=True, check=True)
    # Return the output, stripped of any trailing newline
    return result.stdout.strip()


def save_frames_from_video(camera_index=0, num_chunks=4, chunk_duration=5, output_dir="chunks"):
    # Start the camera if it's a pi camera
    if get_user() == "pi":
        # Set up the camera
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (1920, 1080)}))
        picam2.start()
        fps = 30

    # Start the camera if it's on a macbook
    else:
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
    

    chunk_frame_count = int(chunk_duration * fps)
    
    chunk_index = 1
    
    for _ in range(num_chunks):
        # Create directory for each chunk
        chunk_dir = f"{output_dir}/chunks_{chunk_index}"
        os.makedirs(chunk_dir, exist_ok=True)
        
        for frame_num in range(chunk_frame_count):

            if get_user() == "pi":
                frame = picam2.capture_array()
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            frame_filename = os.path.join(chunk_dir, f"frame_{frame_num:04d}.png")
            cv2.imwrite(frame_filename, frame)
        
        chunk_index += 1
        
        # Check if you want to stop capturing
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if get_user() == "pi":
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
    
    else:
        cap.release()
        cv2.destroyAllWindows() # TODO: is this necessary here?


def alpha_blend_images(image1, image2, alpha=0.5):
    """
    Blend two images with a given alpha value.
    """
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)


def overlay_frames_from_dirs(chunk_dirs, output_dir='overlay_dir', alpha=0.5):
    """
    Overlay frames from multiple directories and save the composite frames into an output directory.
    
    Parameters:
    - chunk_dirs: List of directories containing frames for each chunk.
    - output_dir: Directory where the composite frames will be saved.
    - alpha: Blending factor for overlaying frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the first directory to get frame size and list of frames
    first_chunk_dir = chunk_dirs[0]
    frame_files = sorted(os.listdir(first_chunk_dir))
    
    if not frame_files:
        print("No frames found in the first directory.")
        return
    
    # Create a list of paths for each frame
    frame_paths = {frame_file: [os.path.join(chunk_dir, frame_file) for chunk_dir in chunk_dirs]
                   for frame_file in frame_files}
    
    # Iterate over each frame file
    for frame_file, paths in frame_paths.items():
        frames = [cv2.imread(path) for path in paths]
        
        # if None in frames:
        #     print(f"Error loading frames for {frame_file}.")
        #     continue
        
        # Initialize the composite frame with the first chunk
        composite_frame = frames[0]
        
        # Blend each subsequent frame with the composite frame
        for frame in frames[1:]:
            composite_frame = alpha_blend_images(composite_frame, frame, alpha)
        
        # Save the composite frame
        output_frame_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(output_frame_path, composite_frame)


def face_detected_cv():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    # Capture a single frame
    ret, frame = cap.read()

    # Release the webcam
    cap.release()

    if not ret:
        print("Error: Could not read frame from webcam.")
        return False

    # Load the pre-trained Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=10,  # Increase this to reduce false positives
        minSize=(30, 30)
    )

    # Check if any faces are detected
    if len(faces) > 0:
        return True
    else:
        return False


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def face_detected_mp(confidence_threshold=0.5):
    # Get a picture

    # Start the camera if it's a pi camera
    if get_user() == "pi":
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (1920, 1080)}))
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


def stream_images(data_dir="overlay_dir"):
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    for file in sorted(file_paths):
        frame = cv2.imread(file)
        cv2.imshow("window",frame)

        # Wait for user input
        key = cv2.waitKey(20) # TODO: calculate fps
        
        # Exit if 'q' is pressed
        if key == ord("q"):
            break

    # cv2.destroyAllWindows()


# if __name__ == "__main__":
#     while True:
#         # overlay_faces()
#         stream_images(data_dir="overlay_dir")
#         # time.sleep(1)  # Add a delay between iterations


# def image_generator(data_dir="overlay_dir"):
#     while True:
#         file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
#         for file in sorted(file_paths):
#             yield file

# def stream_images(data_dir="overlay_dir"):
#     img_gen = image_generator(data_dir)

#     while True:
#         file = next(img_gen)
#         frame = cv2.imread(file)
#         cv2.imshow("f", frame)

#         # Wait for user input
#         key = cv2.waitKey(20)  # Adjust as needed to control the display speed

#         # Exit if 'q' is pressed
#         if key == ord("q"):
#             break

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     stream_images(data_dir="overlay_dir")



    # # Works as intended
    # while True:
    #     x = face_detected_mp()
    #     print(x)
    #     time.sleep(3)

if __name__ == "__main__":
    # Test face detection
    while True:
        detection = face_detected_mp()
        print(f"Detection Status: {detection}")
        time.sleep(0.5)
