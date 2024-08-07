import cv2
import os



def stream_images(data_dir="overlay_dir"):
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    for file in sorted(file_paths):
        frame = cv2.imread(file)
        cv2.imshow("f",frame)

        # Wait for user input
        key = cv2.waitKey(20) # TODO: calculate fps
        
        # Exit if 'q' is pressed
        if key == ord('q'):
            break