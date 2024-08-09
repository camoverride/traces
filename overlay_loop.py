from datetime import datetime
import os
import random
import time
from frames import face_detected_mp, save_frames_from_video, overlay_frames_from_dirs



def overlay_faces(frames_save_dir, output_overlay_dir, num_overlays, alpha):
    """
    """
    if face_detected_mp():
        # Get the datetime for naming
        datetime_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        print(f"Face detected - saving frames to: {datetime_string} ...")

        # Save the frames
        save_frames_from_video(duration=5,
                               output_dir=datetime_string,
                               output_base_dir="frames")

        # Create composite images.
        print("Overlaying frames ...")

        # Sort the directories from newest to oldest
        frame_dirs = os.listdir(frames_save_dir)
        sorted_frame_dirs = sorted(frame_dirs, key=lambda e: os.path.getmtime(os.path.join(frames_save_dir, e)), reverse=True)
        sorted_frame_dirs = [os.path.join(frames_save_dir, f) for f in sorted_frame_dirs]

        # Sample some directories
        if num_overlays > len(sorted_frame_dirs):
            num_overlays = len(sorted_frame_dirs)
        frame_dir_sample = random.sample(sorted_frame_dirs, num_overlays)

        # Include the sample and the latest
        frames_for_overlay = frame_dir_sample + sorted_frame_dirs[0:0]

        print(frames_for_overlay)
        # Overlay the images
        overlay_frames_from_dirs(chunk_dirs=frames_for_overlay,
                                output_dir=output_overlay_dir,
                                alpha=alpha)


    else:
        print("No face detected!")
        time.sleep(5)

    print("--------------------------------------------")



if __name__ == "__main__":
    FRAMES_SAVE_DIR = "frames"
    OUTPUT_OVERLAY_DIR = "overlay_dir"
    NUM_OVERLAYS = 4
    ALPHA = 0.5


    while True:
        overlay_faces(frames_save_dir=FRAMES_SAVE_DIR,
                      output_overlay_dir=OUTPUT_OVERLAY_DIR,
                      num_overlays=NUM_OVERLAYS,
                      alpha=ALPHA)
