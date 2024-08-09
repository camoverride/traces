import time
from frames import face_detected_mp, save_frames_from_video, overlay_frames_from_dirs



def overlay_faces(duration, frames_dir, composites_dir, alpha):
    """
    """
    if face_detected_mp():
        # Save the frames
        print("Face detected - saving frames...")
        save_frames_from_video(duration=duration, output_dir=frames_dir)

        # Overlay the images
        print("Overlaying frames ...")
        overlay_frames_from_dirs(chunk_dirs=[frames_dir, composites_dir],
                                output_dir=composites_dir,
                                alpha=alpha)


    else:
        print("No face detected!")
        time.sleep(5)

    print("--------------------------------------------")



if __name__ == "__main__":

    while True:
        overlay_faces(duration=5,
                      frames_dir="frames",
                      composites_dir="composites",
                      alpha=0.5)
