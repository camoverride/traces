import time
from frames import face_detected_mp, save_frames_from_video, overlay_frames_from_dirs



def overlay_faces(duration, width, height, frames_dir, composites_dir, confidence_threshold, alpha):
    """
    Records `duration` seconds of video and writes them to the `frames_dir`,
    overwriting any data that's still there. Then create composite images
    that are an `alpha` blend of the existing images from `composites_dir` and
    the new images from `frames_dir`.
    """
    if face_detected_mp(width=width, height=height, confidence_threshold=confidence_threshold):
        # Save the frames
        print("Face detected - saving frames...")
        save_frames_from_video(duration=duration, width=width, height=height, output_dir=frames_dir)

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
                      width=1080,
                      height=1920,
                      frames_dir="frames",
                      composites_dir="composites",
                      confidence_threshold=0.5,
                      alpha=0.5)
