from datetime import datetime
import os
import time
from frames import face_detected_mp, save_frames_from_video, overlay_frames_from_dirs



def overlay_faces():
    """
    """
    if face_detected_mp():
        print("Face detected - saving frames from video")
        save_frames_from_video(camera_index=0,
                            num_chunks=4,
                            chunk_duration=5,
                            output_dir="chunks")

        print("Overlaying frames")
        chunk_dirs = [os.path.join(("chunks"), d) for d in os.listdir("chunks") if "chunks_" in d]
        overlay_frames_from_dirs(chunk_dirs=chunk_dirs,
                                output_dir="overlay_dir",
                                alpha=0.5)
        
        with open("recording_time.txt", "w") as f:
            f.write(str(datetime.now()))

    else:
        print("No face detected!")
        time.sleep(5)

    print("--------------------------------------------")

if __name__ == "__main__":
    while True:
        overlay_faces()
