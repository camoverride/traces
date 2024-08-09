from datetime import datetime
import os
import time
from frames import face_detected_mp, save_frames_to_memmap, overlay_frames_from_memmaps, copy_file



WIDTH, HEIGHT = 1080, 1920
WIDTH, HEIGHT = 1280, 720


def overlay_faces(duration,
                  width,
                  height,
                  new_images_memmap,
                  existing_composite_images_memmap,
                  new_composite_images_memmap,
                  confidence_threshold,
                  alpha):
    """
    Records `duration` seconds of video (`width` x `height`) and writes it to
    `new_images`.dat, which is a np.memmap file. This memmap is then combined with
    the existing `composite_images`.dat to form new composites. There is a
    `confidence_threshold` for face detection and an `alpha` for blending the composites.
    """
    if face_detected_mp(width=width, height=height, confidence_threshold=confidence_threshold):
        # Save the frames
        print(f"Saving frames to {new_images_memmap}")
        save_frames_to_memmap(duration=duration,
                              width=width,
                              height=height,
                              memmap_filename=new_images_memmap)

        # Overlay the images
        print(f"Combining frames from {new_images_memmap} and {existing_composite_images_memmap} to create {new_composite_images_memmap}")
        overlay_frames_from_memmaps(memmap_filenames=[new_images_memmap, existing_composite_images_memmap],
                                    output_memmap_filename=new_composite_images_memmap,
                                    alpha=alpha)

    else:
        print("No face detected!")
        time.sleep(5)

    print("--------------------------------------------")



if __name__ == "__main__":
    play_dir = "play_files"


    while True:
        # Get the time for filenaming
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # Get the most recent composite to add to the incoming frames to create the next composite.
        composites_paths = list(reversed(sorted([os.path.join(play_dir, f) for f in os.listdir(play_dir)])))
        most_recent_composite = composites_paths[0]

        overlay_faces(duration=5,
                      width=WIDTH,
                      height=HEIGHT,
                      new_images_memmap="_current_frames.dat",
                      existing_composite_images_memmap=most_recent_composite,
                      new_composite_images_memmap=f"play_files/{current_time}.dat",
                      confidence_threshold=0.5,
                      alpha=0.5)
        
        # Clean up old files from play dir
        if len(composites_paths) > 5:
            for f in composites_paths[5:]:
                os.remove(f)
