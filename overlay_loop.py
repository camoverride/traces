import time
from frames import face_detected_mp, save_frames_to_memmap, overlay_frames_from_memmaps, copy_file



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
        print("Face detected - saving frames...")
        save_frames_to_memmap(duration=duration,
                              width=width,
                              height=height,
                              memmap_filename=new_images_memmap)

        # Overlay the images
        print("Overlaying frames ...")
        overlay_frames_from_memmaps(memmap_filenames=[new_images_memmap, existing_composite_images_memmap],
                                    output_memmap_filename=new_composite_images_memmap,
                                    alpha=alpha)
        
        # The `new_composite_images_memmap` is now the `existing_composite_images_memmap`
        copy_file(src=new_composite_images_memmap, dst=existing_composite_images_memmap)

    else:
        print("No face detected!")
        time.sleep(5)

    print("--------------------------------------------")



if __name__ == "__main__":

    while True:
        overlay_faces(duration=5,
                      width=1280,#1080,
                      height=720,#1920,
                      new_images_memmap="current_frames.dat",
                      existing_composite_images_memmap="_composites.dat",
                      new_composite_images_memmap="composites.dat",
                      confidence_threshold=0.5,
                      alpha=0.5)
