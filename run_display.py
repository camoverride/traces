import logging
import os
import yaml
from _screen_utils import get_os_name, set_up_display, rotate_screen_periodically
from _blending_utils import ThreadedFaceBlender



# Set up logging,
logging.basicConfig(
    level=logging.INFO,
    force=True,
    format='%(levelname)s: %(message)s')



if __name__ == "__main__":
    logging.info("Setting up display")

    # Load the config.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set the display.
    os.environ["DISPLAY"] = ":0"

    # Get the name of the OS. Should be either "raspbian", "ubuntu", or "macos".
    os_name = get_os_name()

    # Set up the display to show images.
    set_up_display(operating_system=os_name)

    # Rotate the screen every few minutes.
    # NOTE: this solves the problem where the code starts but the monitor
    # is not yet turned on.
    rotate_screen_periodically(
        operating_system=os_name,
        rotation=config["rotation"],
        interval_minutes=config["screen_rotation_freq"])

    # Hide the cursor.
    os.system("unclutter -idle 0 &")

    # Initialize blender object.
    blender = ThreadedFaceBlender(
        monitor_width=config["monitor_width"],
        monitor_height=config["monitor_height"],
        frame_rotation=config["camera_rotation"],
        record_seconds=config["recording_duration"],
        alpha=config["temporal_alpha"],
        fps=config["fps"],
        blur_size=config["blur_size"],
        min_area=config["min_area"],
        temporal_alpha=config["temporal_alpha"],
        grid_height=config["grid_height"],
        grid_width=config["grid_width"],
        backup_video_path="video_backup.npz",
        backup_video_save_frequency=config["backup_video_save_freq"])

    # Run!
    blender.run()
