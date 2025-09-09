import cv2
import logging
import os
import yaml
from _screen_utils import get_os_name, rotate_screen
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

    # Rotate the screen.
    rotate_screen(
        operating_system=os_name,
        rotation=config["rotation"])

    # Hide the cursor.
    os.system("unclutter -idle 0 &")

    # Make the display fullscreen.
    # cv2.namedWindow(
    #     "Display Image",
    #     cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(
    #     "Display Image",
    #     cv2.WND_PROP_FULLSCREEN,
    #     cv2.WINDOW_FULLSCREEN)

    # Initialize blender object.
    blender = ThreadedFaceBlender(
        record_seconds=config["recording_duration"],
        alpha=config["temporal_alpha"],
        fps=config["fps"],
        blur_size=config["blur_size"],
        min_area=config["min_area"],
        temporal_alpha=config["temporal_alpha"])

    # Run!
    blender.run()
