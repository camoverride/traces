import cv2
import logging
import os
import yaml
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

    # Rotate the screen.
    # NOTE: this works for Pi only.
    os.system(f"wlr-randr --output HDMI-A-1 --transform {config['rotation']}")

    # Hide the cursor.
    os.system("unclutter -idle 0 &")

    # Make the display fullscreen.
    cv2.namedWindow(
        "Blended Video",
        cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Blended Video",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN)

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
