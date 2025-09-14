import cv2
import os
import platform
import re
import numpy as np
import subprocess
import time



def set_up_display(operating_system : str) -> None:
    """
    Sets the OpenCV display canvas to be fullscreen by creating a
    cv2.namedWindow object addressable as "Display Image".

    Parameters
    ----------
    operating_system : str
        The name of the OS. Current options are:
            - "raspbian"
            - "ubuntu"
            - "macos"

    Returns
    -------
    None
        Creates a fullscreen canvas for displaying images.
    """
    # This is an absolutely disgusting hack to get fullscreen enabled.
    if operating_system == "ubuntu":
        os.environ["DISPLAY"] = ':0'
        os.environ["QT_QPA_PLATFORM"] = "xcb"  # Force Qt to use X11
        os.environ["GDK_BACKEND"] = "x11"      # Force GTK to use X11
        time.sleep(5)

        # Create window as normal first.
        cv2.namedWindow("Display Image", cv2.WINDOW_NORMAL)

        # Show an image first, THEN set fullscreen.
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow("Display Image", dummy_image)

        # Brief wait to ensure window is created.
        cv2.waitKey(100)

        # Now set fullscreen.
        cv2.setWindowProperty(
            "Display Image",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN)


def get_os_name() -> str:
    """
    Determines which operating system we are using.

    Parameters
    ----------
    None
        Performs command line operations to determine OS.

    Returns
    -------
    str
        One of the three operating systems this code
        has been tested on:
            - "raspbian"
            - "ubuntu"
            - "macos"

    Raises
    ------
    ValueError
        If the specified operating system is not supported.
    """
    # Get the system.
    system = platform.system()

    # MacOS
    if system == "Darwin":
        operating_system = "macos"

    # If Linux, it might be Raspbian or Ubuntu.
    elif system == "Linux":
        distro = platform.freedesktop_os_release().get("ID", "").lower()

        if "debian" in distro or "raspberrypi" in distro:
            operating_system = "raspbian"

        elif "ubuntu" in distro:
            operating_system = "ubuntu"

    # Raise an error if the OS is not supported.
    if operating_system not in ["raspbian", "ubuntu", "macos"]:
        raise ValueError(f"Unsupported operating system: '{operating_system}'")

    return operating_system


def get_display_info(operating_system : str) -> dict:
    """
    Returns the width and height of the primary monitor in pixels.

    Parameters
    ----------
    operating_system : str
        The identity of the operating system returned by `get_os_name`
        Current options are:
            - "raspbian"
            - "ubuntu"
            - "macos"

    Returns
    -------
    dict
        A dict with the following keys:
            "width": int, width of the monitor (in pixels)
            "height": int, height of the monitor (in pixels)
            "output": str, the output device, e.g. "HDMI-1"

    Raises
    ------
    ValueError
        If the specified operating system is not supported.
    """
    # Raise an error if the OS is not supported.
    if operating_system not in ["raspbian", "ubuntu", "macos"]:
        raise ValueError(f"Unsupported operating system: '{operating_system}'")

    # MacOS.
    if operating_system == "macos":
        # Use system_profiler to get display info.
        output = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True)
        match = re.search(r"Resolution: (\d+) x (\d+)", output)
        if match:
            width, height = map(int, match.groups())

        # The output device doesn't matter for MacOS
        output_device = "NA"

    # Raspbian.
    elif operating_system == "raspbian":
        # Use fbset to get monitor dimensions.
        output = subprocess.check_output(["fbset"], text=True)
        match = re.search(r"mode\s+\"(\d+)x(\d+)\"", output)
        if match:
            width, height = map(int, match.groups())

        # Grep through connected displays to get the correct one.
        command = "DISPLAY=:0 xrandr | grep ' connected'"
        result = subprocess.check_output(command, shell=True, text=True)
        output_device = result.split(" ")[0]

    # Ubuntu.
    elif operating_system == "ubuntu":
        # Use xrandr to get monitor dimensions.
        env = os.environ.copy()
        env['DISPLAY'] = ':0'

        output = subprocess.check_output(["xrandr"], env=env, text=True)
        match = re.search(r"current\s+(\d+)\s+x\s+(\d+)", output)
        if match:
            width, height = map(int, match.groups())

        # Grep through connected displays to get the correct one.
        command = "xrandr | grep ' connected'"
        result = subprocess.check_output(command, env=env, shell=True, text=True)
        output_device = result.split(" ")[0]

    # Store everything in a dict.
    display_info = {
        "width": width,
        "height": height,
        "output_device": output_device
    }

    # Print debug.
    print(f"Monitor width: {width}")
    print(f"Monitor height: {height}")
    print(f"Output: {output_device}")

    return display_info


def rotate_screen(
    operating_system : str,
    rotation: str) -> None:
    """
    Rotates the screen to the desired angle.

    Parameters
    ----------
    operating_system : str
        The identity of the operating system returned by `get_os_name`
        Current options are:
            - "raspbian"
            - "ubuntu"
            - "macos"
    rotation : str
        Current options are:
            - "left"
            - "right"
            - "flip"
            - "normal" (no rotation)

    Returns
    -------
    None
        Rotates the monitor.
    """
    display_info = get_display_info(operating_system=operating_system)

    # Raspbian.
    if operating_system == "raspbian":

        # Translate the flip intp degrees.
        if rotation == "left":
            rotation_degs = 90
        elif rotation == "right":
            rotation_degs = 270
        elif rotation == "normal":
            rotation_degs = 0
        elif rotation == "flip":
            rotation_degs = 180

        os.system(f"WAYLAND_DISPLAY={display_info['output_device']} wlr-randr \
                     --output {display_info['output_device']}--transform {rotation_degs}")

    # Ubuntu.
    elif operating_system == "ubuntu":
        # Set to normal.
        os.system(f"./gnome-randr.py --output {display_info['output_device']} \
                    --rotate normal")

        time.sleep(2)

        # Then rotate.
        os.system(f"./gnome-randr.py --output {display_info['output_device']} \
                    --rotate {rotation}")

        time.sleep(2)

    # MacOS.
    elif operating_system == "macos":
        # MacOS is for testing only.
        pass
