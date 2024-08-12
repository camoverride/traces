# Ghosts

Ghostly videos of people in a "mirror"

Displayed on an HDMI monitor inside picture frame. RPi 4B with PiCamera.


## Setup

Python requirements:
- `python -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera` package.)
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

If there's an issue with jpeg:
- `pip install numpy==1.24.3`
- `pip uninstall simplejpeg`
- `pip install --no-cache-dir simplejpeg`


## Test

Monitor:
- `WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90`
- `export DISPLAY=:0`

Python files:
- `python overlay_loop.py`
- `python stream_images_loop.py`


## Play

- set up system service



## To do

- figure out camera dimensions
- figure out why mediapipe isn't detecting faces unless they are quite close
