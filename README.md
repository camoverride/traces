# Ghosts

Ghostly videos of people in a "mirror" (HDMI monitor inside picture frame. RPi 4B with PiCamera)

## Setup

Monitor:
- `WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90`
- `export DISPLAY=:0`

Python requirements:
- `python -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera` package.)
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

If there's an issue with jpeg:
- `pip install numpy==1.24.3`
- `pip uninstall simplejpeg`
- `pip install --no-cache-dir simplejpeg`


## Play

You need at least two data files to begin, scp them over or generate them from `frames.py`:
- `_composites.dat` and `composites.dat`

- `python overlay_loop.py`
- `python stream_images_loop.py`

currently SLOW - takes ~5 mins to save frames, 