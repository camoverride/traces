# Ghosts

Ghostly videos of people in a "mirror"

Displayed on an HDMI monitor inside picture frame. RPi 4B with PiCamera.


## Setup

Hide the mouse:

- `sudo apt-get install unclutter`

Python requirements:
- `python -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera` package.)
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

If there's an issue with jpeg:
- `pip install numpy==1.24.3`
- `pip uninstall simplejpeg`
- `pip install --no-cache-dir simplejpeg`

Make folders:
- `mkdir debug_frames`
- `mkdir play_files`


## Test

Monitor:
- `WAYLAND_DISPLAY=wayland-1 wlr-randr --output HDMI-A-1 --transform 90`
- `export DISPLAY=:0`

Python files:
- `python overlay_loop.py`
- `python stream_images_loop.py`


## Play

**Overlay:**

Start a service with *systemd*. This will start the program when the computer starts and revive it when it dies:

- `mkdir -p ~/.config/systemd/user`

- Paste the contents of `overlay.service` into `~/.config/systemd/user/overlay.service`

Start the service using the commands below:

- `systemctl --user daemon-reload`
- `systemctl --user enable overlay.service`
- `systemctl --user start overlay.service`

Start it on boot: `sudo loginctl enable-linger pi`

Get the logs: `journalctl --user -u overlay.service`

**Stream:**

- Paste the contents of `stream.service` into `~/.config/systemd/user/stream.service`

Start the service using the commands below:

- `systemctl --user daemon-reload`
- `systemctl --user enable stream.service`
- `systemctl --user start stream.service`

Start it on boot: `sudo loginctl enable-linger pi`

Get the logs: `journalctl --user -u stream.service`


## To do

- [X] mount, check camera angles
- [ ] hide mouse
- [ ] rotate screen on startup
- [ ] experiment with alpha
- [ ] experiment with detection threshold
