# Traces 👻

Ghostly videos of people in a "mirror."


## Setup

Hardware:

- Beelik Mini PC running Ubuntu [link](OSBOT webcam [link](https://www.amazon.com/dp/B0D9W7J9SK?ref=ppx_yo2ov_dt_b_fed_asin_title))
- Sceptre TV [link](https://www.temu.com/-50-4k-televison-uhd-3840x2160-tv--bezel-design--viewing-wall-mount-ready-2x10w-speakers-metal-black-g-602848771304761.html)
- OSBOT webcam [link](https://www.amazon.com/dp/B0D9W7J9SK?ref=ppx_yo2ov_dt_b_fed_asin_title)


MacOS (testing):

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements_macos.txt`
- `pip install git+https://github.com/ageitgey/face_recognition_models`


Ubuntu (production):

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `python3 -m pip install -r requirements.txt`
- `curl https://gitlab.com/Oschowa/gnome-randr/-/raw/master/gnome-randr.py -o gnome-randr.py`
- `chmod +x gnome-randr.py`
- `sudo apt update && sudo apt install -y build-essential libdbus-1-dev python3-dev libglib2.0-dev`
- `pip install dbus-python`
<!-- - `python3 -m pip install git+https://github.com/ageitgey/face_recognition_models` -->

Potentially required to set up webcam in 4K mode:

- `v4l2-ctl --device=/dev/video1 --set-fmt-video=width=3840,height=2160,pixelformat=MJPG`
- `v4l2-ctl --device=/dev/video1 --get-fmt-video`


## Test

- `python run_display.py`


## Run in Production

Start a system service to run the code automatically at system startup:

- `mkdir -p ~/.config/systemd/user`
- `cat display.service > ~/.config/systemd/user/display.service`
- `systemctl --user daemon-reload`
- `systemctl --user enable display.service`
- `systemctl --user start display.service`
- `sudo loginctl enable-linger $(whoami)`

Show the logs:

- `journalctl --user -u display.service`

Clear logs:

- `sudo journalctl --unit=display.service --rotate`
- `sudo journalctl --vacuum-time=1s`

After running for a few days, the USB port connected to the camera might lose power. To fix this, schedule nightly reboots:

- `sudo crontab -e`

Enter the following lines:

- `0 0 * * * /sbin/shutdown -r now`
