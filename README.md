# Music Visualization (System Audio → 3D Spectrum)

This is an MVP that captures **system audio output** (Windows WASAPI loopback) and renders a **real-time 3D frequency spectrum** as a spiral point cloud.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
## Run

```bash
python music_viz_server.py
```

## Notes

- If you hear audio but see a flat line, make sure something is actually playing on your default output device.
- You can tweak visualization + FFT settings in `VizConfig` inside `music_viz_server.py`.

