import asyncio
import errno
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import webbrowser

import numpy as np
import pyaudiowpatch as pyaudio
from aiohttp import web


@dataclass
class VizConfig:
    # Audio
    sample_rate: int = 48_000
    # Larger block_size improves low-frequency resolution (less "bass always on"),
    # at the cost of latency and CPU.
    block_size: int = 4096
    hop_size: int = 1024
    # Optional: substring of the *playback* device name to pick loopback, e.g. "Realtek"
    # (also reads env MUSIC_VIZ_OUTPUT_SUBSTR if empty)
    output_name_substr: str = ""

    # Spectrum
    f_min: float = 30.0
    f_max: float = 18_000.0
    n_points: int = 512
    smoothing: float = 0.84
    # Stable dB mapping avoids per-frame min/max normalization biasing bass.
    db_floor: float = -80.0  # lower clamp (silence / noise floor)
    db_ceil: float = -30.0  # upper clamp (very loud)
    # Auto-range (slow AGC) to prevent constant clipping/saturation.
    autorange: bool = True
    autorange_low_pct: float = 15.0
    autorange_high_pct: float = 98.0
    autorange_attack: float = 0.12  # 0..1, faster = adapts quicker to louder content
    autorange_release: float = 0.02  # 0..1, slower = steadier baseline
    autorange_headroom_db: float = 6.0  # keep peaks from pinning at 1.0
    autorange_min_span_db: float = 45.0  # minimum dynamic range
    # Frequency tilt (simple "EQ"): positive values boost highs vs lows (reduces bass dominance).
    tilt_db_per_decade: float = 6.0
    tilt_ref_hz: float = 200.0

    # Server
    host: str = "127.0.0.1"
    port: int = 8766  # override with env MUSIC_VIZ_PORT


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def _log_space_freqs(fmin: float, fmax: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(fmin), np.log(fmax), n))


def _interp_spectrum(freqs: np.ndarray, mag: np.ndarray, target_freqs: np.ndarray) -> np.ndarray:
    return np.interp(target_freqs, freqs, mag)


def _strip_loopback_suffix(name: str) -> str:
    return name.replace(" [Loopback]", "").strip()


def _pick_wasapi_loopback(pa: pyaudio.PyAudio, default_out: dict, cfg: VizConfig) -> dict:
    """
    Match the default WASAPI *output* device to its loopback input.
    Names are not always identical; compare normalized strings and allow substring matches.
    """
    candidates = list(pa.get_loopback_device_info_generator())
    if not candidates:
        raise RuntimeError("No WASAPI loopback devices found")

    out_name = (default_out.get("name") or "").strip()
    out_norm = _strip_loopback_suffix(out_name).lower()

    frag = (cfg.output_name_substr or os.environ.get("MUSIC_VIZ_OUTPUT_SUBSTR", "") or "").strip().lower()
    if frag:
        for dev in candidates:
            if frag in (dev.get("name") or "").lower():
                return dev

    # Exact match ignoring loopback suffix
    for dev in candidates:
        if _strip_loopback_suffix(dev.get("name") or "").lower() == out_norm:
            return dev

    # Playback name appears inside loopback name (or vice versa)
    for dev in candidates:
        lb = (dev.get("name") or "").lower()
        if out_name and out_name.lower() in lb:
            return dev
        base = _strip_loopback_suffix(dev.get("name") or "")
        if base and base.lower() in out_name.lower():
            return dev

    # Last resort: first loopback (often default order follows default devices)
    return candidates[0]


class LoopbackStream:
    def __init__(self, cfg: VizConfig):
        self.cfg = cfg
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._stream_err: Exception | None = None

        self.window = _hann(cfg.block_size).astype(np.float32)
        self.target_freqs = _log_space_freqs(cfg.f_min, cfg.f_max, cfg.n_points).astype(np.float32)
        self.prev = np.zeros(cfg.n_points, dtype=np.float32)
        self.freqs = np.fft.rfftfreq(cfg.block_size, 1.0 / cfg.sample_rate).astype(np.float32)
        self._db_lo: float | None = None
        self._db_hi: float | None = None

        self._ring = np.zeros(cfg.block_size * 4, dtype=np.float32)
        self._w = 0
        self._count = 0
        self._lock = threading.Lock()
        self.last_peak: float = 0.0
        self.device_label: str = ""

    def start(self):
        """
        Stable Windows loopback capture using PortAudio WASAPI via pyaudiowpatch.
        """
        if self._stream is not None:
            return

        self._pa = pyaudio.PyAudio()

        wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out_idx = wasapi_info["defaultOutputDevice"]
        default_out = self._pa.get_device_info_by_index(default_out_idx)

        loopback = _pick_wasapi_loopback(self._pa, default_out, self.cfg)
        self.device_label = f"{default_out.get('name')} -> {loopback.get('name')} (idx {loopback.get('index')})"
        print(f"Loopback capture: {self.device_label}")

        # Many devices reject arbitrary sample rates; use the device default.
        device_rate = int(loopback.get("defaultSampleRate") or self.cfg.sample_rate)
        if device_rate != self.cfg.sample_rate:
            self.cfg.sample_rate = device_rate
            self.freqs = np.fft.rfftfreq(self.cfg.block_size, 1.0 / self.cfg.sample_rate).astype(np.float32)

        channels = int(loopback.get("maxInputChannels") or 2)
        channels = max(1, min(channels, 2))

        def callback(in_data, frame_count, time_info, status_flags):  # noqa: ARG001
            try:
                x = np.frombuffer(in_data, dtype=np.float32)
                if channels > 1:
                    x = x.reshape(-1, channels).mean(axis=1, dtype=np.float32)
                with self._lock:
                    self._push(x)
                return (None, pyaudio.paContinue)
            except Exception as e:  # noqa: BLE001
                self._stream_err = e
                return (None, pyaudio.paAbort)

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=device_rate,
            input=True,
            input_device_index=loopback["index"],
            frames_per_buffer=self.cfg.hop_size,
            stream_callback=callback,
        )
        self._stream.start_stream()

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            finally:
                self._pa = None

    def _push(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = x.size
        m = self._ring.size
        end = self._w + n
        if end <= m:
            self._ring[self._w : end] = x
        else:
            k = m - self._w
            self._ring[self._w :] = x[:k]
            self._ring[: end - m] = x[k:]
        self._w = (self._w + n) % m
        self._count = min(m, self._count + n)

    def _get_block(self) -> np.ndarray | None:
        if self._count < self.cfg.block_size:
            return None
        m = self._ring.size
        end = self._w
        start = (end - self.cfg.block_size) % m
        if start < end:
            return self._ring[start:end].copy()
        return np.concatenate((self._ring[start:], self._ring[:end])).copy()

    def poll_frame(self) -> np.ndarray | None:
        if self._stream is None:
            return None
        if self._stream_err is not None:
            # Surface the first async exception to caller.
            raise RuntimeError(f"Audio stream callback error: {self._stream_err}") from self._stream_err

        with self._lock:
            blk = self._get_block()
        if blk is None:
            return None

        x = blk * self.window
        spec = np.fft.rfft(x)
        mag = np.abs(spec).astype(np.float32)
        # Convert to dBFS-ish magnitude (relative), then apply gentle frequency tilt.
        db = 20.0 * np.log10(np.maximum(mag, 1e-12)).astype(np.float32)
        if self.cfg.tilt_db_per_decade != 0.0:
            f = np.maximum(self.freqs, 1.0).astype(np.float32)
            tilt = float(self.cfg.tilt_db_per_decade) * np.log10(f / float(self.cfg.tilt_ref_hz)).astype(np.float32)
            db = db + tilt

        db_s = _interp_spectrum(self.freqs, db, self.target_freqs).astype(np.float32)
        # Map dB -> 0..1 using either fixed range or slow auto-range (recommended).
        if self.cfg.autorange:
            lo_t = float(np.percentile(db_s, float(self.cfg.autorange_low_pct)))
            hi_t = float(np.percentile(db_s, float(self.cfg.autorange_high_pct))) + float(self.cfg.autorange_headroom_db)

            if self._db_lo is None or self._db_hi is None:
                self._db_lo, self._db_hi = lo_t, hi_t
            else:
                a_lo = float(self.cfg.autorange_attack) if lo_t > float(self._db_lo) else float(self.cfg.autorange_release)
                a_hi = float(self.cfg.autorange_attack) if hi_t > float(self._db_hi) else float(self.cfg.autorange_release)
                self._db_lo = (1.0 - a_lo) * float(self._db_lo) + a_lo * lo_t
                self._db_hi = (1.0 - a_hi) * float(self._db_hi) + a_hi * hi_t

            lo = float(self._db_lo)
            hi = float(self._db_hi)
            min_span = float(self.cfg.autorange_min_span_db)
            if hi - lo < min_span:
                mid = 0.5 * (hi + lo)
                lo = mid - 0.5 * min_span
                hi = mid + 0.5 * min_span
        else:
            lo = float(self.cfg.db_floor)
            hi = float(self.cfg.db_ceil)
        denom = (hi - lo) if (hi - lo) > 1e-6 else 1.0
        sampled = np.clip((db_s - lo) / denom, 0.0, 1.0).astype(np.float32)

        s = float(self.cfg.smoothing)
        out = s * self.prev + (1.0 - s) * sampled
        self.prev = out
        self.last_peak = float(np.max(np.abs(blk)))
        return out


async def index(request: web.Request) -> web.Response:
    root = Path(__file__).parent
    html = (root / "web" / "index.html").read_text(encoding="utf-8")
    # #region agent log
    try:
        _log_path = Path(__file__).resolve().parent / "debug-f98fec.log"
        _payload = {
            "sessionId": "f98fec",
            "hypothesisId": "H1",
            "location": "music_viz_server.py:index",
            "message": "served_index_html",
            "data": {
                "html_bytes": len(html),
                "contains_axis_legends_root": 'id="axis-legends"' in html,
                "contains_freq_legend_text": "Frequency (Hz)" in html,
            },
            "timestamp": int(time.time() * 1000),
        }
        with _log_path.open("a", encoding="utf-8") as _lf:
            _lf.write(json.dumps(_payload) + "\n")
    except Exception:
        pass
    # #endregion
    return web.Response(text=html, content_type="text/html")


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    # Avoid heartbeat traffic; main loop must yield or the event loop can starve on Windows.
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    stream: LoopbackStream = request.app["stream"]
    cfg: VizConfig = request.app["cfg"]

    try:
        await ws.send_str(
            json.dumps(
                {
                    "type": "hello",
                    "sample_rate": cfg.sample_rate,
                    "block_size": cfg.block_size,
                    "hop_size": cfg.hop_size,
                    "f_min": cfg.f_min,
                    "f_max": cfg.f_max,
                    "n_points": cfg.n_points,
                    "device": getattr(stream, "device_label", ""),
                }
            )
        )

        last_send = 0.0
        while True:
            frame = stream.poll_frame()
            if frame is None:
                await asyncio.sleep(0.02)
                continue

            now = time.time()
            # Cap to ~60 fps even if hop_size is tiny
            if now - last_send < 1.0 / 60.0:
                await asyncio.sleep(0.004)
                continue
            last_send = now

            await ws.send_str(
                json.dumps(
                    {
                        "type": "frame",
                        "a": frame.tolist(),
                        "peak": float(stream.last_peak),
                    }
                )
            )
    except asyncio.CancelledError:
        raise
    except ConnectionResetError:
        # aiohttp's ClientConnectionResetError subclasses ConnectionResetError; peer closed
        # while we were sending — normal when the user refreshes or closes the tab.
        pass
    except BrokenPipeError:
        pass
    except Exception:
        # Other disconnect / protocol errors while streaming.
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
    return ws


async def on_startup(app: web.Application):
    # Opening WASAPI devices can block; keep it off the asyncio thread so HTTP/WS accept quickly.
    loop = asyncio.get_running_loop()
    stream: LoopbackStream = app["stream"]
    await loop.run_in_executor(None, stream.start)


async def on_cleanup(app: web.Application):
    app["stream"].close()


def _address_already_in_use(exc: OSError) -> bool:
    if exc.errno == errno.EADDRINUSE:
        return True
    # Windows
    return getattr(exc, "winerror", None) == 10048


def main():
    # aiohttp + websockets are often more stable on Windows
    # with the selector event loop policy.
    if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    cfg = VizConfig()
    if not cfg.output_name_substr:
        cfg.output_name_substr = os.environ.get("MUSIC_VIZ_OUTPUT_SUBSTR", "")
    port_env = os.environ.get("MUSIC_VIZ_PORT", "").strip()
    if port_env:
        cfg.port = int(port_env)
    preferred_port = cfg.port

    vendor_dir = (Path(__file__).resolve().parent / "web" / "vendor").resolve()
    if (
        not (vendor_dir / "three.module.js").is_file()
        or not (vendor_dir / "three.core.js").is_file()
        or not (vendor_dir / "OrbitControls.js").is_file()
    ):
        print(
            "WARNING: Missing web/vendor/three.module.js, three.core.js, or OrbitControls.js. "
            "three.module.js imports three.core.js from the same Three.js release; copy both from "
            "https://unpkg.com/three@VERSION/build/ (no CDN required at runtime)."
        )

    open_browser = os.environ.get("MUSIC_VIZ_NO_BROWSER", "").strip() not in ("1", "true", "yes")
    for attempt in range(10):
        cfg.port = preferred_port + attempt
        url = f"http://{cfg.host}:{cfg.port}/"
        app = web.Application()
        app["cfg"] = cfg
        app["stream"] = LoopbackStream(cfg)
        app.on_startup.append(on_startup)
        app.on_cleanup.append(on_cleanup)
        app.router.add_static("/vendor/", str(vendor_dir))
        app.router.add_get("/", index)
        app.router.add_get("/ws", ws_handler)
        try:
            print("\n--- Music viz ---")
            print(f"Open this link in your browser: {url}")
            print("(Ctrl+click the URL in many terminals, or copy it into the address bar.)\n")
            if open_browser:
                webbrowser.open(url)
            web.run_app(app, host=cfg.host, port=cfg.port)
            return
        except OSError as e:
            if not _address_already_in_use(e) or attempt == 9:
                raise
            print(
                f"Port {cfg.port} is already in use (another music_viz_server?). "
                f"Trying {cfg.port + 1}… (or close the other process, or set MUSIC_VIZ_PORT)"
            )


if __name__ == "__main__":
    main()

