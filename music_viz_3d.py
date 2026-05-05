import sys

# Windows: used when Python first initializes COM (before pywin32 / WASAPI paths).
if sys.platform == "win32":
    sys.coinit_flags = 2  # COINIT_APARTMENTTHREADED — typical for Qt OLE embedding

import time
from dataclasses import dataclass

import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

# Reduce noisy PyOpenGL error-checking (some Qt/ANGLE paths can trigger benign errors).
import OpenGL  # noqa: E402

OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = False

import pyqtgraph as pg
import pyqtgraph.opengl as gl


@dataclass
class VizConfig:
    # Audio
    sample_rate: int = 48_000
    block_size: int = 2048  # FFT window size
    hop_size: int = 512  # how many samples per update

    # FFT / spectrum
    f_min: float = 30.0
    f_max: float = 18_000.0
    n_points: int = 384  # how many frequency samples to render
    smoothing: float = 0.84  # 0..1, higher = smoother (more lag)

    # 2D plane surface (frequency × depth); amplitude drives mesh height (Z in GL coords)
    plane_width: float = 220.0  # X extent (frequency span)
    plane_depth: float = 220.0  # Y extent (history depth / distance)
    plane_depth_cells: int = 120  # depth resolution; higher = smoother scrolling grid
    height: float = 28.0  # amplitude 0..1 -> added height on Z
    floor_z: float = -6.0  # base height of the surface when amplitude is zero

    # UI
    fps: int = 60


def _hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def _log_space_freqs(fmin: float, fmax: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(fmin), np.log(fmax), n))


def _interp_spectrum(freqs: np.ndarray, mag: np.ndarray, target_freqs: np.ndarray) -> np.ndarray:
    # freqs are linear FFT bin frequencies (ascending). target_freqs are within range.
    return np.interp(target_freqs, freqs, mag)


def _format_hz(f: float) -> str:
    """Human-readable Hz label for axis ticks."""
    if f < 1000.0:
        return f"{f:.0f} Hz" if f >= 100 else f"{f:g} Hz"
    k = f / 1000.0
    if k < 10.0:
        s = f"{k:.2f}".rstrip("0").rstrip(".")
        return f"{s} kHz"
    return f"{k:.1f} kHz"


def _x_for_frequency(f_hz: float, f_min: float, f_max: float, plane_width: float) -> float:
    """Map log-spaced frequency to world X (same mapping as spectrum bins along plane_width)."""
    lf0, lf1 = np.log(f_min), np.log(f_max)
    t = (np.log(max(f_hz, f_min * 1.0001)) - lf0) / (lf1 - lf0)
    t = float(np.clip(t, 0.0, 1.0))
    return -plane_width * 0.5 + t * plane_width


def _freq_colormap_rgba(t: np.ndarray) -> np.ndarray:
    """
    Per-bin color: low frequency (t≈0) -> red, high frequency (t≈1) -> purple.
    t: shape (n,) in [0, 1]. Returns (n, 4) float32 RGBA.
    """
    t = np.clip(t.astype(np.float64), 0.0, 1.0)
    # HSV: hue 0 (red) -> ~0.78 (violet/purple), full saturation/value
    h = t * (280.0 / 360.0)
    s = np.ones_like(h) * 0.92
    v = np.ones_like(h)
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t_ = v * (1.0 - (1.0 - f) * s)
    r = np.choose(i, [v, q, p, p, t_, v])
    g = np.choose(i, [t_, v, v, q, p, p])
    b = np.choose(i, [p, p, t_, v, v, q])
    out = np.stack([r, g, b, np.ones_like(t)], axis=1).astype(np.float32)
    return out


class LoopbackCapture:
    """
    Windows loopback capture via soundcard (WASAPI).
    Captures the system output ("what you hear").
    """

    def __init__(self, cfg: VizConfig):
        self.cfg = cfg
        self._mic = None
        self._rec = None
        self._ring = np.zeros(cfg.block_size * 4, dtype=np.float32)
        self._ring_w = 0
        self._ring_count = 0

    def start(self):
        # Import after QApplication exists so soundcard/WASAPI does not initialize COM
        # before Qt (avoids OleInitialize RPC_E_CHANGED_MODE 0x80010106 on Windows).
        import soundcard as sc

        spk = sc.default_speaker()
        # In soundcard, loopback capture is a "Microphone" created from the speaker.
        self._mic = sc.get_microphone(spk.name, include_loopback=True).recorder(
            samplerate=self.cfg.sample_rate,
            channels=1,
            blocksize=self.cfg.hop_size,
        )
        self._rec = self._mic.__enter__()

    def close(self):
        if self._mic is not None:
            try:
                self._mic.__exit__(None, None, None)
            finally:
                self._mic = None
                self._rec = None

    def read_hop(self) -> None:
        if self._rec is None:
            return
        x = self._rec.record(numframes=self.cfg.hop_size)
        # soundcard returns float32 in [-1, 1] shape (frames, channels)
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = x.size
        ring = self._ring
        w = self._ring_w
        m = ring.size
        end = w + n
        if end <= m:
            ring[w:end] = x
        else:
            k = m - w
            ring[w:] = x[:k]
            ring[: end - m] = x[k:]
        self._ring_w = (w + n) % m
        self._ring_count = min(m, self._ring_count + n)

    def get_block(self) -> np.ndarray | None:
        if self._ring_count < self.cfg.block_size:
            return None
        ring = self._ring
        m = ring.size
        # block ends at write pointer
        end = self._ring_w
        start = (end - self.cfg.block_size) % m
        if start < end:
            blk = ring[start:end].copy()
        else:
            blk = np.concatenate((ring[start:], ring[:end])).copy()
        return blk


class SpectrumPlaneViz(QtWidgets.QMainWindow):
    def __init__(self, cfg: VizConfig):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("System Audio Spectrum Plane (Loopback)")

        # Capture + FFT prep
        self.cap = LoopbackCapture(cfg)
        self.window = _hann(cfg.block_size).astype(np.float32)
        self.freqs = np.fft.rfftfreq(cfg.block_size, 1.0 / cfg.sample_rate).astype(np.float32)
        self.target_freqs = _log_space_freqs(cfg.f_min, cfg.f_max, cfg.n_points).astype(np.float32)
        self.prev = np.zeros(cfg.n_points, dtype=np.float32)

        # UI: GL view with frequency–depth surface (amplitude -> Z height)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = gl.GLViewWidget()
        self.view.opts["distance"] = 240
        self.view.opts["fov"] = 52
        # Camera tuned for a "flying over a grid" look (like the reference screenshot).
        self.view.setCameraPosition(distance=240, elevation=18, azimuth=-90)
        layout.addWidget(self.view, 1)

        g = gl.GLGridItem()
        g.setSize(220, 220)
        g.setSpacing(8, 8)
        g.translate(0, 0, cfg.floor_z - 2.0)
        self.view.addItem(g)

        # Wireframe grid (rows + sparse columns) driven by spectrum history.
        # Depth represents "time/history" so the mesh scrolls away from the camera.
        self.ny = max(8, int(cfg.plane_depth_cells))
        self.x = np.linspace(-cfg.plane_width * 0.5, cfg.plane_width * 0.5, cfg.n_points, dtype=np.float32)
        # Use positive depth so it visually recedes "forward".
        self.y = np.linspace(0.0, float(cfg.plane_depth), self.ny, dtype=np.float32)
        self.z_hist = np.full((cfg.n_points, self.ny), cfg.floor_z, dtype=np.float32)

        t = np.linspace(0.0, 1.0, cfg.n_points, dtype=np.float32)
        self.base_colors = _freq_colormap_rgba(t)  # (n_points, 4)

        # Precompute depth alpha falloff (near = bright, far = dim).
        d = np.linspace(0.0, 1.0, self.ny, dtype=np.float32)
        self.depth_alpha = (1.0 - d) ** 1.75

        self.row_lines: list[gl.GLLinePlotItem] = []
        self.col_lines: list[gl.GLLinePlotItem] = []

        # Axis legends: X = frequency (Hz); vertical scale = normalized amplitude (mapped to Z height).
        self._axis_label_color = QtGui.QColor(220, 225, 235)
        self._axis_tick_font = QtGui.QFont("Segoe UI", 10)
        self._axis_title_font = QtGui.QFont("Segoe UI", 11, QtGui.QFont.Weight.DemiBold)
        self._build_frequency_legend(cfg)
        self._build_amplitude_legend(cfg)

        # Rows (constant y): one line per depth slice.
        for j in range(self.ny):
            pos = np.column_stack([self.x, np.full_like(self.x, self.y[j]), self.z_hist[:, j]]).astype(np.float32)
            col = self.base_colors.copy()
            col[:, 3] = 0.95 * float(self.depth_alpha[j])
            item = gl.GLLinePlotItem(pos=pos, color=col, width=1.0, antialias=True, mode="line_strip")
            item.setGLOptions("additive")
            self.view.addItem(item)
            self.row_lines.append(item)

        # Columns (constant x): fewer lines for performance, still reads as a grid.
        col_step = max(4, int(round(cfg.n_points / 48)))
        self.col_indices = list(range(0, cfg.n_points, col_step))
        for i in self.col_indices:
            pos = np.column_stack(
                [
                    np.full(self.ny, self.x[i], dtype=np.float32),
                    self.y.astype(np.float32),
                    self.z_hist[i, :].astype(np.float32),
                ]
            ).astype(np.float32)
            col = np.repeat(self.base_colors[i : i + 1, :], self.ny, axis=0).astype(np.float32)
            col[:, 3] = 0.60 * self.depth_alpha.astype(np.float32)
            item = gl.GLLinePlotItem(pos=pos, color=col, width=1.0, antialias=True, mode="line_strip")
            item.setGLOptions("additive")
            self.view.addItem(item)
            self.col_lines.append(item)

        # status text
        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet("padding: 6px; font-family: Consolas, monospace;")
        self.info.setText("Starting…")
        layout.addWidget(self.info, 0)

        self.setCentralWidget(central)

        # Timer
        self._last_t = time.time()
        self._frames = 0
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)

    def _build_frequency_legend(self, cfg: VizConfig) -> None:
        """X-axis: frequency bands (Hz) at log-spaced tick positions along the spectrum edge."""
        self._freq_title = gl.GLTextItem(
            pos=(0.0, 0.0, cfg.floor_z - 8.0),
            text="Frequency (Hz) — low ← → high",
            color=self._axis_label_color,
            font=self._axis_title_font,
            alignment=QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
            glOptions="additive",
        )
        self.view.addItem(self._freq_title)

        tick_hz = [40.0, 100.0, 250.0, 630.0, 1600.0, 4000.0, 10000.0, 16000.0]
        tick_hz = [f for f in tick_hz if cfg.f_min * 0.99 <= f <= cfg.f_max * 1.01]
        self._freq_tick_items: list[gl.GLTextItem] = []
        for f in tick_hz:
            xw = _x_for_frequency(f, cfg.f_min, cfg.f_max, cfg.plane_width)
            item = gl.GLTextItem(
                pos=(xw, 0.0, cfg.floor_z - 4.0),
                text=_format_hz(f),
                color=self._axis_label_color,
                font=self._axis_tick_font,
                alignment=QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
                glOptions="additive",
            )
            self.view.addItem(item)
            self._freq_tick_items.append(item)

    def _build_amplitude_legend(self, cfg: VizConfig) -> None:
        """
        Vertical scale for wave height: normalized amplitude 0..1 → Z = floor_z + height * a.
        (World Y is history/depth, not amplitude.)
        """
        x_left = -cfg.plane_width * 0.5 - 14.0
        self._amp_title = gl.GLTextItem(
            pos=(x_left, 0.0, cfg.floor_z + cfg.height * 0.52),
            text="Amplitude\n(norm. 0–1)",
            color=self._axis_label_color,
            font=self._axis_title_font,
            alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
            glOptions="additive",
        )
        self.view.addItem(self._amp_title)

        ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        self._amp_tick_items: list[gl.GLTextItem] = []
        for a in ticks:
            z = cfg.floor_z + cfg.height * a
            item = gl.GLTextItem(
                pos=(x_left, 0.0, z),
                text=f"{a:.2f}",
                color=self._axis_label_color,
                font=self._axis_tick_font,
                alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                glOptions="additive",
            )
            self.view.addItem(item)
            self._amp_tick_items.append(item)

    def start(self):
        self.cap.start()
        self.timer.start(max(1, int(1000 / self.cfg.fps)))

    def closeEvent(self, event):  # noqa: N802
        try:
            self.timer.stop()
            self.cap.close()
        finally:
            super().closeEvent(event)

    def tick(self):
        # pull a hop of new audio
        self.cap.read_hop()
        blk = self.cap.get_block()
        if blk is None:
            return

        # Windowed FFT
        x = blk * self.window
        spec = np.fft.rfft(x)
        mag = np.abs(spec).astype(np.float32)

        # Convert to a more usable scale: log magnitude, then normalize
        mag = np.log10(1e-6 + mag)
        mag -= mag.min()
        denom = mag.max() if mag.max() > 1e-9 else 1.0
        mag /= denom

        # Sample on a log-spaced frequency axis
        sampled = _interp_spectrum(self.freqs, mag, self.target_freqs).astype(np.float32)

        # Smooth
        s = float(self.cfg.smoothing)
        out = s * self.prev + (1.0 - s) * sampled
        self.prev = out

        cfg = self.cfg
        amps = np.clip(out.astype(np.float32), 0.0, 1.0)
        z_plane = cfg.floor_z + cfg.height * amps
        # Scroll history down the depth axis and insert newest slice at y[0].
        self.z_hist[:, 1:] = self.z_hist[:, :-1]
        self.z_hist[:, 0] = z_plane

        # Update row lines
        for j, item in enumerate(self.row_lines):
            pos = np.column_stack([self.x, np.full_like(self.x, self.y[j]), self.z_hist[:, j]]).astype(np.float32)
            item.setData(pos=pos)

        # Update column lines
        for idx, i in enumerate(self.col_indices):
            pos = np.column_stack(
                [
                    np.full(self.ny, self.x[i], dtype=np.float32),
                    self.y.astype(np.float32),
                    self.z_hist[i, :].astype(np.float32),
                ]
            ).astype(np.float32)
            self.col_lines[idx].setData(pos=pos)

        # FPS + peak readout
        self._frames += 1
        now = time.time()
        if now - self._last_t >= 0.5:
            fps = self._frames / (now - self._last_t)
            self._frames = 0
            self._last_t = now
            peak = float(out.max())
            self.info.setText(
                f"fps={fps:5.1f}  peak={peak:0.3f}  "
                f"sr={self.cfg.sample_rate}  n={self.cfg.block_size}  points={self.cfg.n_points}"
            )


def main():
    cfg = VizConfig()
    pg.setConfigOptions(antialias=True)

    # Qt 6: AA_UseOpenGLES and QT_OPENGL=angle are unsupported; use default desktop GL.
    app = QtWidgets.QApplication(sys.argv)
    win = SpectrumPlaneViz(cfg)
    win.resize(1100, 800)
    win.show()
    win.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

