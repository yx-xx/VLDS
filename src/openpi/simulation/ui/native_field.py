"""Native-coordinate field projection for persisted demonstrations of any robot."""

# ruff: noqa: RUF001 -- Chinese interface text.
import numpy as np

from openpi.datasets.libero import SEGMENT_COLORS
from openpi.simulation.ui.tk_runtime import tk
from openpi.simulation.ui.tk_runtime import ttk


class NativeFieldCanvas(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.bundle, self.frame, self.index = None, 0, None
        self.visible = self.vectors = self.rollout = True
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="原状态空间投影").pack(side="left")
        self.x = ttk.Combobox(row, state="readonly", width=17)
        self.y = ttk.Combobox(row, state="readonly", width=17)
        for combo in (self.x, self.y):
            combo.pack(side="left", padx=5)
            combo.bind("<<ComboboxSelected>>", lambda _event: self.draw())
        self.caption = ttk.Label(row, style="Muted.TLabel")
        self.caption.pack(side="left", padx=5)
        self.canvas = tk.Canvas(self, height=155, bg="#f4f6f5", highlightthickness=0)
        self.canvas.pack(fill="x", pady=3)
        self.canvas.bind("<Configure>", lambda _event: self.draw())

    def set_bundle(self, bundle):
        self.bundle = bundle
        self.index = None
        if not bundle:
            self.pack_forget()
            return
        axes = bundle["result"]["coordinates"]["axes"]
        self.x["values"], self.y["values"] = axes, axes
        self.x.current(0)
        self.y.current(1)
        self.caption.configure(text=bundle["result"]["coordinates"]["units"])
        self.draw()

    def configure_field(self, *, visible, vectors, rollout):
        self.visible, self.vectors, self.rollout = visible, vectors, rollout
        self.draw()

    def update_frame(self, frame, index):
        self.frame, self.index = frame, index
        self.draw()

    def draw(self):
        c = self.canvas
        c.delete("all")
        if self.bundle is None or self.index is None or not self.visible:
            return
        result, arrays = self.bundle["result"], self.bundle["arrays"]
        record = result["records"][self.index]
        dimensions = [self.x.current(), self.y.current()]
        if dimensions[0] == dimensions[1]:
            self.y.current((dimensions[0] + 1) % arrays["position"].shape[1])
            dimensions[1] = self.y.current()
        demo = arrays["position"][:, dimensions]
        nominal = arrays["nominal"][:, dimensions]
        a, b = record["start"], record["end"]
        goal = np.asarray(record["goal"])[dimensions]
        extent = np.ptp(demo, axis=0)
        lo, hi = (
            demo.min(axis=0) - np.maximum(extent * 0.08, 0.001),
            demo.max(axis=0) + np.maximum(extent * 0.08, 0.001),
        )
        width, height = max(c.winfo_width(), 100), max(c.winfo_height(), 60)
        scale = np.array([width - 30, height - 20]) / (hi - lo)

        def screen(points):
            values = (np.asarray(points) - lo) * scale + [15, 10]
            values[..., 1] = height - values[..., 1]
            return values

        def line(points, **style):
            if len(points) > 1:
                c.create_line(*screen(points).ravel(), **style)

        from openpi.datasets.dynamics.playback import native_plane_velocity

        if self.vectors:
            plane = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 13), np.linspace(lo[1], hi[1], 5)), axis=-1).reshape(
                -1, 2
            )
            velocity = native_plane_velocity(self.bundle, self.index, dimensions, plane, self.frame)
            for point, vector in zip(plane, velocity, strict=True):
                start = screen(point)
                projected = vector * scale * [1, -1]
                norm = np.linalg.norm(projected)
                if norm > 1e-8:
                    end = start + projected / norm * 12
                    c.create_line(*start, *end, fill="#93aaa1", arrow="last")
        for i, segment in enumerate(result["records"]):
            line(demo[segment["start"] : segment["end"] + 1], fill=SEGMENT_COLORS[i % len(SEGMENT_COLORS)], width=1)
        line(demo[a : b + 1], fill="#df7c1c", width=3)
        if self.rollout:
            line(nominal[a : b + 1], fill="#a638c2", width=2, dash=(5, 2))
        for point, color, radius in ((goal, "#1b9853", 5), (demo[min(self.frame, len(demo) - 1)], "#111111", 4)):
            x, y = screen(point)
            c.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="white")
        model = "高斯 LPV 场" if "field" in record else "线性场"
        c.create_text(
            8,
            9,
            anchor="nw",
            fill="#243b31",
            text=f"场 {self.index + 1}/{len(result['records'])} · {model} · 其余轴固定当前源状态 · 橙：示教 紫：连续积分 绿：目标",
        )
