"""Command-interval navigation synchronized with the demonstration timeline."""

# ruff: noqa: RUF001 -- Chinese interface text.
from openpi.datasets.libero import SEGMENT_COLORS
from openpi.simulation.ui.tk_runtime import tk
from openpi.simulation.ui.tk_runtime import ttk


class SegmentationPanel(ttk.Frame):
    def __init__(self, parent, *, seek, play, configure_overlay):
        super().__init__(parent)
        self.seek = seek
        self.play = play
        self.configure_overlay = configure_overlay
        self.data = None
        self.frame = 0
        self.selected = 0
        options = ttk.Frame(self)
        options.pack(fill="x", pady=(0, 3))
        ttk.Label(options, text="命令分段", style="Subheading.TLabel").pack(side="left", padx=(0, 16))
        self.timeline = tk.Canvas(self, height=34, highlightthickness=0, bg="#f4f6f5")
        self.timeline.pack(fill="x", pady=(3, 2))
        self.timeline.bind("<Configure>", lambda _event: self._draw())
        self.timeline.bind("<Button-1>", self._click)
        row = ttk.Frame(self)
        row.pack(fill="x")
        self.selection = tk.StringVar()
        self.combo = ttk.Combobox(row, textvariable=self.selection, state="readonly", width=36)
        self.combo.pack(side="left", fill="x", expand=True)
        self.combo.bind("<<ComboboxSelected>>", self._choose)
        ttk.Button(row, text="上一段", command=lambda: self._move(-1)).pack(side="left", padx=(6, 0))
        ttk.Button(row, text="下一段", command=lambda: self._move(1)).pack(side="left", padx=4)
        ttk.Button(row, text="播放此段", command=self._play).pack(side="left")
        self.detail = ttk.Label(self, text="", style="Muted.TLabel", wraplength=850)
        self.detail.pack(anchor="w", pady=(3, 0))
        self.overlay_visible = tk.BooleanVar(value=True)
        self.overlay_current_only = tk.BooleanVar(value=False)
        self.overlay_markers = tk.BooleanVar(value=True)
        for label, variable in (
            ("三维轨迹", self.overlay_visible),
            ("只看当前段", self.overlay_current_only),
            ("分段端点", self.overlay_markers),
        ):
            ttk.Checkbutton(options, text=label, variable=variable, command=self._overlay_changed).pack(
                side="left", padx=(0, 12)
            )
        ttk.Label(options, text="当前段加粗 · 白点定位", style="Muted.TLabel").pack(side="left")

    def _overlay_changed(self):
        self.configure_overlay(
            visible=self.overlay_visible.get(),
            current_only=self.overlay_current_only.get(),
            markers=self.overlay_markers.get(),
        )

    def set_data(self, data):
        self.data = data
        self.selected = 0
        if not data:
            self.pack_forget()
            return
        intervals, times = data["intervals"], data["time"]
        self.combo["values"] = [
            f"第 {index + 1} 段 · {item['command_label']} · {times[item['start']]:.2f}–{times[item['end']]:.2f} s"
            for index, item in enumerate(intervals)
        ]
        self.update_frame(0)

    def update_frame(self, frame):
        self.frame = frame
        if not self.data:
            return
        intervals = self.data["intervals"]
        self.selected = next(
            (i for i, item in enumerate(intervals) if item["start"] <= frame < item["end"]), len(intervals) - 1
        )
        self.combo.current(self.selected)
        item = intervals[self.selected]
        terminal = " · 当前为终端观测，无下一动作" if frame == len(self.data["time"]) - 1 else ""
        self.detail.configure(
            text=f"动作 [{item['start']}, {item['end']}) · 观测端点 [{item['start']}, {item['end']}] · "
            f"{self.data['candidate_counts'][self.selected]} 个拟合候选；含区间内全部运动与等待{terminal}"
        )
        self._draw()

    def _draw(self):
        self.timeline.delete("all")
        if not self.data:
            return
        width = max(self.timeline.winfo_width(), 1)
        times = self.data["time"]
        duration = times[-1] - times[0]
        if duration <= 0:
            return
        for index, item in enumerate(self.data["intervals"]):
            left = (times[item["start"]] - times[0]) / duration * width
            right = (times[item["end"]] - times[0]) / duration * width
            color = SEGMENT_COLORS[index % len(SEGMENT_COLORS)]
            self.timeline.create_rectangle(left, 3, right, 28, fill=color, outline="#f4f6f5")
            if right - left > 75:
                self.timeline.create_text(
                    (left + right) / 2, 15, text=f"{index + 1} {item['command_label']}", fill="white"
                )
        x = (times[min(self.frame, len(times) - 1)] - times[0]) / duration * (width - 1)
        self.timeline.create_line(x, 0, x, 34, fill="#17221f", width=2)
        self.timeline.create_polygon(x - 4, 0, x + 4, 0, x, 6, fill="#17221f")

    def _choose(self, _event=None):
        self.selected = self.combo.current()
        self.seek(self.data["intervals"][self.selected]["start"])

    def _move(self, direction):
        if self.data:
            self.selected = max(0, min(len(self.data["intervals"]) - 1, self.selected + direction))
            self.seek(self.data["intervals"][self.selected]["start"])

    def _click(self, event):
        if not self.data:
            return
        times = self.data["time"]
        timestamp = times[0] + max(0, min(1, event.x / max(self.timeline.winfo_width(), 1))) * (times[-1] - times[0])
        index = next(
            (i for i, item in enumerate(self.data["intervals"]) if timestamp < times[item["end"]]),
            len(self.data["intervals"]) - 1,
        )
        self.seek(self.data["intervals"][index]["start"])

    def _play(self):
        if self.data:
            item = self.data["intervals"][self.selected]
            self.play(item["start"], item["end"])
