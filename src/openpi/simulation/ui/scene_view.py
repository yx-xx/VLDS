"""MuJoCo viewport inside the inspector; all Tk drawing stays on the UI thread."""

# ruff: noqa: RUF001 -- Chinese interface text.
from PIL import Image

from openpi.simulation.ui.tk_runtime import tk
from openpi.simulation.ui.tk_runtime import ttk


class SceneView(ttk.Frame):
    def __init__(self, parent, *, send, toggle_play, step):
        super().__init__(parent)
        self.send = send
        self.image = self.photo = None
        self.drag = None
        self.resize_after = None
        self.message = "选择回合或打开场景后，在这里查看三维画面。"
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", pady=(0, 4))
        ttk.Label(toolbar, text="MuJoCo 三维视图", style="Subheading.TLabel").pack(side="left")
        ttk.Button(toolbar, text="复位视角", command=lambda: self._send(action="reset")).pack(side="right")
        self.canvas = tk.Canvas(self, bg="#17221f", highlightthickness=0, width=420, height=280, takefocus=True)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._resized)
        for button in (1, 2, 3):
            self.canvas.bind(f"<ButtonPress-{button}>", self._press)
            self.canvas.bind(f"<B{button}-Motion>", self._motion)
        self.canvas.bind("<MouseWheel>", lambda event: self._zoom(-event.delta / 120))
        self.canvas.bind("<Button-4>", lambda _event: self._zoom(-1))
        self.canvas.bind("<Button-5>", lambda _event: self._zoom(1))
        self.canvas.bind("<space>", lambda _event: self._key(toggle_play))
        self.canvas.bind("<Right>", lambda _event: self._key(step))
        ttk.Label(self, text="左键旋转 · 右键平移 · 滚轮缩放 · 空格播放/暂停 · → 单步", style="Muted.TLabel").pack(
            anchor="w", pady=(4, 0)
        )
        self._draw()

    def _send(self, **values):
        self.send("view", **values)

    @staticmethod
    def _key(callback):
        callback()
        return "break"

    def _press(self, event):
        self.canvas.focus_set()
        self.drag = (event.x, event.y, "rotate" if event.num == 1 else "pan")

    def _motion(self, event):
        if self.drag is not None:
            x, y, action = self.drag
            height = max(1, self.canvas.winfo_height())
            self._send(action=action, dx=(event.x - x) / height, dy=(event.y - y) / height)
            self.drag = (event.x, event.y, action)

    def _zoom(self, amount):
        self._send(action="zoom", dy=amount * 0.05)
        return "break"

    def _resized(self, _event):
        self._draw()
        if self.resize_after is not None:
            self.after_cancel(self.resize_after)
        self.resize_after = self.after(180, self.resize_view)

    def resize_view(self):
        self.resize_after = None
        self._send(action="resize", width=self.canvas.winfo_width(), height=self.canvas.winfo_height())

    def clear(self, message=None):
        self.image = self.photo = None
        if message is not None:
            self.message = message
        self._draw()

    def update_frame(self, image):
        if image is not None:
            self.image = Image.fromarray(image)
            self._draw()

    def _draw(self):
        width, height = max(20, self.canvas.winfo_width()), max(20, self.canvas.winfo_height())
        self.canvas.delete("all")
        if self.image is None:
            self.canvas.create_text(
                max(1, width) / 2,
                max(1, height) / 2,
                text=self.message,
                width=max(100, width - 40),
                fill="#c7d6d0",
                justify="center",
            )
            return
        picture = self.image.copy()
        picture.thumbnail((width, height), Image.Resampling.BILINEAR)
        # Select the system Tk backend before importing ImageTk.
        from PIL import ImageTk

        self.photo = ImageTk.PhotoImage(picture, master=self.canvas)
        self.canvas.create_image(width / 2, height / 2, image=self.photo)
