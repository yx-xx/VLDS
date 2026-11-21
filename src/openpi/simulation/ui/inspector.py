"""Desktop selector and controls for interactive policy inspection."""

# ruff: noqa: RUF001 -- Chinese interface text uses Chinese punctuation.

import os
import pathlib
import queue
import threading

from openpi.simulation import catalog
from openpi.simulation.runtime.controller import InspectorController
from openpi.simulation.ui.dataset_browser import DatasetBrowser
from openpi.simulation.ui.tk_runtime import configure_fonts
from openpi.simulation.ui.tk_runtime import filedialog
from openpi.simulation.ui.tk_runtime import scrolledtext
from openpi.simulation.ui.tk_runtime import tk
from openpi.simulation.ui.tk_runtime import ttk


class InspectorWindow:
    def __init__(self, root: tk.Tk, controller: InspectorController, dataset_path=None):
        self.root = root
        self.controller = controller
        self.catalog = controller.catalog
        self.scene_ready = False
        self.model_ready = False
        self.paused = True
        self.done = False
        self.closing = False
        self.tasks = []
        self.datasets = list(self.catalog)
        self.suites = []
        self.models = []
        root.title("VLDS · 模型观察台")
        root.geometry("1200x920")
        root.minsize(1100, 860)
        root.configure(bg="#f4f6f5")
        body_font = configure_fonts(root)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f6f5")
        style.configure("TLabel", background="#f4f6f5", foreground="#23322e", font=(body_font, 10))
        style.configure("TCheckbutton", background="#f4f6f5", font=(body_font, 10))
        style.configure("TButton", font=(body_font, 10), padding=(10, 7))
        style.configure("TCombobox", font=(body_font, 10), padding=4)
        style.configure("TSpinbox", font=(body_font, 10), padding=4)
        style.configure("Pages.TNotebook", borderwidth=0, background="#f4f6f5")
        style.layout("Pages.TNotebook.Tab", [])
        style.configure(
            "Mode.Toolbutton",
            font=(body_font, 10),
            padding=(18, 7),
            background="#e5ebe7",
            foreground="#607069",
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "Mode.Toolbutton",
            background=[("selected", "#ffffff"), ("active", "#eaf0ec")],
            foreground=[("selected", "#17664f")],
        )
        style.configure(
            "ActiveMode.Toolbutton",
            font=(body_font, 13, "bold"),
            padding=(24, 12),
            background="#ffffff",
            foreground="#17664f",
            relief="flat",
            borderwidth=0,
        )
        style.map("ActiveMode.Toolbutton", background=[("active", "#ffffff")])
        style.configure("Treeview", font=(body_font, 10), rowheight=30, background="#ffffff", fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=(body_font, 9, "bold"), padding=6)
        style.map("Treeview", background=[("selected", "#dceee4")], foreground=[("selected", "#164e3d")])
        style.configure("TEntry", font=(body_font, 10), padding=4)
        style.configure("Accent.TButton", background="#20765f", foreground="white")
        style.map("Accent.TButton", background=[("active", "#175d4a")])
        style.configure("Heading.TLabel", font=(body_font, 20, "bold"))
        style.configure("Subheading.TLabel", font=(body_font, 12, "bold"))
        style.configure("Muted.TLabel", foreground="#66756f")
        style.configure("Status.TLabel", foreground="#20765f", font=(body_font, 12, "bold"))
        style.configure("Error.TLabel", foreground="#ab3f32")

        outer = ttk.Frame(root, padding=24)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="模型观察台", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(
            outer, text="观察模型现场行为，或浏览数据集的真实演示。每次只查看你选中的场景与回合。", style="Muted.TLabel"
        ).pack(anchor="w", pady=(4, 12))
        navigation = ttk.Frame(outer)
        navigation.pack(fill="x", pady=(0, 8))
        self.active_page = tk.IntVar(value=0)
        self.mode_switches = []
        for page, title in enumerate(("模型仿真验证", "数据集浏览")):
            button = ttk.Radiobutton(
                navigation,
                text=title,
                value=page,
                variable=self.active_page,
                command=lambda selected=page: self.tabs.select(selected),
                style="ActiveMode.Toolbutton" if page == 0 else "Mode.Toolbutton",
            )
            button.pack(side="left", anchor="s", padx=(0, 8))
            self.mode_switches.append(button)
        self.tabs = ttk.Notebook(outer, style="Pages.TNotebook")
        self.tabs.pack(fill="both", expand=True)
        simulation_page = ttk.Frame(self.tabs, padding=(0, 12, 0, 0))
        self.tabs.add(simulation_page, text="模型仿真验证")
        self.dataset_browser = DatasetBrowser(self.tabs)
        self.tabs.add(self.dataset_browser, text="数据集浏览")
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)
        columns = ttk.Frame(simulation_page)
        columns.pack(fill="both", expand=True)
        left = ttk.Frame(columns)
        left.pack(side="left", fill="both", expand=True, padx=(0, 28))
        right = ttk.Frame(columns, width=350)
        right.pack(side="right", fill="both")
        right.pack_propagate(flag=False)

        self.dataset = tk.StringVar()
        self.suite = tk.StringVar()
        self.kind = tk.StringVar(value="全部操作")
        self.task = tk.StringVar()
        self.model = tk.StringVar()
        self.checkpoint = tk.StringVar()
        self.seed = tk.StringVar(value="7")
        self.initial = tk.StringVar(value="0")
        self.software = tk.BooleanVar(value=True)
        self.auto_model = tk.BooleanVar(value=True)
        self.dataset_box = self._combo(left, "仿真环境", self.dataset)
        self.dataset_box["values"] = [value["name"] for value in self.catalog.values()]
        self.dataset_box.bind("<<ComboboxSelected>>", self._dataset_changed)
        row = ttk.Frame(left)
        row.pack(fill="x", pady=(10, 0))
        self.suite_box = self._combo(row, "任务集合", self.suite, side="left")
        self.suite_box.bind("<<ComboboxSelected>>", self._suite_changed)
        self.kind_box = self._combo(row, "操作类型", self.kind, side="right")
        self.kind_box.bind("<<ComboboxSelected>>", self._filter_tasks)
        self.task_box = self._combo(left, "具体任务", self.task, pady=10)
        self.task_box.bind("<<ComboboxSelected>>", self._task_changed)
        self.description = ttk.Label(left, text="", wraplength=570, style="Muted.TLabel")
        self.description.pack(anchor="w", fill="x", pady=(0, 12))
        numeric = ttk.Frame(left)
        numeric.pack(fill="x")
        self.initial_spin = self._number(numeric, "初始状态编号", self.initial, 1000000)
        self._number(numeric, "随机种子", self.seed, 2**32 - 1)
        self.initial_note = ttk.Label(left, text="", style="Muted.TLabel", wraplength=570)
        self.initial_note.pack(anchor="w", pady=(5, 10))
        self.model_box = self._combo(left, "模型配置", self.model)
        self.model_box.bind("<<ComboboxSelected>>", self._model_changed)
        ttk.Label(left, text="自定义检查点（留空使用对应环境的默认权重）").pack(anchor="w", pady=(10, 4))
        path_row = ttk.Frame(left)
        path_row.pack(fill="x")
        ttk.Entry(path_row, textvariable=self.checkpoint).pack(side="left", fill="x", expand=True)
        ttk.Button(path_row, text="选择目录", command=self._browse).pack(side="right", padx=(6, 0))
        self.policy_note = ttk.Label(left, text="", style="Muted.TLabel", wraplength=570)
        self.policy_note.pack(anchor="w", pady=(5, 10))
        ttk.Checkbutton(left, text="使用软件绘制三维界面（模型仍使用 GPU）", variable=self.software).pack(anchor="w")
        ttk.Checkbutton(left, text="打开场景时同时加载模型", variable=self.auto_model).pack(anchor="w", pady=(5, 10))
        actions = ttk.Frame(left)
        actions.pack(fill="x")
        self.open_button = ttk.Button(actions, text="打开所选场景", style="Accent.TButton", command=self.open_scene)
        self.open_button.pack(side="left", fill="x", expand=True)
        self.load_button = ttk.Button(actions, text="加载 / 切换模型", command=self.load_model)
        self.load_button.pack(side="left", padx=(8, 0))

        ttk.Label(right, text="当前场景", style="Subheading.TLabel").pack(anchor="w")
        self.status = ttk.Label(right, text="先选择场景", wraplength=345, style="Status.TLabel")
        self.status.pack(anchor="w", pady=(10, 8))
        self.instruction = ttk.Label(right, text="完整 MuJoCo 三维窗口会单独打开。", wraplength=345)
        self.instruction.pack(anchor="w", pady=(0, 12))
        self.model_status = ttk.Label(right, text="模型尚未加载", wraplength=345, style="Muted.TLabel")
        self.model_status.pack(anchor="w", pady=(0, 10))
        self.step_label = ttk.Label(right, text="控制步 0 · 本次推理 —", style="Muted.TLabel")
        self.step_label.pack(anchor="w", pady=(0, 12))
        controls = ttk.Frame(right)
        controls.pack(fill="x")
        self.play_button = ttk.Button(controls, text="开始 / 继续", style="Accent.TButton", command=self.toggle_play)
        self.play_button.pack(side="left", expand=True, fill="x")
        self.step_button = ttk.Button(controls, text="单步", command=lambda: self.command("step"))
        self.step_button.pack(side="left", padx=(8, 0))
        self.reset_button = ttk.Button(right, text="重置为当前初始状态", command=self.reset)
        self.reset_button.pack(fill="x", pady=(8, 0))
        self.next_initial_button = ttk.Button(right, text="换一个初始状态", command=self.next_initial)
        self.next_initial_button.pack(fill="x", pady=(8, 0))
        self.next_task_button = ttk.Button(right, text="打开下一个任务", command=self.next_task)
        self.next_task_button.pack(fill="x", pady=(8, 0))
        self.stop_button = ttk.Button(right, text="关闭当前场景", command=controller.stop_scene)
        self.stop_button.pack(fill="x", pady=(8, 18))
        ttk.Label(right, text="三维窗口操作", style="Subheading.TLabel").pack(anchor="w")
        ttk.Label(
            right,
            text="左键旋转 · 右键平移 · 滚轮缩放\n空格暂停 / 继续 · → 单步\n关节、接触点和坐标系可在侧栏查看。",
            wraplength=345,
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(6, 16))
        ttk.Label(
            right,
            text="每次只运行所选任务；手动重置或切换。\n这里的初始状态用于重现场景，模型会现场生成动作。",
            wraplength=345,
            style="Muted.TLabel",
        ).pack(anchor="w")

        ttk.Label(simulation_page, text="运行日志", style="Muted.TLabel").pack(anchor="w", pady=(14, 4))
        self.log = scrolledtext.ScrolledText(
            simulation_page,
            height=5,
            font="TkFixedFont",
            bg="#ffffff",
            fg="#46534e",
            relief="flat",
            state="disabled",
        )
        self.log.pack(fill="x")
        self.dataset_box.current(0)
        self._dataset_changed()
        self._update_controls()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)
        if dataset_path:
            self.tabs.select(self.dataset_browser)
            self.dataset_browser.load(dataset_path)

    def _tab_changed(self, _event=None):
        dataset_selected = self.tabs.select() == str(self.dataset_browser)
        self.active_page.set(1 if dataset_selected else 0)
        for page, button in enumerate(self.mode_switches):
            button.configure(style="ActiveMode.Toolbutton" if page == self.active_page.get() else "Mode.Toolbutton")
        if dataset_selected:
            if self.scene_ready and not self.paused:
                self.command("pause")
        else:
            self.dataset_browser.pause()

    @staticmethod
    def _combo(parent, label, variable, *, side=None, pady=0):
        frame = ttk.Frame(parent)
        frame.pack(
            side=side or "top", fill="x", expand=side is not None, pady=(pady, 0), padx=(0, 8) if side == "left" else 0
        )
        ttk.Label(frame, text=label).pack(anchor="w", pady=(0, 4))
        box = ttk.Combobox(frame, textvariable=variable, state="readonly", width=20 if side else 50)
        box.pack(fill="x")
        return box

    @staticmethod
    def _number(parent, label, variable, maximum):
        frame = ttk.Frame(parent)
        frame.pack(side="left", expand=True, fill="x", padx=(0, 14))
        ttk.Label(frame, text=label).pack(anchor="w", pady=(0, 4))
        spin = ttk.Spinbox(frame, textvariable=variable, from_=0, to=maximum, width=12)
        spin.pack(fill="x")
        return spin

    def dataset_id(self):
        return self.datasets[self.dataset_box.current()]

    def _dataset_changed(self, _event=None):
        data = self.catalog[self.dataset_id()]
        self.suites = data["suites"]
        self.suite_box["values"] = [s["name"] for s in self.suites]
        self.models = list(catalog.POLICIES[self.dataset_id()])
        self.model_box["values"] = [catalog.POLICIES[self.dataset_id()][model][0] for model in self.models]
        self.model_box.current(0)
        self.checkpoint.set("")
        self.initial.set("0")
        self.initial_spin.configure(to=1000000)
        self.initial_note.configure(
            text="官方初始状态编号将在场景打开后确认。"
            if self.dataset_id() == "libero"
            else "ALOHA 的编号作为随机种子偏移量，用于重现场景。"
        )
        self.open_button.configure(state="normal" if data["available"] else "disabled")
        if self.suites:
            self.suite_box.current(0)
            self._suite_changed()
        else:
            self.tasks = []
            self.task_box.set("")
        self._model_changed()

    def _suite_changed(self, _event=None):
        suite = self.suites[self.suite_box.current()]
        self.kind_box["values"] = ["全部操作", *sorted({t["kind"] for t in suite["tasks"]})]
        self.kind_box.current(0)
        self._filter_tasks()

    def _filter_tasks(self, _event=None):
        suite = self.suites[self.suite_box.current()]
        self.tasks = [t for t in suite["tasks"] if self.kind.get() in ("全部操作", t["kind"])]
        self.task_box["values"] = [f"{t['id']:02d} · {t['name']}" for t in self.tasks]
        if self.tasks:
            self.task_box.current(0)
            self._task_changed()

    def _task_changed(self, _event=None):
        self.initial.set("0")
        self.description.configure(text=self.tasks[self.task_box.current()]["name"] if self.tasks else "环境尚未安装。")

    def _model_changed(self, _event=None):
        config = catalog.POLICIES[self.dataset_id()][self.models[self.model_box.current()]][1]
        cache_root = pathlib.Path(os.environ.get("OPENPI_DATA_HOME", str(pathlib.Path.home() / ".cache/openpi")))
        cached = (cache_root / "openpi-assets/checkpoints" / config / "params").is_dir()
        hint = "已发现本地模型缓存。" if cached else "默认权重未缓存；点击加载模型时将下载。"
        self.auto_model.set(cached)
        self.policy_note.configure(text=self.catalog[self.dataset_id()]["note"] + "\n" + hint)

    def _browse(self):
        path = filedialog.askdirectory(title="选择模型检查点目录")
        if path:
            self.checkpoint.set(path)

    def selection(self):
        return {
            "dataset": self.dataset_id(),
            "suite": self.suites[self.suite_box.current()]["id"],
            "task_id": self.tasks[self.task_box.current()]["id"],
            "initial_state": int(self.initial.get()),
            "seed": int(self.seed.get()),
            "model": self.models[self.model_box.current()],
            "checkpoint": self.checkpoint.get(),
            "software_rendering": self.software.get(),
        }

    def open_scene(self):
        try:
            self.controller.open_scene(self.selection(), load_model=self.auto_model.get())
            self.scene_ready = self.model_ready = self.done = False
            self._update_controls()
        except (ValueError, IndexError) as error:
            self.show_error(str(error))

    def load_model(self):
        selected = self.controller.selection
        if selected and selected["dataset"] != self.dataset_id():
            self.show_error("请先打开左侧所选环境，再加载对应模型。")
            return
        self.controller.load_policy(self.models[self.model_box.current()], self.checkpoint.get())

    def command(self, command, **arguments):
        try:
            self.controller.send(command, **arguments)
        except ValueError as error:
            self.show_error(str(error))

    def toggle_play(self):
        self.command("play" if self.paused else "pause")

    def reset(self):
        try:
            self.command("reset", initial_state=int(self.initial.get()))
        except ValueError:
            self.show_error("初始状态编号须为整数。")

    def next_initial(self):
        try:
            next_index = int(self.initial.get()) + 1
            maximum = int(float(self.initial_spin.cget("to")))
            self.initial.set(str(next_index if next_index <= maximum else 0))
            self.reset()
        except ValueError:
            self.show_error("初始状态编号须为整数。")

    def next_task(self):
        if self.tasks:
            self.task_box.current((self.task_box.current() + 1) % len(self.tasks))
            self._task_changed()
            self.open_scene()

    def show_error(self, message):
        self.status.configure(text=message, style="Error.TLabel")
        self.append_log(message)

    def append_log(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 150:
            self.log.delete("1.0", "30.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _update_controls(self):
        active = self.scene_ready and self.model_ready and not self.done and not self.closing
        for button in (self.play_button, self.step_button):
            button.configure(state="normal" if active else "disabled")
        self.play_button.configure(text="开始 / 继续" if self.paused else "暂停")
        for button in (self.reset_button, self.next_initial_button, self.stop_button, self.load_button):
            button.configure(state="normal" if self.scene_ready and not self.closing else "disabled")
        self.next_task_button.configure(state="normal" if self.tasks and not self.closing else "disabled")

    def poll(self):
        try:
            while True:
                event = self.controller.events.get_nowait()
                if event.get("closed"):
                    self.root.destroy()
                    return
                if "session" in event and event["session"] != self.controller.session:
                    continue
                for key in ("scene_ready", "model_ready", "paused", "done"):
                    if key in event:
                        setattr(self, key, event[key])
                if "status" in event:
                    self.status.configure(text=event["status"], style="Status.TLabel")
                if "model_status" in event:
                    self.model_status.configure(text=event["model_status"])
                if "instruction" in event:
                    self.instruction.configure(text=event["instruction"])
                if "initial_count" in event and event["initial_count"] is not None:
                    self.initial_spin.configure(to=event["initial_count"] - 1)
                    self.initial_note.configure(
                        text=f"本任务有 {event['initial_count']} 个官方初始状态；重置可重现同一场景。"
                    )
                if "initial_state" in event:
                    self.initial.set(str(event["initial_state"]))
                if "step" in event:
                    latency = event.get("latency_ms")
                    self.step_label.configure(
                        text=f"控制步 {event['step']} · 本次推理 {str(latency) + ' ms' if latency is not None else '—'}"
                    )
                if "log" in event:
                    self.append_log(event["log"])
                if "error" in event:
                    self.show_error(event["error"])
        except queue.Empty:
            pass
        self._update_controls()
        self.root.after(100, self.poll)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.status.configure(text="正在关闭场景和模型服务…")
        self._update_controls()

        self.dataset_browser.close()

        def shutdown():
            self.controller.close()
            self.controller.publish(closed=True)

        threading.Thread(target=shutdown, daemon=True).start()


def main(dataset_path=None, *, replay=False, watch=False):
    root = tk.Tk()
    controller = InspectorController()
    try:
        window = InspectorWindow(root, controller, dataset_path)
        window.dataset_browser.auto_replay.set(replay)
        window.dataset_browser.auto_refresh.set(watch)
        root.mainloop()
    finally:
        if "window" in locals():
            window.dataset_browser.close()
        controller.close()
