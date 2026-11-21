"""Dataset demonstration browser embedded in the model inspection window."""

# ruff: noqa: RUF001 -- Chinese interface text.
import concurrent.futures
import os
import pathlib
import queue
import threading
import time

import numpy as np
from PIL import Image

from openpi.simulation import paths
from openpi.simulation.data.descriptions import dataset_description
from openpi.simulation.data.descriptions import dataset_source
from openpi.simulation.data.descriptions import task_label
from openpi.simulation.data.discovery import discover_datasets
from openpi.simulation.data.download import ENDPOINTS
from openpi.simulation.data.download import PREVIEW_ROOT
from openpi.simulation.data.download import SOURCES
from openpi.simulation.data.download import download_preview
from openpi.simulation.data.reader import EpisodeReader
from openpi.simulation.data.reader import scan_dataset
from openpi.simulation.data.watch import StableFileWatch
from openpi.simulation.data.watch import file_signature
from openpi.simulation.data.watch import watch_paths
from openpi.simulation.runtime.replay import DatasetReplay
from openpi.simulation.ui.dataset_library import DatasetLibrary
from openpi.simulation.ui.segmentation import SegmentationPanel
from openpi.simulation.ui.tk_runtime import filedialog
from openpi.simulation.ui.tk_runtime import scrolledtext
from openpi.simulation.ui.tk_runtime import tk
from openpi.simulation.ui.tk_runtime import ttk


class DatasetBrowser(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=(0, 12, 0, 0))
        self.events = queue.Queue()
        self.io = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-reader")
        self.library_io = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-library")
        self.downloads = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dataset-download")
        self.replay = DatasetReplay()
        self.reader = None  # Used only on the IO thread.
        self.index = None
        self.episode = None
        self.filtered_episodes = []
        self.closed = False
        self.generation = 0
        self.render_pending = False
        self.requested = None
        self.frame = 0
        self.playing = False
        self.play_limit = None
        self.last_tick = 0.0
        self.image = None
        self.photo = None
        self.updating_scale = False
        self.replay_ready = False
        self.replay_pending = None
        self.replay_sent_at = 0.0
        self.replay_loading = False
        self.replay_after_read = False
        self.reload_restore = None
        self.reloading = False
        self.reload_retry = False
        self.last_reload = 0.0
        self.restoring = None
        self.watch = StableFileWatch()
        self.watch_pending = False
        self.last_watch = 0.0
        self.downloading = False
        self.download_cancelled = threading.Event()
        self.preferred_episode = None
        self.paths = []
        self.task_rows = {}
        self.preview_message = "选择任务，查看它的演示画面。"

        self.content = ttk.Frame(self)
        heading = ttk.Frame(self.content)
        heading.pack(fill="x", pady=(0, 7))
        ttk.Button(heading, text="← 数据集总览", command=self.show_library).pack(side="left", padx=(0, 16))
        self.overview_title = ttk.Label(heading, text="数据集内容", style="Subheading.TLabel")
        self.overview_title.pack(side="left")
        self.auto_refresh = tk.BooleanVar(value=False)
        ttk.Checkbutton(heading, text="自动检测文件更新", variable=self.auto_refresh).pack(side="right", padx=8)
        self.reload_button = ttk.Button(heading, text="重新载入当前数据", command=self.reload_current)
        self.reload_button.pack(side="right")
        self.overview_description = ttk.Label(self.content, text="", style="Muted.TLabel", wraplength=1080)
        self.overview_description.pack(anchor="w", pady=(0, 5))
        self.overview_stats = ttk.Label(self.content, text="", style="Status.TLabel")
        self.overview_stats.pack(anchor="w", pady=(0, 14))
        body = ttk.Frame(self.content)
        body.pack(fill="both", expand=True)
        sidebar = ttk.Frame(body, width=355)
        sidebar.pack(side="left", fill="y", padx=(0, 16))
        sidebar.pack_propagate(flag=False)
        self.sidebar_canvas = tk.Canvas(sidebar, bg="#f4f6f5", highlightthickness=0, width=335)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=self.sidebar_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        left = ttk.Frame(self.sidebar_canvas)
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind(
            "<Configure>", lambda _event: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))
        )
        self.sidebar_canvas.bind(
            "<Configure>", lambda event: self.sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width)
        )
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self.path = tk.StringVar()
        self.local = tk.StringVar()
        self.local_box = self.combo(left, "切换数据集", self.local)
        self.local_box.bind("<<ComboboxSelected>>", self._local_changed)
        self.summary = ttk.Label(left, text="正在读取任务目录…", wraplength=330, style="Muted.TLabel")
        self.summary.pack(anchor="w", pady=(6, 8))
        self.kind = tk.StringVar(value="全部操作")
        self.kind_box = self.combo(left, "操作类型", self.kind)
        self.kind_box.bind("<<ComboboxSelected>>", self._filter_tasks)
        self.task = tk.StringVar()
        self.search = tk.StringVar()
        ttk.Label(left, text="任务目录", style="Subheading.TLabel").pack(anchor="w", pady=(12, 5))
        search_row = ttk.Frame(left)
        search_row.pack(fill="x")
        ttk.Label(search_row, text="搜索").pack(side="left", padx=(0, 6))
        ttk.Entry(search_row, textvariable=self.search).pack(side="left", fill="x", expand=True)
        self.search.trace_add("write", lambda *_args: self._filter_tasks())
        self.only_local = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="只看本地可播放的任务", variable=self.only_local, command=self._filter_tasks).pack(
            anchor="w", pady=(5, 3)
        )
        task_list = ttk.Frame(left)
        task_list.pack(fill="x")
        self.task_tree = ttk.Treeview(
            task_list, columns=("available",), show="tree headings", height=6, selectmode="browse"
        )
        self.task_tree.heading("#0", text="任务（可搜索中英文）", anchor="w")
        self.task_tree.heading("available", text="本地 / 总回合")
        self.task_tree.column("#0", width=225, minwidth=100, stretch=True)
        self.task_tree.column("available", width=95, minwidth=80, stretch=False, anchor="center")
        self.task_tree.tag_configure("remote", foreground="#7c8781")
        self.task_tree.tag_configure("local", foreground="#17664f")
        task_scrollbar = ttk.Scrollbar(task_list, orient="vertical", command=self.task_tree.yview)
        task_scrollbar.pack(side="right", fill="y")
        self.task_tree.configure(yscrollcommand=task_scrollbar.set)
        self.task_tree.pack(side="left", fill="x", expand=True)
        self.task_tree.bind("<<TreeviewSelect>>", self._task_selected)
        self.task_count = ttk.Label(left, text="", style="Muted.TLabel")
        self.task_count.pack(anchor="w", pady=(4, 8))
        self.episode_name = tk.StringVar()
        self.episode_box = self.combo(left, "演示回合", self.episode_name)
        self.episode_box.bind("<<ComboboxSelected>>", self._select_episode)
        self.camera = tk.StringVar()
        self.camera_box = self.combo(left, "录制相机", self.camera)
        self.camera_box.bind("<<ComboboxSelected>>", lambda _event: self.request_frame())
        self.flip = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="上下翻转录制画面", variable=self.flip, command=self._draw).pack(anchor="w", pady=7)
        self.replay_button = ttk.Button(
            left, text="打开此演示的完整三维回放", command=self.open_replay, state="disabled"
        )
        self.replay_button.pack(fill="x")
        self.auto_replay = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left,
            text="选择回合时自动打开三维回放",
            variable=self.auto_replay,
            command=lambda: self.open_replay() if self.auto_replay.get() else None,
        ).pack(anchor="w", pady=(5, 0))
        self.replay_note = ttk.Label(
            left, text="含场景 XML 与仿真状态的 LIBERO HDF5 可使用。", wraplength=330, style="Muted.TLabel"
        )
        self.replay_note.pack(anchor="w", pady=(5, 8))
        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Label(left, text="添加一个演示回合", style="Subheading.TLabel").pack(anchor="w")
        self.online = tk.StringVar()
        self.online_box = self.combo(left, "数据源", self.online)
        self.online_box["values"] = [item[0] for item in SOURCES.values()]
        self.online_box.current(0)
        network = ttk.Frame(left)
        network.pack(fill="x", pady=6)
        self.remote_episode = tk.StringVar(value="0")
        ttk.Label(network, text="回合编号").pack(side="left")
        ttk.Spinbox(network, textvariable=self.remote_episode, from_=0, to=100000, width=7).pack(side="left", padx=8)
        self.download_button = ttk.Button(network, text="下载并打开", command=self.download)
        self.download_button.pack(side="right")
        self.endpoint = tk.StringVar(value=os.environ.get("HF_ENDPOINT", ENDPOINTS[0]))
        self.endpoint_box = self.combo(left, "下载源（连接失败时可切换）", self.endpoint)
        self.endpoint_box["values"] = ENDPOINTS
        if self.endpoint.get() not in ENDPOINTS:
            self.endpoint.set(ENDPOINTS[0])

        self.instruction = ttk.Label(
            right, text="查看真实演示，理解模型应当完成的动作。", style="Subheading.TLabel", wraplength=750
        )
        self.instruction.pack(anchor="w", pady=(0, 5))
        self.original_instruction = ttk.Label(right, text="", wraplength=750, style="Muted.TLabel")
        self.original_instruction.pack(anchor="w", pady=(0, 5))
        self.status = ttk.Label(right, text="尚未加载数据。", wraplength=750, style="Muted.TLabel")
        self.status.pack(anchor="w", pady=(0, 8))
        self.task_download = ttk.Button(
            right, text="下载此任务的一个演示回合", command=self.download_selected_task, style="Accent.TButton"
        )
        self.canvas = tk.Canvas(right, bg="#17221f", highlightthickness=0, height=220)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw())
        self.canvas.create_text(300, 150, text="选择本地数据，或下载一个演示回合", fill="#c7d6d0", tags="placeholder")
        self.segments = SegmentationPanel(
            right,
            seek=self.seek_frame,
            play=self.play_segment,
            configure_overlay=self.replay.configure_overlay,
        )
        self.position = tk.DoubleVar(value=0)
        self.scale = ttk.Scale(right, from_=0, to=1, variable=self.position, command=self._seek)
        self.scale.pack(fill="x", pady=(10, 3))
        self.frame_label = ttk.Label(right, text="帧 — · 时间 —", style="Muted.TLabel")
        self.frame_label.pack(anchor="w")
        controls = ttk.Frame(right)
        controls.pack(fill="x", pady=(8, 10))
        self.play_button = ttk.Button(
            controls, text="播放", style="Accent.TButton", command=self.toggle_play, state="disabled"
        )
        self.play_button.pack(side="left")
        self.previous_button = ttk.Button(controls, text="← 上一帧", command=lambda: self.step(-1), state="disabled")
        self.previous_button.pack(side="left", padx=6)
        self.next_button = ttk.Button(controls, text="下一帧 →", command=lambda: self.step(1), state="disabled")
        self.next_button.pack(side="left")
        self.speed = tk.StringVar(value="1×")
        speed = ttk.Combobox(
            controls, textvariable=self.speed, values=("0.25×", "0.5×", "1×", "2×"), state="readonly", width=7
        )
        speed.pack(side="right")
        speed.bind("<<ComboboxSelected>>", lambda _event: self._restart_clock())
        ttk.Label(controls, text="播放速度").pack(side="right", padx=6)
        self.values = scrolledtext.ScrolledText(
            right,
            height=4,
            font="TkFixedFont",
            relief="flat",
            bg="#ffffff",
            fg="#46534e",
            state="disabled",
            wrap="word",
        )
        self.values.pack(fill="x")
        ttk.Label(right, text="回放到末尾会暂停并保留画面；切换回合由你手动选择。", style="Muted.TLabel").pack(
            anchor="w", pady=(7, 0)
        )
        self._bind_sidebar_scroll(left)
        self.library = DatasetLibrary(
            self,
            open_dataset=self.load,
            browse_directory=self._browse_directory,
            browse_file=self._browse_file,
            refresh=self.refresh_library,
        )
        self.show_library()
        self.after(30, self.poll)

    def _bind_sidebar_scroll(self, widget):
        def scroll(event):
            if self.sidebar_canvas.yview() != (0.0, 1.0):
                direction = -1 if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0 else 1
                self.sidebar_canvas.yview_scroll(direction * 3, "units")
                return "break"
            return None

        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, scroll, add="+")
        for child in widget.winfo_children():
            if child is not self.task_tree:
                self._bind_sidebar_scroll(child)

    @staticmethod
    def combo(parent, label, variable):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 3))
        box = ttk.Combobox(parent, textvariable=variable, state="readonly", width=32)
        box.pack(fill="x")
        return box

    def refresh_local(self, extra=None):
        cache_root = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
        roots = [PREVIEW_ROOT, paths.ROOT / "data/datasets", paths.ROOT / "data/processed", cache_root]
        existing = [*self.paths, *([pathlib.Path(extra).resolve()] if extra is not None else [])]
        found = discover_datasets(roots, extra=existing)
        if extra is not None and pathlib.Path(extra).resolve() not in found:
            found.append(pathlib.Path(extra).resolve())
        source_order = {repo: number for number, (_label, repo, _revision) in enumerate(SOURCES.values())}
        self.paths = sorted(
            found, key=lambda path: (source_order.get(dataset_source(path), len(source_order)), str(path))
        )
        names = {repo: label for label, repo, _revision in SOURCES.values()}
        self.local_box["values"] = [names.get(dataset_source(p), f"{p.name} · {p.parent.name}") for p in self.paths]
        if extra is not None:
            requested = pathlib.Path(extra).resolve()
            selected = requested if requested in self.paths else None
            if selected is not None:
                self.local_box.current(self.paths.index(selected))

    def show_library(self):
        self.pause()
        self.generation += 1
        self.requested = None
        self.replay.stop()
        self.replay_ready = False
        self.replay_pending = None
        self.replay_loading = False
        self.content.pack_forget()
        self.library.pack(fill="both", expand=True)
        self.refresh_library()

    def refresh_library(self):
        self.refresh_local()
        revision = self.library.populate(self.paths)
        for path in self.paths:
            self.library_io.submit(self._read_library_item, revision, path)

    def _read_library_item(self, revision, path):
        if self.closed or revision != self.library.revision:
            return
        index, image, error = None, None, ""
        try:
            index = scan_dataset(path)
            episode = next((ep for ep in index.episodes if ep.cameras), None)
            if episode is not None:
                reader = EpisodeReader(episode)
                try:
                    image, _ = reader.read(0, episode.cameras[0])
                finally:
                    reader.close()
        except Exception as exception:
            error = str(exception)
        self.events.put((None, "library_item", (revision, path, index, image, error)))

    def _local_changed(self, _event=None):
        if self.local_box.current() >= 0:
            self.load(self.paths[self.local_box.current()])

    def _browse_directory(self):
        path = filedialog.askdirectory(title="选择包含 meta/info.json 的 LeRobot 根目录或 HDF5 目录")
        if path:
            self.load(path)

    def _browse_file(self):
        path = filedialog.askopenfilename(title="选择 LIBERO / ALOHA HDF5 演示", filetypes=[("HDF5", "*.hdf5 *.h5")])
        if path:
            self.load(path)

    def load(self, path, preferred_episode=None):
        if self.closed:
            return
        if not str(path).strip():
            self.error("请先选择数据目录或 HDF5 文件。")
            return
        self.library.revision += 1
        self.library.pack_forget()
        self.content.pack(fill="both", expand=True)
        self.pause()
        self.replay.stop()
        self.replay_ready = False
        self.replay_pending = None
        self.replay_loading = False
        self.generation += 1
        generation = self.generation
        self.index = self.episode = None
        self.requested = None
        self.task.set("")
        self.path.set(str(path))
        self.preferred_episode = preferred_episode
        self.reloading = False
        self.reload_retry = False
        self.restoring = None
        self.reload_restore = None
        self.watch.reset()
        self.reload_button.configure(state="normal")
        self.overview_title.configure(text="正在读取数据集…")
        self.overview_description.configure(text="")
        self.overview_stats.configure(text="")
        self.summary.configure(text="正在读取回合信息…")
        self.original_instruction.configure(text="")
        self.task_download.pack_forget()
        self.task_tree.delete(*self.task_tree.get_children())
        self.task_rows.clear()
        self.search.set("")
        self.only_local.set(False)
        self.instruction.configure(text="查看真实演示，理解模型应当完成的动作。")
        self.frame_label.configure(text="帧 — · 时间 —")
        self._show_values({})
        self.index = self.episode = None
        self.requested = None
        self.image = self.photo = None
        self.preview_message = "正在读取所选数据集的任务目录…"
        self._draw()
        self._controls(enabled=False)
        self.replay_button.configure(state="disabled")
        for box in (self.kind_box, self.episode_box, self.camera_box):
            box.set("")
            box["values"] = ()
        self.status.configure(text="正在读取数据目录与回合元数据…", style="Muted.TLabel")

        def scan():
            self._close_reader()
            try:
                index = scan_dataset(path)
                self.events.put((generation, "index", index))
            except Exception as error:
                self.events.put((generation, "error", str(error)))

        self.io.submit(scan)

    def _filter_tasks(self, _event=None):
        if self.index is None:
            return
        previous = self.task.get()
        query = self.search.get().strip().lower()
        tasks = [
            task
            for task in self.index.tasks
            if self.kind.get() in ("全部操作", task.kind)
            and (not self.only_local.get() or task.local_episodes)
            and (not query or query in f"{task.name} {task_label(task.name)} {task.kind}".lower())
        ]
        self.task_tree.delete(*self.task_tree.get_children())
        self.task_rows = {str(number): task for number, task in enumerate(tasks)}
        for row, task in self.task_rows.items():
            total = str(task.total_episodes) if task.total_episodes is not None else "—"
            self.task_tree.insert(
                "",
                "end",
                iid=row,
                text=task_label(task.name),
                values=(f"{len(task.local_episodes)} / {total}",),
                tags=("local" if task.local_episodes else "remote",),
            )
        self.task_count.configure(text=f"显示 {len(tasks)} / {len(self.index.tasks)} 项任务 · 绿色为本地可播放")
        if tasks:
            selected = next((task for task in tasks if task.name == previous), None)
            selected = selected or next((task for task in tasks if task.local_episodes), tasks[0])
            row = next(key for key, task in self.task_rows.items() if task is selected)
            self.task.set(selected.name)
            self.task_tree.selection_set(row)
            self.task_tree.see(row)
            self._filter_episodes()
        else:
            self.task.set("")
            self._clear_episode("没有符合条件的任务。试试其他关键词，或取消筛选。")
            self.instruction.configure(text="没有符合条件的任务")
            self.original_instruction.configure(text="")
            self.status.configure(text="请调整左侧搜索或操作类型。", style="Muted.TLabel")

    def _task_selected(self, _event=None):
        selection = self.task_tree.selection()
        task = self.task_rows.get(selection[0]) if selection else None
        if task is not None and self.task.get() != task.name:
            self.task.set(task.name)
            self._filter_episodes()

    def _selected_task(self):
        return next((task for task in self.index.tasks if task.name == self.task.get()), None) if self.index else None

    def _clear_episode(self, message):
        self.pause()
        self.generation += 1
        self.replay.stop()
        self.replay_ready = False
        self.replay_pending = None
        self.replay_loading = False
        self.episode = None
        self.requested = None
        self.filtered_episodes = []
        self.image = self.photo = None
        self.segments.set_data(None)
        self.replay_after_read = False
        self.preview_message = message
        self.episode_box["values"] = ()
        self.episode_name.set("")
        self.camera_box["values"] = ()
        self.camera.set("")
        self.frame_label.configure(text="帧 — · 时间 —")
        self._show_values({})
        self._draw()
        self._controls(enabled=False)
        self.replay_button.configure(state="disabled")
        self.replay_note.configure(text="选择有本地演示的任务后，可查看回放方式。")
        self.task_download.pack_forget()

    def _filter_episodes(self, _event=None):
        task = self._selected_task()
        if task is None:
            return
        self.filtered_episodes = task.local_episodes
        self.episode_box["values"] = [f"{ep.name} · {ep.frames} 帧" for ep in self.filtered_episodes]
        if self.filtered_episodes:
            selected = 0
            if self.restoring:
                selected = next(
                    (
                        i
                        for i, episode in enumerate(self.filtered_episodes)
                        if self._episode_key(episode) == self.restoring["episode"]
                    ),
                    0,
                )
            self.episode_box.current(selected)
            self._select_episode()
        else:
            self._clear_episode("此任务的演示尚未下载\n可以先查看任务说明，再按需下载一个回合")
            self.instruction.configure(text=task_label(task.name))
            self.original_instruction.configure(text=task.name if task_label(task.name) != task.name else "")
            total = (
                f"元数据中有 {task.total_episodes} 个回合" if task.total_episodes is not None else "元数据未标注回合数"
            )
            self.status.configure(text=f"{task.kind} · {total} · 本地可播放 0 个", style="Muted.TLabel")
            if self._download_source() is not None and task.example_episode is not None:
                self.task_download.configure(state="disabled" if self.downloading else "normal")
                self.task_download.pack(anchor="w", pady=(0, 10), before=self.canvas)

    def _download_source(self):
        if self.index:
            repo = dataset_source(self.index.root)
            return next((key for key, (_label, source, _revision) in SOURCES.items() if source == repo), None)
        return None

    def download_selected_task(self):
        task, source = self._selected_task(), self._download_source()
        if task is not None and task.example_episode is not None and source is not None:
            self.online_box.current(list(SOURCES).index(source))
            self.remote_episode.set(str(task.example_episode))
            self.download()

    def _select_episode(self, _event=None):
        if not self.filtered_episodes:
            return
        restore = self.restoring
        if restore is None:
            self.reloading = self.reload_retry = False
            self.reload_restore = None
            self.reload_button.configure(state="normal")
        self.pause()
        self.generation += 1
        self.replay_ready = False
        self.replay_pending = None
        self.episode = self.filtered_episodes[self.episode_box.current()]
        episode = self.episode
        same_episode = restore and self._episode_key(episode) == restore["episode"]
        if episode.processed_root and not same_episode:
            self.flip.set(True)
        self.frame = min(restore["frame"], episode.frames - 1) if same_episode else 0
        self.watch.reset()
        self.image = self.photo = None
        self.segments.set_data(None)
        self._draw()
        self.camera_box["values"] = episode.cameras
        if episode.cameras:
            camera = restore["camera"] if restore else ""
            self.camera_box.current(episode.cameras.index(camera) if camera in episode.cameras else 0)
        else:
            self.camera.set("")
        self.task_download.pack_forget()
        label = task_label(self.task.get() or episode.task)
        self.instruction.configure(text=label)
        self.original_instruction.configure(text=episode.task if label != episode.task else "")
        self.status.configure(
            text=f"{episode.format} · {episode.fps:g} FPS · {episode.frames / episode.fps:.1f} 秒", style="Muted.TLabel"
        )
        self.replay_button.configure(state="normal" if episode.replay and not episode.processed_root else "disabled")
        self.replay_note.configure(text=self._replay_description())
        if not episode.cameras:
            self.preview_message = "此回合保存了完整仿真状态。\n打开三维回放后，可用下方时间轴观看。"
            self._draw()
        self.scale.configure(to=max(episode.frames - 1, 1))
        self._controls(enabled=True)
        self.request_frame()
        self.replay_loading = False
        self.replay_after_read = False
        if episode.replay and (self.auto_replay.get() or (same_episode and restore["replay"])):
            self.replay_loading = True
            self.replay_after_read = bool(episode.processed_root)
            if not self.replay_after_read:
                self.replay.reload(episode, frame=self.frame)
            self.replay_note.configure(text="正在打开或更新三维回放…")
        else:
            self.replay.stop()
        if same_episode and restore["playing"]:
            self.toggle_play()

    def _close_reader(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def request_frame(self):
        if self.episode is None or self.closed:
            return
        self.requested = (self.generation, self.episode, self.frame, self.camera.get() or None)
        self._position_label()
        self._submit_frame()

    def _submit_frame(self):
        if self.render_pending or self.requested is None:
            return
        generation, episode, frame, camera = self.requested
        self.requested = None
        self.render_pending = True

        def read():
            try:
                if self.reader is None or self.reader.episode is not episode:
                    self._close_reader()
                    self.reader = EpisodeReader(episode)
                    self.events.put((generation, "segmentation", self.reader.segmentation))
                image, values = self.reader.read(frame, camera)
                self.events.put((generation, "frame", (frame, camera, image, values)))
            except Exception as error:
                self.events.put((generation, "frame_error", str(error)))

        self.io.submit(read)

    def _position_label(self):
        if self.episode is None:
            return
        self.updating_scale = True
        self.position.set(self.frame)
        self.updating_scale = False
        self.segments.update_frame(self.frame)
        self.frame_label.configure(
            text=f"帧 {self.frame} / {self.episode.frames - 1} · {self.frame / self.episode.fps:.2f} / {(self.episode.frames - 1) / self.episode.fps:.2f} 秒"
        )

    def _seek(self, value):
        if self.episode is None or self.updating_scale:
            return
        self.pause()
        self.frame = min(self.episode.frames - 1, max(0, round(float(value))))
        self.request_frame()

    def seek_frame(self, frame):
        self._seek(frame)

    def play_segment(self, start, end):
        self.seek_frame(start)
        self.toggle_play(stop=end)

    def step(self, delta):
        if self.episode is None:
            return
        self.pause()
        self.frame = min(self.episode.frames - 1, max(0, self.frame + delta))
        self.request_frame()

    def pause(self):
        self.playing = False
        self.play_button.configure(text="播放")
        self.replay.send("playing", playing=False)

    def toggle_play(self, *, stop=None):
        if self.episode is None:
            return
        if self.playing:
            self.pause()
        else:
            if self.frame == self.episode.frames - 1:
                self.frame = 0
                self.request_frame()
            self.playing = True
            self.play_limit = stop
            self.play_button.configure(text="暂停")
            self._restart_clock()
            self.replay.send("playing", playing=True)

    def _restart_clock(self):
        self.last_tick = time.monotonic()

    def _controls(self, enabled):
        for widget in (self.play_button, self.previous_button, self.next_button, self.scale):
            widget.configure(state="normal" if enabled else "disabled")

    def _draw(self):
        self.canvas.delete("all")
        width, height = max(self.canvas.winfo_width(), 20), max(self.canvas.winfo_height(), 20)
        if self.image is None:
            self.canvas.create_text(
                width / 2,
                height / 2,
                text=self.preview_message,
                font="TkDefaultFont",
                width=max(width - 40, 20),
                fill="#c7d6d0",
                justify="center",
            )
            return
        image = self.image[::-1] if self.flip.get() else self.image
        picture = Image.fromarray(image)
        ratio = min(width / picture.width, height / picture.height)
        picture = picture.resize(
            (max(1, round(picture.width * ratio)), max(1, round(picture.height * ratio))), Image.Resampling.LANCZOS
        )
        # ImageTk imports tkinter; defer it until tk_runtime has selected the desktop backend.
        from PIL import ImageTk

        self.photo = ImageTk.PhotoImage(picture, master=self.canvas)
        self.canvas.create_image(width / 2, height / 2, image=self.photo)

    def _show_values(self, values):
        lines = []
        for key, value in values.items():
            if isinstance(value, list | tuple):
                rendered = np.array2string(
                    np.asarray(value), precision=4, separator=", ", threshold=30, max_line_width=1000
                )
            else:
                rendered = str(value)
            lines.append(f"{key}: {rendered}")
        self.values.configure(state="normal")
        self.values.delete("1.0", "end")
        self.values.insert("1.0", "\n".join(lines) or "此帧未包含动作或机器人状态。")
        self.values.configure(state="disabled")

    def open_replay(self):
        if self.episode is not None and self.episode.replay:
            if self.episode.replay_source and self.episode.replay_source.frame(self.frame) is None:
                self.replay_note.configure(text="终端观测没有对应的完整仿真状态；请跳到此前区间再打开三维回放。")
                return
            self.pause()
            self.replay_ready = False
            self.replay_note.configure(text="正在打开三维回放窗口…")
            self.replay_pending = None
            self.replay_loading = True
            self.replay.reload(self.episode, frame=self.frame)

    def download(self):
        if self.downloading:
            return
        try:
            number = int(self.remote_episode.get())
            if number < 0:
                raise ValueError
        except ValueError:
            self.error("下载回合编号须为非负整数。")
            return
        source = list(SOURCES)[self.online_box.current()]
        endpoint = self.endpoint.get()
        self.downloading = True
        self.download_button.configure(state="disabled")
        self.task_download.configure(state="disabled")

        def run():
            try:
                path = download_preview(
                    source,
                    number,
                    endpoint,
                    lambda message: self.events.put((None, "download_status", message)),
                    cancelled=self.download_cancelled.is_set,
                )
                self.events.put((None, "downloaded", (path, number)))
            except Exception as error:
                self.events.put((None, "download_error", str(error)))

        self.downloads.submit(run)

    def error(self, message):
        self.pause()
        self.status.configure(text=message, style="Error.TLabel")

    @staticmethod
    def _episode_key(episode):
        return str(episode.path), episode.group, episode.name

    def _replay_description(self):
        if self.episode is None:
            return "请选择一个演示回合。"
        if self.episode.replay_error:
            return self.episode.replay_error
        if self.episode.replay_source:
            if self.episode.processed_root:
                return "三维彩色轨迹与命令分段同步，当前段加粗、白点定位；示教回放，尚未执行拟合场。"
            return "原始参考回放：按帧映射显示来源场景；数值区显示处理后的数据。"
        if self.episode.replay:
            return "记录状态三维回放，与下方时间轴同步。"
        return "此回合提供录制画面；完整三维回放需要场景 XML 和仿真 states，或明确的来源帧映射。"

    def _sync_replay_frame(self, frame):
        if self.episode and self.episode.replay_source and self.episode.replay_source.frame(frame) is None:
            self.replay_ready = self.replay_loading = False
            self.replay_pending = None
            self.replay.stop()
            self.replay_note.configure(text="终端观测无完整仿真状态，三维窗口已关闭；录制末帧与本体仍可查看。")
            return
        if self.replay_ready:
            self.replay_pending = frame
            self.replay_sent_at = time.monotonic()
            self.replay.send("frame", frame=frame)

    def reload_current(self):
        if self.index is None or self.reloading or self.closed:
            return
        restore = self.reload_restore if self.reload_retry else None
        restore = restore or {
            "episode": self._episode_key(self.episode) if self.episode else None,
            "frame": self.frame,
            "camera": self.camera.get(),
            "kind": self.kind.get(),
            "playing": self.playing,
            "replay": self.replay_ready or self.replay_loading,
        }
        self.reload_restore = restore
        self.pause()
        self.replay_pending = None
        self.replay_ready = False
        self.reloading = True
        self.last_reload = time.monotonic()
        self.reload_button.configure(state="disabled")
        self.generation += 1
        generation, root = self.generation, self.index.root
        self.requested = None
        self.status.configure(text="正在更新数据，保留当前回合与观看位置…", style="Muted.TLabel")

        def scan():
            self._close_reader()
            try:
                self.events.put((generation, "reload_index", (scan_dataset(root), restore)))
            except Exception as error:
                self.events.put((generation, "reload_error", str(error)))

        self.io.submit(scan)

    def _poll_watch(self):
        if not self.auto_refresh.get() or self.index is None or self.closed or self.reloading:
            return
        now = time.monotonic()
        if self.reload_retry and now - self.last_reload > 5:
            self.reload_current()
            return
        if self.watch_pending or now - self.last_watch < 1:
            return
        self.last_watch = now
        self.watch_pending = True
        generation, watched = self.generation, watch_paths(self.index, self.episode)

        def check():
            self.events.put((generation, "watch", file_signature(watched)))

        self.io.submit(check)

    def _install_index(self, value, restore=None):
        self.restoring = restore
        try:
            self.index = value
            self.refresh_local(value.root)
            title, description, robot = dataset_description(value)
            self.overview_title.configure(text=title)
            fields = []
            if any("state" in key or "qpos" in key for key in value.fields):
                fields.append("机器人状态")
            if any("action" in key for key in value.fields):
                fields.append("动作")
            contents = "、".join(["录制画面" if value.cameras else "仿真状态", *fields])
            self.overview_description.configure(text=f"{robot} · {description}  数据包含：{contents}。")
            total = (
                f"元数据 {value.declared_episodes:,} 个回合"
                if value.declared_episodes is not None
                else "总回合数未标注"
            )
            self.overview_stats.configure(
                text=f"{len(value.tasks)} 项任务    ·    {total}    ·    本地可播放 {len(value.episodes)} 个    ·    {len(value.cameras)} 路相机"
            )
            local_tasks = sum(bool(task.local_episodes) for task in value.tasks)
            self.summary.configure(
                text="\n".join([f"{local_tasks} 项任务有本地演示，其余可先查看说明。", *value.notes[:2]])
            )
            self.kind_box["values"] = ["全部操作", *sorted({task.kind for task in value.tasks})]
            kind = restore["kind"] if restore else "全部操作"
            self.kind.set(kind if kind in self.kind_box["values"] else "全部操作")
            self._filter_tasks()
            if self.preferred_episode is not None and restore is None:
                preferred = next(
                    (ep for ep in value.episodes if ep.name == f"episode_{self.preferred_episode:06d}"), None
                )
                if preferred is not None:
                    selected = next(task for task in value.tasks if preferred in task.local_episodes)
                    self.task.set(selected.name)
                    self._filter_tasks()
                    self.episode_box.current(self.filtered_episodes.index(preferred))
                    self._select_episode()
                self.preferred_episode = None
        finally:
            self.restoring = None

    def poll(self):
        if self.closed:
            return
        try:
            while True:
                generation, kind, value = self.events.get_nowait()
                if kind in ("frame", "frame_error"):
                    self.render_pending = False
                if kind == "watch":
                    self.watch_pending = False
                if generation is not None and generation != self.generation:
                    continue
                if kind == "library_item":
                    self.library.set_result(*value)
                elif kind == "index":
                    self._install_index(value)
                elif kind == "reload_index":
                    index, restore = value
                    self.reloading = self.reload_retry = False
                    self.reload_button.configure(state="normal")
                    self._install_index(index, restore)
                    self.reload_restore = None
                elif kind == "reload_error":
                    self.reloading = False
                    self.reload_retry = True
                    self.reload_button.configure(state="normal")
                    self.error("暂时无法更新，保留当前画面：" + value)
                elif kind == "watch":
                    if self.auto_refresh.get() and self.watch.changed(value):
                        self.reload_current()
                elif kind == "segmentation":
                    self.segments.set_data(value)
                    if value:
                        self.segments.pack(fill="x", pady=(8, 0), before=self.scale)
                        self.segments.update_frame(self.frame)
                    self.replay_button.configure(state="normal" if self.episode.replay else "disabled")
                    if self.replay_after_read:
                        self.replay_after_read = False
                        if self.episode.replay and self.episode.replay_source.frame(self.frame) is not None:
                            self.replay.reload(self.episode, frame=self.frame)
                        else:
                            self.replay_loading = False
                elif kind == "frame":
                    frame, camera, image, values = value
                    if self.episode is not None and frame == self.frame and camera == (self.camera.get() or None):
                        self.image = image
                        if values.get("image_valid") is False:
                            self.preview_message = "初始录制图像缺失。\n可打开三维状态回放查看真实来源状态。"
                        self._draw()
                        self._show_values(values)
                        self._sync_replay_frame(frame)
                elif kind in ("error", "frame_error"):
                    self.error(value)
                elif kind == "download_status":
                    self.status.configure(text=value, style="Muted.TLabel")
                elif kind in ("downloaded", "download_error"):
                    self.downloading = False
                    self.download_button.configure(state="normal")
                    self.task_download.configure(state="normal")
                    if kind == "downloaded":
                        self.load(value[0], preferred_episode=value[1])
                    else:
                        self.error("下载失败，可切换下载源后重试：" + value)
        except queue.Empty:
            pass
        self._submit_frame()
        try:
            while True:
                generation, event = self.replay.events.get_nowait()
                if generation != self.replay.generation:
                    continue
                if event.get("ready"):
                    self.replay_loading = False
                    self.replay_ready = True
                    self._sync_replay_frame(self.frame)
                    self.replay.send("playing", playing=self.playing)
                    self.replay_note.configure(text=self._replay_description() + " 三维窗口已就绪。")
                if "frame" in event:
                    if event["frame"] == self.replay_pending:
                        self.replay_pending = None
                    self.replay_note.configure(
                        text=self._replay_description()
                        + f" 当前三维帧 {event['frame']}，来源帧 {event['source_frame']}。"
                    )
                if event.get("stopped"):
                    self.replay_loading = False
                    self.replay_ready = False
                    self.replay_pending = None
                    self.replay_note.configure(text="三维回放已关闭，可重新打开。")
                if "playing" in event and event["playing"] != self.playing:
                    self.toggle_play()
                if event.get("step"):
                    self.step(1)
                if "error" in event:
                    self.replay_loading = False
                    self.replay_pending = None
                    self.error("三维回放失败：" + event["error"])
        except queue.Empty:
            pass
        if self.replay_pending is not None and time.monotonic() - self.replay_sent_at > 5:
            self.replay_pending = None
            self.error("三维回放响应较慢，已暂停；可重新打开三维窗口。")
        self._poll_watch()
        # Advance only after both the numeric/image read and the native 3D update are acknowledged.
        if (
            self.playing
            and self.episode is not None
            and not self.render_pending
            and self.replay_pending is None
            and not self.replay_loading
            and not self.reloading
        ):
            interval = 1 / (self.episode.fps * float(self.speed.get().rstrip("×")))
            now = time.monotonic()
            if now - self.last_tick >= interval:
                self.last_tick = now
                if self.frame >= (self.play_limit if self.play_limit is not None else self.episode.frames - 1):
                    self.pause()
                else:
                    self.frame += 1
                    self.request_frame()
        self.after(15, self.poll)

    def close(self):
        if self.closed:
            return
        self.pause()
        self.closed = True
        self.download_cancelled.set()
        self.generation += 1
        self.replay.stop()
        self.io.submit(self.replay.close)
        self.io.submit(self._close_reader)
        self.io.shutdown(wait=False)
        self.library_io.shutdown(wait=False, cancel_futures=True)
        self.downloads.shutdown(wait=False, cancel_futures=True)
