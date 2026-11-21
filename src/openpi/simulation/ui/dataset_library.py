"""Visual entry point to the locally indexed demonstration collections."""

# ruff: noqa: RUF001
from PIL import Image

from openpi.simulation.data.descriptions import dataset_description
from openpi.simulation.ui.tk_runtime import tk
from openpi.simulation.ui.tk_runtime import ttk


class DatasetLibrary(ttk.Frame):
    def __init__(self, parent, *, open_dataset, browse_directory, browse_file, refresh):
        super().__init__(parent)
        self.open_dataset = open_dataset
        self.cards = {}
        self.revision = 0
        style = ttk.Style(self)
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("Card.TLabel", background="#ffffff")
        style.configure(
            "CardTitle.TLabel",
            background="#ffffff",
            font=style.lookup("Subheading.TLabel", "font"),
            foreground="#194f40",
        )
        style.configure("CardNote.TLabel", background="#ffffff", foreground="#66756f")
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="先看看数据集里有什么", style="Subheading.TLabel").pack(side="left")
        ttk.Button(header, text="刷新", command=refresh).pack(side="right")
        ttk.Button(header, text="打开 HDF5", command=browse_file).pack(side="right", padx=6)
        ttk.Button(header, text="打开数据目录", command=browse_directory).pack(side="right")
        self.introduction = ttk.Label(
            self,
            text="每张卡片展示一个数据集的内容和真实画面。点开后可查看完整任务目录，再选择一个回合观看。",
            style="Muted.TLabel",
        )
        self.introduction.pack(anchor="w", pady=(0, 12))
        viewport = ttk.Frame(self)
        viewport.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(viewport, bg="#f4f6f5", highlightthickness=0)
        scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_frame = ttk.Frame(self.canvas)
        self.grid_frame.columnconfigure((0, 1), weight=1, uniform="datasets")
        self.grid_window = self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.grid_window, width=event.width))
        ttk.Label(
            self,
            text="缩略图来自本地演示首帧。任务与总回合数来自元数据；“本地可播放”表示已下载且可直接观看的回合。",
            style="Muted.TLabel",
            wraplength=1080,
        ).pack(anchor="w", pady=(10, 0))

    def populate(self, paths):
        self.revision += 1
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.cards.clear()
        if not paths:
            ttk.Label(self.grid_frame, text="还没有本地数据集。可以打开数据目录或 HDF5 文件。", padding=30).grid()
        for number, path in enumerate(paths):
            card = ttk.Frame(self.grid_frame, style="Card.TFrame", padding=16)
            card.grid(row=number // 2, column=number % 2, sticky="nsew", padx=(0, 12), pady=(0, 12))
            title = ttk.Label(card, text=path.stem, style="CardTitle.TLabel", wraplength=475)
            title.pack(anchor="w", pady=(0, 10))
            body = ttk.Frame(card, style="Card.TFrame", borderwidth=0)
            body.pack(fill="x")
            preview = tk.Canvas(body, width=156, height=110, bg="#20332c", highlightthickness=0)
            preview.pack(side="left", padx=(0, 12))
            preview.create_text(78, 55, text="正在读取预览…", fill="#d9e6df", font="TkDefaultFont")
            description = ttk.Label(body, text="正在读取内容目录…", style="Card.TLabel", wraplength=290, justify="left")
            description.pack(side="left", fill="x", expand=True, anchor="n")
            stats = ttk.Label(card, text="正在索引元数据…", style="Card.TLabel")
            stats.pack(anchor="w", pady=(10, 3))
            available = ttk.Label(card, text="", style="CardNote.TLabel")
            available.pack(anchor="w")
            button = ttk.Button(
                card,
                text="查看任务与回放 →",
                command=lambda selected=path: self.open_dataset(selected),
                state="disabled",
            )
            button.pack(anchor="e", pady=(10, 0))
            self.cards[path] = {
                "title": title,
                "description": description,
                "stats": stats,
                "available": available,
                "preview": preview,
                "button": button,
                "photo": None,
                "index": None,
            }
        return self.revision

    def set_result(self, revision, path, index, image, error):
        if revision != self.revision or path not in self.cards:
            return
        card = self.cards[path]
        if index is None:
            card["description"].configure(text=error)
            card["stats"].configure(text="暂时无法索引此目录")
            card["preview"].delete("all")
            card["preview"].create_text(78, 55, text="暂无预览", fill="#d9e6df", font="TkDefaultFont")
            return
        title, description, robot = dataset_description(index)
        card["index"] = index
        card["title"].configure(text=title)
        card["description"].configure(text=f"{robot}\n\n{description}")
        total = f"{index.declared_episodes:,} 个回合" if index.declared_episodes is not None else "总回合数未标注"
        card["stats"].configure(text=f"{len(index.tasks)} 项任务 · {total} · {len(index.cameras)} 路相机")
        kinds = " / ".join(sorted({task.kind for task in index.tasks}))
        card["available"].configure(text=f"本地可播放 {len(index.episodes)} 个回合 · {kinds}", wraplength=470)
        card["button"].configure(state="normal")
        preview = card["preview"]
        preview.delete("all")
        if image is not None:
            # Import after the application's Tk backend has been chosen.
            from PIL import ImageTk

            picture = Image.fromarray(image)
            picture.thumbnail((156, 110), Image.Resampling.LANCZOS)
            card["photo"] = ImageTk.PhotoImage(picture, master=preview)
            preview.create_image(78, 55, image=card["photo"])
        else:
            preview.create_text(78, 55, text="暂无本地画面", fill="#d9e6df", font="TkDefaultFont")
