"""Nonblocking controls for fixed-field diagnostics beside recorded-state replay."""

# ruff: noqa: RUF001 -- Chinese interface text.
from collections import OrderedDict
import concurrent.futures
import hashlib
import json
from pathlib import Path
import queue
import tempfile

import numpy as np

from openpi.simulation.ui.tk_runtime import filedialog
from openpi.simulation.ui.tk_runtime import tk
from openpi.simulation.ui.tk_runtime import ttk


class FieldPreviewPanel(ttk.Frame):
    def __init__(self, parent, *, replay, seek, play, on_segmentation=None):
        super().__init__(parent)
        self.replay, self.seek, self.play = replay, seek, play
        self.on_segmentation = on_segmentation
        self.episode = self.segmentation = self.result = self.snapshot = None
        self.fields_path = None
        self.offline = None
        self.offline_selected = None
        self.profile_code = "replay"
        self.profile_codes = {
            "回放配置": "replay",
            "生产控制配置": "production_libero",
            "生产约束/放慢时间": "production_retimed",
            "长片段/非线性场": "coherent",
            "长片段/过点场": "coherent_flow",
        }
        self.auto_cache = OrderedDict()
        self.auto_key = None
        self.auto_failed = set()
        self.auto_request_key = None
        # Offline field snapshots are prepared away from Tk.  The previous
        # implementation sampled and wrote the next field synchronously when
        # the cursor crossed a segment boundary, which made playback pause.
        self.offline_pending = set()
        self.frame = self.generation = self.job = 0
        self.closed = self.busy = False
        self.events = queue.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="field-preview")
        self.temporary = tempfile.TemporaryDirectory(prefix="vlds-field-preview-")
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="动力系统场", style="Subheading.TLabel").pack(side="left", padx=(0, 10))
        ttk.Label(row, text="配置").pack(side="left")
        self.profile = tk.StringVar(value="回放配置")
        self.profile_choice = ttk.Combobox(row, textvariable=self.profile, state="readonly", width=13)
        self.profile_choice.pack(side="left", padx=(4, 8))
        self.profile_choice.bind("<<ComboboxSelected>>", lambda _event: self._profile_changed())
        self.fit_button = ttk.Button(row, text="诊断整命令段", command=self.fit_current)
        self.fit_button.pack(side="left")
        self.representation = tk.StringVar(value="SEDS")
        ttk.Combobox(
            row, textvariable=self.representation, values=("SEDS", "线性吸引子"), state="readonly", width=11
        ).pack(side="left", padx=5)
        ttk.Label(row, text="手动诊断限速 m/s").pack(side="left")
        self.max_speed = tk.StringVar(value="0.3")
        ttk.Combobox(row, textvariable=self.max_speed, values=("0.1", "0.2", "0.3"), state="readonly", width=4).pack(
            side="left", padx=5
        )
        self.load_button = ttk.Button(row, text="载入拟合结果…", command=self.choose_file)
        self.load_button.pack(side="left", padx=5)
        self.focus_button = ttk.Button(row, text="聚焦当前场", command=self.focus, state="disabled")
        self.focus_button.pack(side="left", padx=5)
        self.export_button = ttk.Button(row, text="导出诊断图…", command=self.export, state="disabled")
        self.export_button.pack(side="right")
        row = ttk.Frame(self)
        row.pack(fill="x", pady=(4, 0))
        self.choice = ttk.Combobox(row, state="readonly", width=34)
        self.choice.pack(side="left", fill="x", expand=True)
        self.choice.bind("<<ComboboxSelected>>", lambda _event: self.show_record())
        self.choices = []
        self.play_button = ttk.Button(row, text="播放该场区间", command=self.play_interval, state="disabled")
        self.play_button.pack(side="left", padx=5)
        self.automatic = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="自动分段与切换场", variable=self.automatic, command=self._automatic_changed).pack(
            side="left", padx=5
        )
        self.visible = tk.BooleanVar(value=True)
        self.vectors = tk.BooleanVar(value=True)
        self.rollout = tk.BooleanVar(value=True)
        for text, variable in (("显示场", self.visible), ("速度箭头", self.vectors), ("名义轨迹", self.rollout)):
            ttk.Checkbutton(row, text=text, variable=variable, command=self.configure_overlay).pack(
                side="left", padx=(7, 0)
            )
        self.detail = ttk.Label(self, text="", style="Muted.TLabel", wraplength=1050)
        self.detail.pack(anchor="w", pady=(3, 0))
        self.representation.trace_add("write", lambda *_: self._automatic_changed())
        self.max_speed.trace_add("write", lambda *_: self._automatic_changed())
        self.after(80, self.poll)
        from openpi.simulation.ui.native_field import NativeFieldCanvas

        self.native = NativeFieldCanvas(self)

    def set_episode(self, episode, segmentation):
        self.generation += 1
        self.episode, self.segmentation = episode, segmentation
        self.offline = segmentation.get("offline_bundle") if segmentation else None
        self.profile_code = segmentation.get("profile", "replay") if segmentation else "replay"
        self._set_profile_choices()
        self.offline_selected = None
        self.offline_pending.clear()
        self.native.set_bundle(self.offline)
        if self.offline and episode.processed_root is None:
            self.native.pack(fill="x", pady=(4, 0))
        else:
            self.native.pack_forget()
        self.result = self.snapshot = self.fields_path = None
        self.busy = False
        self.auto_key = None
        self.choices = []
        self.choice.set("")
        self.choice["values"] = ()
        self.replay.configure_overlay(field_preview=None)
        self._buttons()
        if episode is None or (episode.processed_root is None and not self.offline) or not segmentation:
            self.pack_forget()
            return
        self.detail.configure(text="自动准备当前片段的动力系统场… 橙线为示教，紫线为名义积分，绿点为目标。")
        self._ensure_automatic()
        self._prefetch_offline()

    def _set_profile_choices(self):
        if self.episode is None:
            self.profile_choice["values"] = ()
            return
        from openpi.datasets.dynamics.playback import available_profiles

        available = set(available_profiles(self.episode))
        if self.offline and self.offline.get("manual"):
            available.add(self.profile_code)
        labels = [label for label, code in self.profile_codes.items() if code in available]
        self.profile_choice["values"] = labels
        selected = next((label for label, code in self.profile_codes.items() if code == self.profile_code), "回放配置")
        if selected in labels:
            self.profile.set(selected)
        elif labels:
            self.profile.set(labels[0])

    def _profile_changed(self):
        if self.busy or self.episode is None or self.offline is None:
            return
        code = self.profile_codes.get(self.profile.get(), "replay")
        if code == self.profile_code:
            return
        from openpi.datasets.dynamics.playback import load_for_episode

        episode = self.episode
        self.detail.configure(text=f"正在载入{self.profile.get()}的持久化动力系统场…")
        self._submit(
            "profile",
            lambda: (
                code,
                load_for_episode(episode, profile=code),
            ),
        )

    def update_frame(self, frame):
        self.frame = frame
        self._ensure_automatic()
        if self.snapshot and not self.offline:
            self._describe()

    def _automatic_changed(self):
        self.auto_failed.clear()
        self.auto_key = None
        self._ensure_automatic()

    def _ensure_automatic(self):
        if not self.automatic.get() or self.episode is None or not self.segmentation or self.busy:
            return
        if self.offline:
            self._ensure_offline()
            return
        from openpi.datasets.libero.field_preview_sequence import automatic_previews
        from openpi.datasets.libero.field_preview_sequence import interval_at
        from openpi.simulation import paths

        episode = self.episode
        if episode.processed_root is None:
            return
        index = interval_at(self.segmentation["intervals"], self.frame)
        if index is None:
            return
        speed = float(self.max_speed.get())
        representation = "fitted" if self.representation.get() == "SEDS" else "linear"
        manifest = (Path(episode.processed_root) / "manifest.json").stat()
        key = (
            str(episode.processed_root),
            episode.processed_index,
            index,
            speed,
            representation,
            manifest.st_mtime_ns,
            manifest.st_size,
        )
        if key in self.auto_failed:
            return
        if key in self.auto_cache:
            self.auto_cache.move_to_end(key)
            entries = self.auto_cache[key]
            selected = interval_at([value[1]["interval"] for value, _ in entries], self.frame)
            if selected is not None:
                value, path = entries[selected]
                if self.auto_key != (key, selected):
                    self.auto_key = (key, selected)
                    self.snapshot = value
                    self.choice.set(f"自动场 {selected + 1}/{len(entries)}")
                    self.configure_overlay()
                    self.replay.configure_overlay(field_preview=str(path))
                    self._buttons()
            return
        self.detail.configure(text=f"正在准备命令段 {index + 1} 的场：复用已核验结果，或自动拟合并细分…")
        generation = self.generation
        self.auto_request_key = key
        self._submit(
            "automatic",
            lambda: (
                key,
                automatic_previews(
                    episode.processed_root,
                    episode.processed_index,
                    index,
                    max_speed=speed,
                    representation=representation,
                    search_root=paths.ROOT / "data/experiments",
                    cancelled=lambda: self.closed or generation != self.generation,
                ),
            ),
        )

    def _offline_key(self, index):
        return ("offline", self.offline["path"], int(index))

    def _queue_offline_snapshot(self, index):
        """Prepare one persisted field without blocking the Tk event loop."""
        if self.offline is None or self.episode is None:
            return
        key = self._offline_key(index)
        if key in self.auto_cache or key in self.offline_pending:
            return
        from openpi.datasets.dynamics.playback import preview
        from openpi.fields.visualization import save_snapshot

        bundle = self.offline
        episode = self.episode
        generation = self.generation
        profile = self.profile_code
        self.offline_pending.add(key)

        def work():
            identity = hashlib.sha256(bundle["path"].encode()).hexdigest()[:16]
            path = Path(self.temporary.name) / f"offline-{generation}-{profile}-{identity}-{index}.npz"
            try:
                snapshot = preview(bundle, index, episode)
                save_snapshot(path, *snapshot)
                self.events.put((generation, None, "offline_snapshot", (key, index, snapshot, path), None))
            except Exception as error:
                path.unlink(missing_ok=True)
                self.events.put((generation, None, "offline_snapshot", (key, index), str(error)))

        self.executor.submit(work)

    def _prefetch_offline(self):
        """Queue the current episode's fields, prioritizing the visible one."""
        if self.offline is None or self.episode is None or self.episode.processed_root is None:
            return
        from openpi.datasets.libero.field_preview_sequence import interval_at

        records = self.offline["result"]["records"]
        if not records:
            return
        offset = self.offline["frame_offset"]
        current = interval_at(records, self.frame + offset)
        order = ([current] if current is not None else []) + [i for i in range(len(records)) if i != current]
        for index in order:
            self._queue_offline_snapshot(index)

    def _activate_offline(self, index):
        key = self._offline_key(index)
        if key not in self.auto_cache:
            return False
        self.auto_cache.move_to_end(key)
        ((self.snapshot, path),) = self.auto_cache[key]
        if self.offline_selected == index:
            return True
        self.offline_selected = index
        self.configure_overlay()
        self.replay.configure_overlay(field_preview=str(path))
        self._buttons()
        return True

    def _ensure_offline(self):
        from openpi.datasets.libero.field_preview_sequence import interval_at

        result, offset = self.offline["result"], self.offline["frame_offset"]
        index = interval_at(result["records"], self.frame + offset)
        if index is None:
            return
        self.native.update_frame(self.frame + offset, index)
        record = result["records"][index]
        self.choice.set(f"已保存场 {index + 1}/{len(result['records'])}")
        status = "通过" if record["accepted"] else "未通过"
        reason_labels = {
            "position_maximum": "位置误差超限",
            "rotation_maximum": "姿态误差超限",
            "nominal_arrival_timeout": "到达时间超限",
            "orientation_log_chart": "姿态超出稳定证书区域",
            "angular_speed": "角速度超限",
        }
        reasons = "、".join(reason_labels.get(reason, reason) for reason in record["report"].get("reasons", []))
        reason_text = f" · 原因：{reasons}" if reasons else ""
        scale = result["limits"].get("time_scale", 1)
        timing_text = (
            f"示教时间放慢 {scale} 倍，控制周期保持 {result['limits']['dt']:.3f}s；" if scale > 1 else "源时间"
        )
        speed_text = ""
        if self.episode.processed_root is not None:
            cap = record["maximum_speed_scaled_per_second"] * result["tolerance"][0]
            speed_text = (
                f" · 本场限速 {cap:.3f} m/s · 整回合 RMS {result['summary']['position_rms_native'] * 1000:.2f} mm"
            )
        from openpi.datasets.dynamics.playback import record_quality

        quality = record_quality(self.offline, index)
        quality_text = (
            "长度合格"
            if quality["eligible_length"]
            else "短命令事件，仅诊断"
            if quality["short_command_event"]
            else "过短，仅诊断"
        )
        length_text = f"原始 {quality['transitions']} 个转移 / {quality['source_seconds']:.2f}s · {quality_text}"
        model = "高斯门控稳定场" if "field" in record else "离线线性稳定场"
        goal_text = ""
        if "goal_source_index" in record:
            ahead = record["goal_source_index"] - record["end"]
            goal_text = f" · 过点目标前视 {ahead} 帧，按源边界切换" if ahead else " · 原末端目标（停止/终点或前视无改善）"
            if result["summary"].get("flow_fallback"):
                goal_text += " · 整回合回退原终点方案"
        self.detail.configure(
            text=f"{model} · {result.get('profile_label', '回放配置')} · {length_text} · 当前场验收{status}{reason_text} · 整回合覆盖 {result['summary']['actions_covered']} 个观测转移 · 当前段最大误差 {record['report'].get('maximum_scaled_error', result['summary']['maximum_scaled_error']):.3f} 倍阈值{speed_text}{goal_text}\n{result['coordinates']['space']} · {result['coordinates']['units']} · {timing_text}状态连续切换，无重置；速度衔接需另行复核；物理执行尚未验证。"
        )
        self.play_button.configure(state="normal")
        if self.episode.processed_root is None:
            return
        if not self._activate_offline(index):
            self._queue_offline_snapshot(index)
            if "后台预取" not in self.detail.cget("text"):
                self.detail.configure(text=self.detail.cget("text") + " · 当前场正在后台预取，播放不会暂停")

    def _buttons(self):
        state = "disabled" if self.busy else "normal"
        if self.episode is not None and self.episode.processed_root is None:
            state = "disabled"
        self.fit_button.configure(state=state)
        self.load_button.configure(state="normal" if self.episode is not None and not self.busy else "disabled")
        self.choice.configure(state="disabled" if self.busy else "readonly")
        self.profile_choice.configure(state="disabled" if self.busy else "readonly")
        self.export_button.configure(state="normal" if self.snapshot and not self.busy else "disabled")
        self.focus_button.configure(state="normal" if self.snapshot else "disabled")
        self.play_button.configure(state="normal" if (self.snapshot or self.offline) and not self.busy else "disabled")

    def _submit(self, kind, callback):
        self.job += 1
        generation, job = self.generation, self.job
        self.busy = True
        self._buttons()

        def work():
            try:
                self.events.put((generation, job, kind, callback(), None))
            except Exception as error:
                self.events.put((generation, job, kind, None, str(error)))

        self.executor.submit(work)

    def fit_current(self):
        if self.busy or not self.segmentation or self.episode is None:
            return
        from openpi.datasets.libero.vector_field_inspection import preview

        self.automatic.set(False)
        intervals = self.segmentation["intervals"]
        index = next(
            (i for i, item in enumerate(intervals) if item["start"] <= self.frame < item["end"]), len(intervals) - 1
        )
        episode = self.episode
        speed = float(self.max_speed.get())
        representation = "fitted" if self.representation.get() == "SEDS" else "linear"
        self.choice.set(f"当前命令段 {index + 1}（新拟合）")
        self.detail.configure(text=f"正在拟合第 {index + 1} 个命令段… 可以继续拖动时间轴和调整视角。")
        self._submit(
            "preview",
            lambda: preview(
                episode.processed_root,
                episode.processed_index,
                interval_index=index,
                max_speed=speed,
                representation=representation,
            ),
        )

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="载入当前回合的动力系统拟合结果", filetypes=[("拟合回合 JSON", "*.json")]
        )
        if path:
            self.load_file(path)

    def load_file(self, path):
        if self.busy or self.episode is None:
            return
        from openpi.datasets.libero.reader import ProcessedLiberoDataset
        from openpi.datasets.libero.vector_field_inspection import load_result

        self.automatic.set(False)
        episode = self.episode
        self.detail.configure(text="正在核对拟合结果与当前示教来源…")
        def work():
            from openpi.datasets.dynamics.playback import load_saved_episode

            if json.loads(Path(path).read_text()).get("schema") == "vlds.dynamics.episode.v1":
                return "dynamics", load_saved_episode(episode, path)
            return "fields", (
                Path(path),
                load_result(ProcessedLiberoDataset(episode.processed_root), episode.processed_index, path)[0],
            )

        self._submit("file", work)

    def show_record(self):
        if self.busy or not self.choices or self.choice.current() < 0:
            return
        from openpi.datasets.libero.vector_field_inspection import preview

        self.automatic.set(False)
        record_index = self.choices[self.choice.current()]
        episode, path = self.episode, self.fields_path
        self.detail.configure(text="正在对已保存的固定场进行名义积分…")
        self._submit(
            "preview",
            lambda: preview(
                episode.processed_root, episode.processed_index, fields_path=path, record_index=record_index
            ),
        )

    def configure_overlay(self):
        self.native.configure_field(visible=self.visible.get(), vectors=self.vectors.get(), rollout=self.rollout.get())
        self.replay.configure_overlay(
            field_visible=self.visible.get(), field_vectors=self.vectors.get(), field_rollout=self.rollout.get()
        )

    def _describe(self):
        if self.busy:
            return
        arrays, info = self.snapshot
        report, interval = info["report"], info["interval"]
        start, end = interval["start"], interval["end"]
        reasons = {
            "position_rms": "位置轨迹误差超限",
            "rotation_rms": "姿态轨迹误差超限",
            "nominal_arrival_timeout": "规定时间内未到达目标",
        }
        outcome = (
            "名义验收通过"
            if report["passed"]
            else "名义验收未通过：" + "、".join(reasons.get(reason, reason) for reason in report["reasons"])
        )
        now = "当前帧在该场区间外，三维叠加已隐藏"
        if start <= self.frame <= end:
            now = f"当前同步位置误差 {arrays['position_error'][self.frame - start] * 1000:.1f} mm"
        continuous = info.get("continuous_report")
        sequence_text = ""
        if continuous:
            status = "通过" if continuous["time_aligned_passed"] else "未通过"
            sequence_text = f"；本命令段 {continuous['fields']} 场连续切换位置 RMS {continuous['position_rms_m'] * 1000:.1f} mm（{status}，按源时刻切换）"
        fitting = info.get("fit") or {}
        kind = fitting.get("fit", fitting).get("kind")
        model = {
            "short_linear": "短段线性吸引子",
            "linear_baseline": "线性吸引子",
            "analytic_hold": "保持场",
            "seds": "SEDS 约束场",
            "rollout_linear": "轨迹拟合线性稳定场",
            "gaussian_contractive": "高斯门控稳定场",
        }.get(kind, "固定场")
        source = "成功回放参考轨迹" if info.get("teacher_source") else "原始示教"
        self.detail.configure(
            text=f"[{start}, {end}] · {source} · {model} · {outcome} · 位置 RMS {report['position_rms_m'] * 1000:.1f} mm / 姿态 RMS {np.degrees(report['rotation_rms_rad']):.1f}°\n橙：示教 · 紫：名义积分 · 绿：目标 · 箭头：位置场速度，限速 {info['control']['max_speed']:.1f} m/s · {now}{sequence_text}；不代表物理执行成功。"
        )

    def play_interval(self):
        if self.offline and self.automatic.get():
            from openpi.datasets.libero.field_preview_sequence import interval_at

            records = self.offline["result"]["records"]
            offset = self.offline["frame_offset"]
            interval = records[interval_at(records, self.frame + offset)]
            self.play(max(0, interval["start"] - offset), interval["end"] - offset)
        elif self.snapshot is not None:
            interval = self.snapshot[1]["interval"]
            self.play(interval["start"], interval["end"])

    def focus(self):
        if self.snapshot is not None:
            grid = self.snapshot[0]["grid"]
            self.replay.send(
                "view",
                action="focus",
                lookat=((grid.min(axis=0) + grid.max(axis=0)) / 2).tolist(),
                distance=max(0.35, float(np.ptp(grid, axis=0).max()) * 2.5),
            )

    def export(self):
        if self.snapshot is None or self.busy:
            return
        from openpi.fields.visualization import plot_field

        path = filedialog.asksaveasfilename(
            title="导出动力系统场诊断", defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")]
        )
        if path:
            arrays, info = self.snapshot
            self._submit("exported", lambda: plot_field(path, arrays, info))

    def poll(self):
        if self.closed:
            return
        try:
            while True:
                generation, job, kind, value, error = self.events.get_nowait()
                if generation != self.generation:
                    continue
                if kind == "offline_snapshot":
                    key = value[0]
                    if key not in self.offline_pending:
                        if len(value) > 3 and isinstance(value[3], Path):
                            value[3].unlink(missing_ok=True)
                        continue
                    self.offline_pending.discard(key)
                    if error:
                        if self.offline is not None and key[1] == self.offline.get("path"):
                            self.detail.configure(text="场预取失败：" + error + "；保留当前场。")
                        continue
                    _, index, snapshot, path = value
                    self.auto_cache[key] = [(snapshot, path)]
                    while len(self.auto_cache) > 64:
                        _, removed = self.auto_cache.popitem(last=False)
                        for _, old_path in removed:
                            old_path.unlink(missing_ok=True)
                    if self.offline is not None and key[1] == self.offline.get("path"):
                        self._ensure_offline()
                    continue
                if job != self.job:
                    continue
                self.busy = False
                if kind == "file" and not error:
                    family, value = value
                    if family == "dynamics":
                        kind, value = "profile", (value["profile"], value)
                    else:
                        kind = "loaded"
                if error:
                    if kind == "automatic":
                        self.auto_failed.add(self.auto_request_key)
                    if kind == "profile":
                        label = next(
                            (name for name, code in self.profile_codes.items() if code == self.profile_code),
                            "回放配置",
                        )
                        self.profile.set(label)
                    self.detail.configure(text="场诊断失败：" + error + ("；保留上一场。" if self.snapshot else ""))
                elif kind == "loaded":
                    self.fields_path, self.result = value
                    values, self.choices = [], []
                    ordered = list(dict.fromkeys(i for seq in self.result["sequences"] for i in seq["record_indices"]))
                    ordered.extend(i for i in range(len(self.result["records"])) if i not in ordered)
                    for i in ordered:
                        record = self.result["records"][i]
                        if record.get("primitive"):
                            candidate = record["candidate"]
                            selected = [
                                str(j + 1)
                                for j, seq in enumerate(self.result["sequences"])
                                if i in seq["record_indices"]
                            ]
                            tag = "方案 " + "/".join(selected) if selected else "候选"
                            status = "通过" if record["accepted"] else "未通过"
                            self.choices.append(i)
                            values.append(f"{tag} · 场 {i + 1} · [{candidate['start']}, {candidate['end']}] · {status}")
                    self.choice["values"] = values
                    if values:
                        first = (
                            self.result["sequences"][0]["record_indices"][0]
                            if self.result["sequences"]
                            else self.choices[0]
                        )
                        self.choice.current(self.choices.index(first))
                        self.show_record()
                    else:
                        self.detail.configure(text="此结果没有可视化所需的有效场参数。")
                elif kind == "profile":
                    from openpi.datasets.dynamics.playback import segmentation

                    code, bundle = value
                    self.automatic.set(True)
                    self.profile_code = code
                    self.offline = bundle
                    self._set_profile_choices()
                    self.segmentation = segmentation(bundle)
                    self.offline_selected = None
                    self.snapshot = None
                    self.offline_pending.clear()
                    self.native.set_bundle(bundle)
                    self.replay.configure_overlay(intervals=self.segmentation["intervals"], field_preview=None)
                    if self.on_segmentation:
                        self.on_segmentation(self.segmentation)
                    self._ensure_offline()
                    self._prefetch_offline()
                elif kind == "automatic":
                    from openpi.fields.visualization import save_snapshot

                    key, values = value
                    entries = []
                    for i, snapshot in enumerate(values):
                        path = Path(self.temporary.name) / f"auto-{generation}-{job}-{i}.npz"
                        save_snapshot(path, *snapshot)
                        entries.append((snapshot, path))
                    self.auto_cache[key] = entries
                    while len(self.auto_cache) > 32:
                        _, removed = self.auto_cache.popitem(last=False)
                        for _, path in removed:
                            path.unlink(missing_ok=True)
                    self._ensure_automatic()
                    if self.snapshot:
                        self._describe()
                elif kind == "preview":
                    from openpi.fields.visualization import save_snapshot

                    self.snapshot = value
                    arrays, info = value
                    path = Path(self.temporary.name) / f"preview-{generation}-{job}.npz"
                    save_snapshot(path, arrays, info)
                    self.configure_overlay()
                    self.replay.configure_overlay(field_preview=str(path))
                    self.seek(info["interval"]["start"])
                    self._describe()
                elif kind == "exported":
                    self.detail.configure(
                        text=f"已导出 {value} 及同名 JSON/NPZ；三维箭头只表示位置场，曲线为名义积分。"
                    )
                self._buttons()
        except queue.Empty:
            pass
        self.after(80, self.poll)

    def close(self):
        self.closed = True
        self.generation += 1
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.temporary.cleanup()
