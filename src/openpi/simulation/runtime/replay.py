"""Process isolation for optional recorded-state replay."""

# ruff: noqa: RUF001
import json
import queue
import subprocess
import threading

from openpi_client.simulation_frames import FrameBuffer

from openpi.datasets.libero import SEGMENT_COLORS
from openpi.simulation import paths
from openpi.simulation.runtime.controller import InspectorController
from openpi.simulation.runtime.controller import simulation_environment


class DatasetReplay:
    def __init__(self):
        self.process = None
        self.frames = None
        self.process_generation = None
        self.events = queue.Queue()
        self.generation = 0
        self.lock = threading.RLock()
        self.closed = False
        self.source = None
        self.overlay_options = {"visible": True, "current_only": False, "markers": True}

    def open(self, episode, *, frame=0):
        if episode.replay_source and episode.replay_source.frame(frame) is None:
            self.events.put((self.generation, {"error": "此观测没有对应的完整仿真状态。"}))
            return
        self.generation += 1
        generation = self.generation

        def launch():
            with self.lock:
                if generation != self.generation or self.closed:
                    return
                self._stop()
                interpreter = paths.ROOT / "examples/libero/.venv/bin/python"
                if not interpreter.is_file():
                    self.events.put((generation, {"error": "三维回放需要先安装 LIBERO 仿真环境。"}))
                    return
                self.source = episode.replay_source
                source_path, source_group = self._location(episode)
                self.frames = FrameBuffer()
                command = [
                    str(interpreter),
                    str(paths.WORKERS_ROOT / "replay.py"),
                    "--render-buffer",
                    str(self.frames.path),
                    "--path",
                    str(source_path),
                    "--group",
                    source_group,
                    "--frame",
                    str(self._source_frame(frame)),
                    "--timeline-frame",
                    str(frame),
                ]
                overlay = self._overlay(episode)
                if overlay is not None:
                    command.extend(["--overlay", json.dumps(overlay)])
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=paths.ROOT,
                        env=simulation_environment({"dataset": "libero", "software_rendering": True}),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        start_new_session=True,
                    )
                except OSError as error:
                    self.frames.close()
                    self.frames = None
                    self.events.put((generation, {"error": str(error)}))
                    return
                self.process = process
                self.process_generation = generation
            recent = []
            for line in process.stdout:
                if line.startswith("@dataset "):
                    self.events.put((generation, json.loads(line[len("@dataset ") :])))
                else:
                    recent.append(line.strip())
                    recent = recent[-5:]
            process.stdout.close()
            code = process.wait()
            if code:
                self.events.put((generation, {"error": "三维回放已退出：" + "\n".join(recent)}))
            self.events.put((generation, {"stopped": True}))

        threading.Thread(target=launch, daemon=True).start()

    def _overlay(self, episode):
        if episode.processed_root is None:
            return None
        return {
            "path": str(episode.path),
            "group": episode.group,
            "colors": SEGMENT_COLORS,
            **self.overlay_options,
        }

    def configure_overlay(self, **options):
        self.overlay_options.update(options)
        self.send("overlay", **self.overlay_options)

    @staticmethod
    def _location(episode):
        source = episode.replay_source
        return (source.path, source.group) if source else (episode.path, episode.group)

    def _source_frame(self, frame):
        source_frame = self.source.frame(frame) if self.source else frame
        if source_frame is None:
            raise ValueError("此观测没有对应的完整仿真状态，无法打开三维回放。")
        return source_frame

    def reload(self, episode, *, frame=0):
        if episode.replay_source and episode.replay_source.frame(frame) is None:
            self.events.put((self.generation, {"error": "此观测没有对应的完整仿真状态。"}))
            return
        with self.lock:
            if self.process is None or self.process.poll() is not None or self.process_generation != self.generation:
                self.open(episode, frame=frame)
                return
            self.source = episode.replay_source
            path, group = self._location(episode)
            self.send(
                "reload",
                path=str(path),
                group=group,
                frame=self._source_frame(frame),
                timeline_frame=frame,
                overlay=self._overlay(episode),
            )

    def send(self, command, **values):
        if not self.lock.acquire(blocking=False):
            return
        try:
            if self.process is not None and self.process.poll() is None:
                if command == "frame":
                    values["timeline_frame"] = values["frame"]
                    values["frame"] = self._source_frame(values["frame"])
                try:
                    self.process.stdin.write(json.dumps({"command": command, **values}) + "\n")
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
        finally:
            self.lock.release()

    def read_frame(self):
        frames = self.frames
        return frames.read() if frames is not None else None

    def _stop(self):
        process, self.process = self.process, None
        self.process_generation = None
        InspectorController._terminate(process, worker=True)  # noqa: SLF001 -- Shared owned-process cleanup.
        if self.frames is not None:
            self.frames.close()
            self.frames = None

    def stop(self):
        self.generation += 1
        generation = self.generation

        def shutdown():
            with self.lock:
                if generation == self.generation:
                    self._stop()

        threading.Thread(target=shutdown, daemon=True).start()

    def close(self):
        self.closed = True
        self.generation += 1
        with self.lock:
            self._stop()
