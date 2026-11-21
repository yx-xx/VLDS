"""Process isolation for optional recorded-state replay."""

# ruff: noqa: RUF001
import json
import queue
import subprocess
import threading

from openpi.simulation import paths
from openpi.simulation.runtime.controller import InspectorController
from openpi.simulation.runtime.controller import simulation_environment


class DatasetReplay:
    def __init__(self):
        self.process = None
        self.process_generation = None
        self.events = queue.Queue()
        self.generation = 0
        self.lock = threading.RLock()
        self.closed = False
        self.source = None

    def open(self, episode, *, frame=0):
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
                command = [
                    str(interpreter),
                    str(paths.WORKERS_ROOT / "replay.py"),
                    "--path",
                    str(source_path),
                    "--group",
                    source_group,
                    "--frame",
                    str(self._source_frame(frame)),
                ]
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

    @staticmethod
    def _location(episode):
        source = episode.replay_source
        return (source.path, source.group) if source else (episode.path, episode.group)

    def _source_frame(self, frame):
        return self.source.frame(frame) if self.source else frame

    def reload(self, episode, *, frame=0):
        with self.lock:
            if self.process is None or self.process.poll() is not None or self.process_generation != self.generation:
                self.open(episode, frame=frame)
                return
            self.source = episode.replay_source
            path, group = self._location(episode)
            self.send("reload", path=str(path), group=group, frame=self._source_frame(frame))

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

    def _stop(self):
        process, self.process = self.process, None
        self.process_generation = None
        InspectorController._terminate(process, worker=True)  # noqa: SLF001 -- Shared owned-process cleanup.

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
