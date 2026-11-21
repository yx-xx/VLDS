"""Small file signatures for opt-in refresh of an open dataset."""

from pathlib import Path


def watch_paths(index, episode):
    root = index.root
    paths = {root}
    if root.is_dir():
        paths.update(root / name for name in ("manifest.json", "run.json"))
        paths.update(root / "meta" / name for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "replay.json"))
        paths.update(item.path.parent for item in index.episodes)
    if episode is not None:
        paths.add(episode.path)
        paths.update(episode.videos.values())
        if episode.replay_source:
            paths.add(episode.replay_source.path)
    return tuple(sorted(paths))


def file_signature(paths):
    result = []
    for path in paths:
        try:
            stat = Path(path).stat()
            result.append((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns))
        except OSError:
            result.append((str(path), None, None, None))
    return tuple(result)


class StableFileWatch:
    """Accept atomic replacements immediately; debounce files modified in place."""

    def __init__(self):
        self.current = None
        self.pending = None

    def reset(self):
        self.current = self.pending = None

    def changed(self, signature):
        if self.current is None:
            self.current = signature
        if signature == self.current:
            self.pending = None
            return False
        previous = {row[0]: row[1] for row in self.current}
        if any(
            previous.get(path) is not None and inode is not None and previous[path] != inode
            for path, inode, _size, _mtime in signature
        ):
            self.current = signature
            self.pending = None
            return True
        if signature != self.pending:
            self.pending = signature
            return False
        self.current = signature
        self.pending = None
        return True
