"""Bounded RGB frame transport shared by the desktop and isolated Python 3.8 workers."""

from __future__ import annotations

import mmap
import pathlib
import struct
import tempfile
import threading

import numpy as np

MAX_WIDTH = 1280
MAX_HEIGHT = 960
_HEADER = struct.Struct("<QII")
_SIZE = _HEADER.size + MAX_WIDTH * MAX_HEIGHT * 3


class FrameBuffer:
    """Keep only the latest frame; an even sequence number publishes a complete write."""

    def __init__(self, path=None):
        self._directory = tempfile.TemporaryDirectory(prefix="vlds-view-") if path is None else None
        self.path = pathlib.Path(self._directory.name) / "frames.rgb" if path is None else pathlib.Path(path)
        if self._directory is not None:
            with self.path.open("wb") as file:
                file.truncate(_SIZE)
        with self.path.open("r+b") as file:
            self._mapping = mmap.mmap(file.fileno(), _SIZE)
        self._lock = threading.Lock()
        self._sequence = 0
        self._seen = 0

    def write(self, image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a uint8 RGB image")
        height, width = image.shape[:2]
        if not 0 < width <= MAX_WIDTH or not 0 < height <= MAX_HEIGHT:
            raise ValueError("Rendered frame exceeds the shared buffer")
        pixels = np.ascontiguousarray(image).tobytes()
        with self._lock:
            if self._mapping is None:
                return
            self._sequence += 2
            _HEADER.pack_into(self._mapping, 0, self._sequence - 1, width, height)
            self._mapping[_HEADER.size : _HEADER.size + len(pixels)] = pixels
            struct.pack_into("<Q", self._mapping, 0, self._sequence)

    def read(self):
        with self._lock:
            if self._mapping is None:
                return None
            sequence, width, height = _HEADER.unpack_from(self._mapping)
            if not sequence or sequence % 2 or sequence == self._seen:
                return None
            if not 0 < width <= MAX_WIDTH or not 0 < height <= MAX_HEIGHT:
                return None
            pixels = self._mapping[_HEADER.size : _HEADER.size + width * height * 3]
            if struct.unpack_from("<Q", self._mapping)[0] != sequence:
                return None
            self._seen = sequence
            return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)

    def close(self):
        with self._lock:
            if self._mapping is not None:
                self._mapping.close()
                self._mapping = None
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None
