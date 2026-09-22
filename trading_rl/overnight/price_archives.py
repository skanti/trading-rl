"""Bounded, run-scoped decoding of immutable price-archive inputs."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np


class PriceArchive:
    def __init__(self, path: Path):
        self._archive = np.load(path, allow_pickle=False)
        self.files = self._archive.files
        self._arrays = {}
        self._derived = {}

    def __getitem__(self, name):
        if name not in self._arrays:
            array = self._archive[name]
            array.setflags(write=False)
            self._arrays[name] = array
        return self._arrays[name]

    def derived(self, name, build):
        """Reuse parsing, never selections or validation against session targets."""
        if name not in self._derived:
            self._derived[name] = build()
        return self._derived[name]

    def close(self):
        self._archive.close()
        self._arrays.clear()
        self._derived.clear()


class _ArchiveCache:
    def __init__(self, max_archives):
        self.max_archives = max_archives
        self.entries = OrderedDict()

    def get(self, path):
        path = path.resolve()
        stat = path.stat()
        identity = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        prior = self.entries.pop(path, None)
        if prior is not None:
            old_identity, archive = prior
            if old_identity == identity:
                self.entries[path] = prior
                return archive
            archive.close()
        while len(self.entries) >= self.max_archives:
            _, (_, archive) = self.entries.popitem(last=False)
            archive.close()
        archive = PriceArchive(path)
        self.entries[path] = (identity, archive)
        return archive

    def close(self):
        for _, archive in self.entries.values():
            archive.close()
        self.entries.clear()


_ACTIVE_CACHE = ContextVar("price_archive_cache", default=None)


@contextmanager
def cache_price_archives(max_archives=4):
    """Reuse at most four NPZ files until this invocation exits, including on error."""
    if max_archives < 1:
        raise ValueError("max_archives must be positive")
    cache = _ArchiveCache(max_archives)
    token = _ACTIVE_CACHE.set(cache)
    try:
        yield
    finally:
        _ACTIVE_CACHE.reset(token)
        cache.close()


@contextmanager
def open_price_archive(path: Path):
    cache = _ACTIVE_CACHE.get()
    if cache is not None:
        yield cache.get(path)
    else:
        archive = PriceArchive(path)
        try:
            yield archive
        finally:
            archive.close()
