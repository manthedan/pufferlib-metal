#!/usr/bin/env python3
"""Fingerprint the root trainer before/after each gate, including replacement."""
import hashlib
import json
from pathlib import Path
import stat
import sys


def snapshot(path):
    path = Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    result = {key: getattr(info, key) for key in
              ('st_mode', 'st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}
    if stat.S_ISLNK(info.st_mode):
        result['link'] = str(path.readlink())
    # Follow a preexisting symlink too: modifying its target must be detected.
    if path.is_file():
        with path.open('rb') as f:
            sha = hashlib.sha256()
            for block in iter(lambda: f.read(1024 * 1024), b''):
                sha.update(block)
        result['sha256'] = sha.hexdigest()
        target = path.stat()
        result['target'] = [target.st_ino, target.st_mtime_ns, target.st_ctime_ns]
    return result


if __name__ == '__main__':
    print(json.dumps(snapshot(sys.argv[1]), sort_keys=True))
