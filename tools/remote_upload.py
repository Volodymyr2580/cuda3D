#!/usr/bin/env python3
import os
import posixpath
import stat
import sys
import time

import paramiko


SKIP_DIRS = {".git", "__pycache__"}


def ensure_dir(sftp, path):
    parts = []
    cur = path
    while cur not in ("", "/"):
        parts.append(cur)
        cur = posixpath.dirname(cur)
    for item in reversed(parts):
        try:
            sftp.stat(item)
        except FileNotFoundError:
            sftp.mkdir(item)


def upload_tree(sftp, local_root, remote_root):
    ensure_dir(sftp, remote_root)
    total_bytes = 0
    total_files = 0
    started = time.time()
    for root, dirs, files in os.walk(local_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, local_root)
        rel = "" if rel == "." else rel.replace(os.sep, "/")
        remote_dir = remote_root if not rel else posixpath.join(remote_root, rel)
        ensure_dir(sftp, remote_dir)
        for name in files:
            local_path = os.path.join(root, name)
            remote_path = posixpath.join(remote_dir, name)
            size = os.path.getsize(local_path)
            sftp.put(local_path, remote_path)
            total_files += 1
            total_bytes += size
            if total_files % 10 == 0 or size > 128 * 1024 * 1024:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"uploaded files={total_files} bytes={total_bytes} "
                    f"rate={total_bytes / elapsed / 1024 / 1024:.2f} MiB/s last={rel}/{name}",
                    flush=True,
                )
    return total_files, total_bytes


def main() -> int:
    if len(sys.argv) != 6:
        print("usage: remote_upload.py HOST PORT USER LOCAL_ROOT REMOTE_ROOT", file=sys.stderr)
        return 2
    host = sys.argv[1]
    port = int(sys.argv[2])
    user = sys.argv[3]
    local_root = sys.argv[4]
    remote_root = sys.argv[5]
    password = os.environ.get("CUDA3D_REMOTE_PASS")
    if not password:
        print("CUDA3D_REMOTE_PASS is not set", file=sys.stderr)
        return 2

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=20)
    sftp = client.open_sftp()
    files, bytes_written = upload_tree(sftp, local_root, remote_root)
    sftp.close()
    client.close()
    print(f"DONE files={files} bytes={bytes_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
