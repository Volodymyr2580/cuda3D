#!/usr/bin/env python3
import os
import posixpath
import sys

import paramiko


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


def main() -> int:
    if len(sys.argv) < 6 or (len(sys.argv) - 4) % 2:
        print("usage: remote_put.py HOST PORT USER LOCAL REMOTE [LOCAL REMOTE ...]", file=sys.stderr)
        return 2
    password = os.environ.get("CUDA3D_REMOTE_PASS")
    if not password:
        print("CUDA3D_REMOTE_PASS is not set", file=sys.stderr)
        return 2
    host, port, user = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    pairs = list(zip(sys.argv[4::2], sys.argv[5::2]))

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=20)
    sftp = client.open_sftp()
    for local, remote in pairs:
        ensure_dir(sftp, posixpath.dirname(remote))
        sftp.put(local, remote)
        print(f"PUT {local} -> {remote}")
    sftp.close()
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
