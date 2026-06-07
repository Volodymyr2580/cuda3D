#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import paramiko


def main() -> int:
    if len(sys.argv) < 6 or (len(sys.argv) - 4) % 2 != 0:
        print(
            "usage: remote_get.py host port user remote_path local_path "
            "[remote_path local_path ...]",
            file=sys.stderr,
        )
        return 2

    host = sys.argv[1]
    port = int(sys.argv[2])
    user = sys.argv[3]
    password = os.environ.get("CUDA3D_REMOTE_PASS")
    if not password:
        print("CUDA3D_REMOTE_PASS is not set", file=sys.stderr)
        return 2

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        port=port,
        username=user,
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
    )
    try:
        sftp = client.open_sftp()
        try:
            pairs = sys.argv[4:]
            for remote, local in zip(pairs[0::2], pairs[1::2]):
                target = Path(local)
                target.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(target))
                print(f"GET {remote} -> {target}")
        finally:
            sftp.close()
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
