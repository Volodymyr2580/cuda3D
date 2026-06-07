#!/usr/bin/env python3
import os
import sys

import paramiko


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: remote_exec.py HOST PORT USER COMMAND", file=sys.stderr)
        return 2
    host = sys.argv[1]
    port = int(sys.argv[2])
    user = sys.argv[3]
    command = sys.stdin.read() if sys.argv[4] == "-" else sys.argv[4]
    command = command.replace("\r\n", "\n").replace("\r", "").lstrip("\ufeff")
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
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    stdin, stdout, stderr = client.exec_command(command, timeout=300)
    remote_stdin = os.environ.get("CUDA3D_REMOTE_STDIN")
    if remote_stdin is not None:
        stdin.write(remote_stdin)
        stdin.flush()
        stdin.channel.shutdown_write()
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if out:
        print(out, end="")
    if err:
        print(err, end="", file=sys.stderr)
    client.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
