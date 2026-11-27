"""Which network routes does this process have? One line per attempt.

    python3 net_probe.py [--http-port P] [--unix PATH] [--abstract NAME]
                         [--inherited-fd FD]

Standard library only, x86_64 Linux. Used by iso_check.sh to test a seccomp
filter (an Apptainer profile or noinet.py) from inside.
"""

import argparse
import ctypes
import mmap
import os
import socket
import subprocess
import sys

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long


def raw_socket(nr, family, kind=1):
    r = libc.syscall(ctypes.c_long(nr), ctypes.c_long(family), ctypes.c_long(kind), 0)
    if r >= 0:
        os.close(r)
        return "created"
    return f"refused: {os.strerror(ctypes.get_errno())}"


def i386_socket():
    """socket(AF_INET, SOCK_STREAM, 0) through the 32-bit entry (int 0x80,
    i386 syscall 359); run in a child, which a filter may kill."""
    code = bytes(
        [0x53]  # push rbx
        + [0xB8]
        + list((359).to_bytes(4, "little"))  # mov eax, 359
        + [0xBB, 2, 0, 0, 0]  # mov ebx, AF_INET
        + [0xB9, 1, 0, 0, 0]  # mov ecx, SOCK_STREAM
        + [0x31, 0xD2]  # xor edx, edx
        + [0xCD, 0x80]  # int 0x80
        + [0x5B, 0xC3]  # pop rbx; ret
    )
    buf = mmap.mmap(
        -1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC
    )
    buf.write(code)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    r = ctypes.CFUNCTYPE(ctypes.c_int)(addr)()
    print("created" if r >= 0 else f"refused: {os.strerror(-r)}")


def attempt(name, fn):
    try:
        print(f"{name}: {fn()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{name}: refused: {e}", flush=True)


def http(sock):
    sock.sendall(b"GET / HTTP/1.0\r\nHost: x\r\n\r\n")
    return "answered " + sock.recv(64).split(b"\r\n")[0].decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--http-port", type=int)
    p.add_argument("--unix")
    p.add_argument("--abstract")
    p.add_argument("--inherited-fd", type=int)
    a = p.parse_args()
    for fam in ("AF_INET", "AF_INET6", "AF_NETLINK", "AF_PACKET"):
        kind = socket.SOCK_DGRAM if fam != "AF_INET6" else socket.SOCK_STREAM
        attempt(
            f"socket({fam})",
            lambda: (socket.socket(getattr(socket, fam), kind).close(), "created")[1],
        )
    attempt(
        "socket(AF_UNIX)", lambda: (socket.socket(socket.AF_UNIX).close(), "created")[1]
    )
    attempt(
        "socketpair(AF_UNIX)",
        lambda: [s.close() for s in socket.socketpair()] and "created",
    )
    attempt(
        "raw socket(0x100000002) (AF_INET with high bits)",
        lambda: raw_socket(41, 0x100000002),
    )
    # in children: a filter may kill the process for a foreign-ABI syscall
    here = os.path.dirname(os.path.abspath(__file__))
    for name, call in (
        ("x32 socket(AF_INET)", "print(net_probe.raw_socket(0x40000000 | 41, 2))"),
        ("i386 int 0x80 socket(AF_INET)", "net_probe.i386_socket()"),
    ):
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {here!r}); import net_probe; {call}",
            ],
            capture_output=True,
            text=True,
        )
        got = out.stdout.strip() or f"no result, the child ended with {out.returncode}"
        print(f"{name}: {got}", flush=True)
    try:
        with open("/proc/sys/kernel/io_uring_disabled") as f:
            print(f"io_uring_disabled (kernel setting): {f.read().strip()}")
    except OSError:
        pass
    if a.http_port:
        attempt(
            f"TCP 127.0.0.1:{a.http_port}",
            lambda: http(
                socket.create_connection(("127.0.0.1", a.http_port), timeout=5)
            ),
        )
    if a.unix:

        def unix():
            s = socket.socket(socket.AF_UNIX)
            s.connect(a.unix)
            return http(s)

        attempt(f"Unix socket {a.unix}", unix)
    if a.abstract:

        def abstract():
            s = socket.socket(socket.AF_UNIX)
            s.connect("\0" + a.abstract)
            return http(s)

        attempt("abstract Unix socket of the host", abstract)
    if a.inherited_fd is not None:
        attempt(
            "inherited TCP socket",
            lambda: http(socket.socket(fileno=a.inherited_fd)),
        )


if __name__ == "__main__":
    main()
