"""Run a command with no way to create network sockets other than AF_UNIX.

    python3 noinet.py <command> [args...]

Installs a seccomp filter on itself (unprivileged: PR_SET_NO_NEW_PRIVS
first), then execs the command; the filter is inherited by everything the
command starts and cannot be removed. Meant as the shell of an agent harness
(the harness keeps its own network, the commands it runs do not). Standard
library only (ctypes), x86_64 Linux.

The filter, an allow-list for socket creation:
- a syscall from any architecture other than x86_64 (i386 `int 0x80`), or
  with the x32 bit set, fails with EPERM
- socket() and socketpair() fail with EPERM unless the family argument is
  exactly AF_UNIX (all 64 bits compared, so a family with high bits set,
  which the kernel would truncate, is refused too)
- io_uring_setup() fails with EPERM (io_uring can create sockets itself)
- everything else is allowed

What it does not do: sockets the process already holds, or receives over a
Unix socket, stay usable; abstract Unix sockets of the host are reachable when
the network namespace is shared (on DAIS it always is).
"""

import ctypes
import os
import struct
import sys

AUDIT_ARCH_X86_64 = 0xC000003E
X32_SYSCALL_BIT = 0x40000000
NR_SOCKET, NR_SOCKETPAIR, NR_IO_URING_SETUP = 41, 53, 425
AF_UNIX = 1
EPERM = 1

LD_W_ABS, JEQ_K, JGE_K, RET_K = 0x20, 0x15, 0x35, 0x06
RET_ALLOW = 0x7FFF0000
RET_EPERM = 0x00050000 | EPERM
PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 38, 22, 2
# offsets in struct seccomp_data: nr, arch, instruction_pointer, args[6]
OFF_NR, OFF_ARCH, OFF_ARG0_LO, OFF_ARG0_HI = 0, 4, 16, 20


def program():
    """The BPF program as (code, jt, jf, k) tuples; jumps are resolved from
    labels ('allow', 'deny', 'check' or None for the next instruction)."""
    raw = [
        ("ld", OFF_ARCH),
        ("jeq", AUDIT_ARCH_X86_64, None, "deny"),
        ("ld", OFF_NR),
        ("jge", X32_SYSCALL_BIT, "deny", None),
        ("jeq", NR_IO_URING_SETUP, "deny", None),
        ("jeq", NR_SOCKET, "check", None),
        ("jeq", NR_SOCKETPAIR, "check", "allow"),
        ("label", "check"),
        ("ld", OFF_ARG0_HI),
        ("jeq", 0, None, "deny"),
        ("ld", OFF_ARG0_LO),
        ("jeq", AF_UNIX, "allow", "deny"),
        ("label", "allow"),
        ("ret", RET_ALLOW),
        ("label", "deny"),
        ("ret", RET_EPERM),
    ]
    labels, pos = {}, 0
    for ins in raw:
        if ins[0] == "label":
            labels[ins[1]] = pos
        else:
            pos += 1
    out, pos = [], 0
    for ins in raw:
        if ins[0] == "label":
            continue
        if ins[0] == "ld":
            out.append((LD_W_ABS, 0, 0, ins[1]))
        elif ins[0] == "ret":
            out.append((RET_K, 0, 0, ins[1]))
        else:
            code = JEQ_K if ins[0] == "jeq" else JGE_K

            def rel(target):
                return 0 if target is None else labels[target] - pos - 1

            out.append((code, rel(ins[2]), rel(ins[3]), ins[1]))
        pos += 1
    return out


def install():
    insns = program()
    buf = b"".join(struct.pack("HBBI", *i) for i in insns)
    filt = ctypes.create_string_buffer(buf, len(buf))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    prog = SockFprog(len(insns), ctypes.cast(filt, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, None, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(prog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    install()
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
