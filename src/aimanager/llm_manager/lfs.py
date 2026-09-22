"""Refuse to parse a Git LFS pointer as if it were data.

`*.csv`, `*.parquet` and `*.pt` are LFS-tracked in this repo, so a checkout
without `git lfs pull` leaves 130-byte text stubs where the tables should be.
`pd.read_csv` parses one happily -- three rows, columns named
`version https://git-lfs.github.com/spec/v1` -- and everything downstream is
plausible nonsense. Every read in this arm goes through `assert_real_file`.
"""

import os

LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(len(LFS_MAGIC)) == LFS_MAGIC


def assert_real_file(path, what="file"):
    """Path exists, is non-empty and is not an unfetched LFS pointer."""
    assert os.path.exists(path), f"{what} not found: {path}"
    size = os.path.getsize(path)
    assert size > 0, f"{what} is empty: {path}"
    assert not is_lfs_pointer(path), (
        f"{what} is an unfetched Git LFS pointer, not data: {path}\n"
        f"run `git lfs pull --include={path}` first"
    )
    return path
