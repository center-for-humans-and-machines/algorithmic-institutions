"""A Git LFS pointer must never be parsed as if it were the table.

`*.csv`, `*.parquet` and `*.pt` are LFS-tracked here, so a checkout without
`git lfs pull` leaves a 130-byte text stub where the data should be. Pandas
reads one without complaint and everything downstream is plausible nonsense,
which is much worse than a crash.
"""

import pytest

from aimanager.llm_manager.lfs import assert_real_file, is_lfs_pointer

POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:b80ff619b6b8c565a9e20c197ad5885adb958e3fde8dc7a0c37c25f9c66faec8\n"
    "size 5094\n"
)


def test_a_pointer_is_recognised_and_refused(tmp_path):
    p = tmp_path / "validation_design.csv"
    p.write_text(POINTER)
    assert is_lfs_pointer(p)
    with pytest.raises(AssertionError, match="LFS pointer"):
        assert_real_file(p, "design")


def test_real_data_passes(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("name,pool\nthr9_p10,62.32\n")
    assert not is_lfs_pointer(p)
    assert assert_real_file(p) == p


def test_missing_and_empty_are_refused_separately(tmp_path):
    with pytest.raises(AssertionError, match="not found"):
        assert_real_file(tmp_path / "nope.csv")
    e = tmp_path / "e.csv"
    e.write_bytes(b"")
    with pytest.raises(AssertionError, match="empty"):
        assert_real_file(e)
