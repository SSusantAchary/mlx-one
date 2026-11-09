"""File integrity checks must include all bytes across streaming boundaries."""

import hashlib

import pytest

from mlx_one.utils.hashing import sha256_file


@pytest.mark.parametrize("payload", [b"", b"binary\x00\xff", bytes(range(256)) * 4097])
def test_streamed_file_checksum_matches_full_buffer_oracle(tmp_path, payload):
    path = tmp_path / "asset.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_missing_asset_cannot_receive_a_checksum(tmp_path):
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing.bin")
