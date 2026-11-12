import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import relay_wireseg_blobs as relay


def blob(payload, digest=None):
    return relay.Blob("depth/depth-00019-of-00100.tar.gz", len(payload),
                      digest or hashlib.sha256(payload).hexdigest())


class Partial(io.BytesIO):
    status = 206

    def __init__(self, data, content_range):
        super().__init__(data)
        self.headers = {"Content-Range": content_range}


def server(payload, requests):
    def open_range(request, timeout):
        start, stop = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        requests.append((start, stop))
        return Partial(payload[start:stop + 1], f"bytes {start}-{stop}/{len(payload)}")
    return open_range


def test_resume_revalidates_receipts_and_only_fetches_missing_ranges(tmp_path):
    payload = b"abcdefg"
    item = blob(payload)
    seen = []
    fetch = server(payload, seen)
    first = relay.fetch_blob(tmp_path, item, chunk_size=3, max_new_parts=1, open_range=fetch)
    assert first["verified"] is False
    assert not (tmp_path / item.name).exists()
    second = relay.fetch_blob(tmp_path, item, chunk_size=3, max_new_parts=2, open_range=fetch)
    assert second["verified"] is True
    assert seen == [(0, 2), (3, 5), (6, 6)]
    assert relay.verify_blob(tmp_path / item.name, item)
    assert (tmp_path / item.name).read_bytes() == payload


def test_wrong_content_range_keeps_blob_out_of_dataset(tmp_path):
    item = blob(b"abcdef")
    def misrouted(*args, **kwargs):
        return Partial(b"abc", "bytes 3-5/6")
    with pytest.raises(ValueError, match="Content-Range"):
        relay.fetch_blob(tmp_path, item, chunk_size=3, open_range=misrouted)
    assert not (tmp_path / item.name).exists()


def test_wrong_final_lfs_sha_never_promotes_assembled_blob(tmp_path):
    payload = b"abcdef"
    item = blob(payload, digest="a" * 64)
    with pytest.raises(ValueError, match="full LFS"):
        relay.fetch_blob(tmp_path, item, chunk_size=3, open_range=server(payload, []))
    assert not (tmp_path / item.name).exists()
    assert len(list((tmp_path / ".wireseg-ranges").rglob("*.json"))) == 2


def test_tampered_cached_chunk_is_not_reused(tmp_path):
    item = blob(b"abcdef")
    fetch = server(b"abcdef", [])
    relay.fetch_blob(tmp_path, item, chunk_size=3, max_new_parts=1, open_range=fetch)
    part, = (tmp_path / ".wireseg-ranges").rglob("*.part")
    part.write_bytes(b"xxx")
    with pytest.raises(ValueError, match="cached chunk"):
        relay.fetch_blob(tmp_path, item, chunk_size=3, open_range=fetch)
    assert not (tmp_path / item.name).exists()


def test_catalogue_rejects_duplicate_identities_even_with_claimed_revision(tmp_path):
    item = blob(b"data")
    path = tmp_path / "manifest.json"
    row = {"rfilename": item.name, "size": item.size, "lfs": {"sha256": item.sha256}}
    path.write_text(json.dumps({"id": relay.REPOSITORY, "sha": relay.REVISION,
                                "siblings": [row, row]}))
    with pytest.raises(ValueError, match="duplicate"):
        relay.read_catalogue(path)


def test_verify_enforces_full_size_even_when_sha_is_correct(tmp_path):
    item = blob(b"abcdef")
    path = Path(tmp_path) / "payload"
    path.write_bytes(b"abc")
    assert relay.verify_blob(path, relay.Blob(item.name, 6, hashlib.sha256(b"abc").hexdigest())) is False


def test_retries_one_transient_timeout_for_the_same_range(tmp_path):
    payload = b"abc"
    item = blob(payload)
    seen = []

    def open_range(request, timeout):
        seen.append(request.get_header("Range"))
        if len(seen) == 1:
            raise TimeoutError("interrupted transport")
        return Partial(payload, "bytes 0-2/3")

    result = relay.fetch_blob(tmp_path, item, chunk_size=3, attempts=2, open_range=open_range)
    assert result["verified"] and result["new_parts"] == 1
    assert seen == ["bytes=0-2", "bytes=0-2"]
