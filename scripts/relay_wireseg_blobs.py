"""Fetch and independently verify pinned WireSeg blobs using HTTPS ranges."""

import argparse
from dataclasses import dataclass
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

REPOSITORY = "DeformX/WireSeg-36K"
REVISION = "1dadc86f4e988a015cd9fd02b6dfd16c11c9c40b"
CATALOGUE_SHA256 = "569022f7f5c1985cba7eedb6cd563619a55417001e60f81fabff1cdb5b68f7ac"
IO_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Blob:
    name: str
    size: int
    sha256: str


def read_catalogue(path, expected_sha=CATALOGUE_SHA256):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest["id"] != REPOSITORY or manifest["sha"] != REVISION:
        raise ValueError("HF manifest must identify the pinned WireSeg revision")
    blobs = {}
    for entry in manifest["siblings"]:
        name = entry["rfilename"]
        relative = PurePosixPath(name)
        if (not relative.parts or name != relative.as_posix() or relative.is_absolute()
                or ".." in relative.parts or "\\" in name or ":" in name):
            raise ValueError("invalid HF source path")
        if (not name.startswith(("data/", "real/", "depth/"))
                or not name.endswith((".parquet", ".tar.gz"))):
            continue
        if name in blobs:
            raise ValueError(f"duplicate HF source identity: {name}")
        size, sha = entry["size"], entry["lfs"]["sha256"]
        if type(size) is not int or size <= 0 or not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError(f"invalid LFS blob identity: {name}")
        blobs[name] = Blob(name, size, sha)
    catalogue = sorted((b.name, b.size, b.sha256) for b in blobs.values())
    canonical = json.dumps(catalogue, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != expected_sha:
        raise ValueError("HF blob catalogue differs from the pinned release")
    return hashlib.sha256(raw).hexdigest(), blobs


def cache_path(root, name):
    root = root.resolve(strict=True)
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("HF source path escapes the cache")
    return path


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(IO_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_blob(path, blob):
    return path.is_file() and path.stat().st_size == blob.size and digest_file(path) == blob.sha256


class SecureRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("WireSeg redirect must use HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_https_range(request, timeout):
    return build_opener(SecureRedirect()).open(request, timeout=timeout)


def write_atomic(path, blocks):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
            for block in blocks:
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def read_receipt(part, receipt, expected_range, count):
    if not receipt.is_file():
        return False
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    if (recorded.get("content_range") != expected_range or recorded.get("bytes") != count
            or not part.is_file() or part.stat().st_size != count
            or recorded.get("sha256") != digest_file(part)):
        raise ValueError(f"cached chunk differs from its recorded receipt: {part.name}")
    return True


def download_part(blob, start, end, part, receipt, open_range, timeout):
    range_spec = f"bytes={start}-{end}"
    expected_range = f"bytes {start}-{end}/{blob.size}"
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{quote(blob.name)}"
    request = Request(url, headers={
        "Range": range_spec, "Accept-Encoding": "identity", "Cache-Control": "no-cache",
        "User-Agent": "DH116-WireSeg-relay/1",
    })
    with open_range(request, timeout=timeout) as response:
        if (response.status != 206 or response.headers.get("Content-Range") != expected_range
                or response.headers.get("Content-Encoding", "identity") != "identity"):
            raise ValueError(f"unexpected HTTP status or Content-Range for {blob.name} {range_spec}")
        count = end - start + 1
        data = response.read(count + 1)
    if len(data) != count:
        raise ValueError(f"incomplete range for {blob.name} {range_spec}")
    write_atomic(part, [data])
    entry = {"content_range": expected_range, "bytes": count, "sha256": hashlib.sha256(data).hexdigest()}
    write_atomic(receipt, [json.dumps(entry, sort_keys=True).encode() + b"\n"])


def fetch_blob(root, blob, *, chunk_size, max_new_parts=None, open_range=open_https_range,
               timeout=30, attempts=2):
    final = cache_path(root, blob.name)
    summary = {"name": blob.name, "size": blob.size, "sha256": blob.sha256,
               "new_parts": 0, "verified": False}
    if final.exists():
        if not verify_blob(final, blob):
            raise ValueError(f"existing full LFS blob has incorrect size or SHA: {blob.name}")
        return {**summary, "verified": True}
    state = cache_path(root, f".wireseg-ranges/{blob.sha256}/{chunk_size}")
    parts = []
    for start in range(0, blob.size, chunk_size):
        end = min(start + chunk_size, blob.size) - 1
        part = state / f"{start:014d}.part"
        receipt = part.with_suffix(".json")
        if not read_receipt(part, receipt, f"bytes {start}-{end}/{blob.size}", end - start + 1):
            if max_new_parts is not None and summary["new_parts"] >= max_new_parts:
                return summary
            for attempt in range(attempts):
                try:
                    download_part(blob, start, end, part, receipt, open_range, timeout)
                    break
                except (TimeoutError, ConnectionError, IncompleteRead):
                    if attempt + 1 == attempts:
                        raise
            summary["new_parts"] += 1
        parts.append(part)

    def assembled_blocks():
        for part in parts:
            with part.open("rb") as stream:
                yield from iter(lambda: stream.read(IO_SIZE), b"")

    assembled = state / "assembled"
    write_atomic(assembled, assembled_blocks())
    if not verify_blob(assembled, blob):
        raise ValueError(f"assembled full LFS size or SHA mismatch: {blob.name}")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(assembled, final)
    return {**summary, "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "verify"))
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument("--max-new-parts", type=int)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--attempts", type=int, default=2)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--blob", action="append")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if (not 1 <= args.chunk_mib <= 64 or not 1 <= args.timeout <= 600
            or not 1 <= args.attempts <= 5
            or (args.max_new_parts is not None and args.max_new_parts < 0)):
        parser.error("invalid chunk size, timeout or part budget")
    _, blobs = read_catalogue(args.manifest or args.cache / "hf_revision_manifest.json")
    selected = args.blob or sorted(blobs)
    if len(set(selected)) != len(selected) or any(name not in blobs for name in selected):
        parser.error("all blob names must be distinct members of the pinned catalogue")
    budget = args.max_new_parts
    verified = 0
    for name in selected:
        blob = blobs[name]
        if args.command == "fetch":
            result = fetch_blob(args.cache, blob, chunk_size=args.chunk_mib * IO_SIZE,
                                max_new_parts=budget, timeout=args.timeout, attempts=args.attempts)
            if budget is not None:
                budget -= result["new_parts"]
        else:
            result = {"name": name, "size": blob.size, "sha256": blob.sha256,
                      "verified": verify_blob(cache_path(args.cache, name), blob)}
        verified += result["verified"]
        print(json.dumps({"revision": REVISION, **result}), flush=True)
        if args.command == "fetch" and budget == 0:
            break
    print(json.dumps({"repository": REPOSITORY, "revision": REVISION, "requested": len(selected),
                      "verified": verified, "catalogue_complete": verified == len(blobs)}))
    return 0 if verified == len(selected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
