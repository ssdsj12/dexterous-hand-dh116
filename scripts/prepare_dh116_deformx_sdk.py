"""Verify pinned NVIDIA CUDA compiler packages and assemble a private SDK."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import posixpath
import shutil
import subprocess
import tarfile
import zipfile


def relative_symlink(name, target):
    member = Path(name)
    link = Path(target)
    resolved = Path(posixpath.normpath(str(member.parent / link)))
    if (member.is_absolute() or ".." in member.parts or link.is_absolute()
            or "\\" in name or "\\" in target or ".." in resolved.parts):
        raise ValueError(f"unsafe archive symlink: {name} -> {target}")
    return member


def extract_package(archive, destination, kind):
    destination.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        entry, = [name for name in bundle.namelist()
                  if name.startswith(f"{kind}-") and name.endswith(".tar.zst")]
        compressed = destination / "payload.tar.zst"
        with bundle.open(entry) as source, compressed.open("xb") as stream:
            shutil.copyfileobj(source, stream)

    links = []

    def package_filter(member, path):
        if member.issym():
            links.append((relative_symlink(member.name, member.linkname), member.linkname))
            return None
        if not member.isdir() and not member.isfile():
            raise ValueError(f"unsupported archive member: {member.name}")
        return tarfile.data_filter(member, path)

    with subprocess.Popen(["zstd", "-dc", str(compressed)], stdout=subprocess.PIPE) as decoder:
        with tarfile.open(fileobj=decoder.stdout, mode="r|") as tar:
            tar.extractall(destination, filter=package_filter)
        if decoder.wait():
            raise ValueError(f"decompression failed: {archive.name}")
    compressed.unlink()
    for name, target in links:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)


def merge_component(source, target):
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if rel.parts[0] == "info":
            continue
        out = target / rel
        if path.is_symlink():
            link = os.readlink(path)
            relative_symlink(str(rel), link)
            if out.is_symlink() and os.readlink(out) == link:
                continue
            if out.exists() or out.is_symlink():
                raise ValueError(f"conflicting SDK path: {rel}")
            out.symlink_to(link)
        elif out.is_symlink():
            raise ValueError(f"conflicting SDK path: {rel}")
        elif path.is_dir():
            out.mkdir(exist_ok=True)
        elif out.exists():
            if not out.is_file() or out.read_bytes() != path.read_bytes():
                raise ValueError(f"conflicting SDK path: {rel}")
        else:
            shutil.copy2(path, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.lock.read_text())["nvidia_packages"]
    root = args.output.resolve()
    root.mkdir(mode=0o700)
    sdk = root / "sdk"
    components = root / "components"
    metadata = root / "metadata"
    for path in (sdk, components, metadata):
        path.mkdir()
    for name, row in inputs.items():
        filename = row["file"]
        if Path(name).name != name or Path(filename).name != filename or not filename.endswith(".conda"):
            raise ValueError("invalid package identity in SDK lock")
        archive = args.archives / filename
        if (archive.stat().st_size != row["size"]
                or hashlib.sha256(archive.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError(f"size or SHA256 differs: {filename}")
        component = components / name
        extract_package(archive, component, "pkg")
        info = metadata / name
        extract_package(archive, info, "info")
        records = json.loads((info / "info/paths.json").read_text())["paths"]
        if any(row.get("prefix_placeholder") for row in records):
            raise ValueError(f"unexpected prefix-relocation dependency: {name}")
        merge_component(component, sdk)
    (sdk / "include").symlink_to("targets/x86_64-linux/include")
    (sdk / "lib64").symlink_to("targets/x86_64-linux/lib")
    (sdk / "targets/x86_64-linux/nvvm").symlink_to("../../nvvm")
    for path in sdk.rglob("*"):
        if path.is_symlink() and not path.resolve(strict=True).is_relative_to(sdk):
            raise ValueError(f"SDK link resolves outside assembled root: {path}")
    print(sdk)


if __name__ == "__main__":
    main()
