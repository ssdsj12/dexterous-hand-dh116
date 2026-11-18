"""Non-destructive, hash-verified copies into an independent project."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def copy_file(source,root,relative,category):
    source=Path(source);root=Path(root).resolve();relative=Path(relative)
    if source.is_symlink() or not source.is_file():raise ValueError('regular source required')
    destination=root/relative
    if relative.is_absolute() or '..' in relative.parts or not destination.resolve().is_relative_to(root):
        raise ValueError('destination escapes project')
    if destination.is_symlink():raise ValueError('destination symlink forbidden')
    digest=sha256(source)
    if destination.exists():
        if sha256(destination)!=digest:raise FileExistsError(destination)
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        fd,temp=tempfile.mkstemp(prefix='.migration-',dir=destination.parent);os.close(fd)
        try:
            shutil.copy2(source,temp)
            if sha256(temp)!=digest or sha256(source)!=digest:raise RuntimeError('source changed during copy')
            os.link(temp,destination)
        finally:os.unlink(temp)
    return dict(source=str(source.resolve()),destination=relative.as_posix(),sha256=digest,
                bytes=destination.stat().st_size,category=category,status='copied_verified')
