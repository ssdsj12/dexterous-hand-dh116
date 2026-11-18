import hashlib
import pytest
from tools.migration_copy import copy_file


def test_copy_and_digest(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'original')
    root=tmp_path/'space root'
    entry=copy_file(source,root,'src/a.py','code')
    assert (root/'src/a.py').read_bytes()==b'original'
    assert entry['sha256']==hashlib.sha256(b'original').hexdigest()
    assert entry['category']=='code'


def test_refuse_conflict(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'a')
    root=tmp_path/'dest';root.mkdir();(root/'x').write_bytes(b'b')
    with pytest.raises(FileExistsError):copy_file(source,root,'x','code')
    assert (root/'x').read_bytes()==b'b'


def test_refuse_escape_and_symlink(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'a')
    with pytest.raises(ValueError):copy_file(source,tmp_path/'dest','../escape','code')
    link=tmp_path/'link';link.symlink_to(source)
    with pytest.raises(ValueError):copy_file(link,tmp_path/'dest','x','code')


def test_idempotent_equal_file(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'a')
    a=copy_file(source,tmp_path/'dest','x','code')
    assert copy_file(source,tmp_path/'dest','x','code')==a
