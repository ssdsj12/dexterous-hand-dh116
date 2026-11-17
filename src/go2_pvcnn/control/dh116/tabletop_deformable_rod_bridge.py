"""Evidence for the owned native objects of an isolated DeformX/Isaac bridge."""
import json
import os
import time
import weakref


class NativeOwnershipTrace:
    """Record Python liveness separately from SDK release and process exit."""

    def __init__(self, path):
        self.path = path
        self.references = {}
        path.touch(exist_ok=False)

    def event(self, phase, **details):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(phase=phase, time_unix_s=time.time(), **details),
                                    allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def watch(self, name, obj, *, owner, release, gpu_view=False, callbacks=False, composer=False):
        if name in self.references:
            raise ValueError(f"duplicate native resource name: {name}")
        try:
            self.references[name] = weakref.ref(obj)
        except TypeError:
            self.references[name] = None
        self.event("acquire", resource=name, object_type=type(obj).__name__, owner=owner,
                   release=release, gpu_view=gpu_view, callbacks=callbacks, composer=composer,
                   python_weakref_available=self.references[name] is not None)

    def checkpoint(self, phase):
        self.event(phase, python_liveness={
            name: ref() is not None if ref is not None else None
            for name, ref in self.references.items()
        })
