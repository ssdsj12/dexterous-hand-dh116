"""Durable text-only lifecycle events. No native or exception objects are stored."""
import json
import os
import time
from pathlib import Path

class StageJournal:
    def __init__(self,path):
        self._stream=Path(path).open("x",encoding="utf-8")
        self._sequence=0
    def emit(self,row):
        event=dict(sequence=self._sequence,monotonic_ns=time.monotonic_ns(),**row)
        self._stream.write(json.dumps(event,sort_keys=True,allow_nan=False)+"\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._sequence+=1
    def close(self):
        stream,self._stream=self._stream,None
        if stream is not None:
            try:
                stream.flush();os.fsync(stream.fileno())
            finally:stream.close()
    def __enter__(self):return self
    def __exit__(self,kind,error,tb):
        try:self.close()
        except BaseException as cleanup:
            if error is None:raise
            error.diagnostic_close_error=repr(cleanup)
        return False
