"""Opt-in presentation after the real owner's durable physical event.

No physics, solver, camera, optimizer, or lifecycle ownership lives here.
Presentation must itself validate the current physical clock around rendering.
"""
class LiveFlexibleStereoScope:
    def __init__(self, owner, *, snapshot_from_exchange, present):
        if not callable(snapshot_from_exchange) or not callable(present):
            raise TypeError("snapshot and presentation callbacks are required")
        self.owner = owner
        self.snapshot_from_exchange = snapshot_from_exchange
        self.present = present
        self._active = False
        self._pending = None

    def __enter__(self):
        if self._active:
            raise RuntimeError("presentation scope already active")
        owner = self.owner
        self._control = owner.control
        self._emit = owner.ledger.emit
        self._active = True

        def emit(kind, row):
            # Preserve the authoritative journal and its original exceptions.
            self._emit(kind, row)
            if kind == "physical":
                exchange = row["exchange"]
                if exchange.get("native_teacher_source_error") is not None:
                    raise ValueError("native material source failed")
                self._pending = self.snapshot_from_exchange(exchange)

        def control(action):
            self._pending = None
            try:
                result = self._control(action)
                completed, self._pending = self._pending, None
                if completed is not None:
                    self.present(completed)
                return result
            except BaseException:
                self._pending = None
                raise

        owner.ledger.emit = emit
        owner.control = control
        return self

    def __exit__(self, exc_type, exc, tb):
        self.owner.control = self._control
        self.owner.ledger.emit = self._emit
        self._pending = None
        self._active = False
        self._control = self._emit = None
        return False
