import pytest
from go2_pvcnn.control.dh116.tabletop_live_flexible_stereo_scope import LiveFlexibleStereoScope

class Ledger:
    def __init__(self, log):
        self.log = log
    def emit(self, kind, row):
        self.log.append(("persist", kind, row))

class Owner:
    def __init__(self, log):
        self.log = log
        self.ledger = Ledger(log)
    def control(self, action):
        self.log.append(("begin", action))
        if action == "no step":
            return "no step"
        for n in (1, 2):
            self.ledger.emit("physical", {"exchange":{"n":n}})
        self.log.append(("end", action))
        return "result"

def test_present_only_after_real_control_and_durable_record():
    log=[]
    owner=Owner(log)
    def snapshot(row):
        log.append(("copy", row["n"]))
        return row["n"]
    def present(row):
        log.append(("present",row))
    old_control=owner.control
    old_emit=owner.ledger.emit
    with LiveFlexibleStereoScope(owner, snapshot_from_exchange=snapshot, present=present):
        assert owner.control("original") == "result"
    assert log == [("begin","original"),("persist","physical",{"exchange":{"n":1}}),
                   ("copy",1),("persist","physical",{"exchange":{"n":2}}),
                   ("copy",2),("end","original"),("present",2)]
    assert owner.control == old_control
    assert owner.ledger.emit == old_emit

def test_bad_source_fails_without_present_and_restores_callbacks():
    log=[]
    owner=Owner(log)
    old_control,old_emit=owner.control,owner.ledger.emit
    def broken(row):
        raise ValueError("unqualified native source")
    with pytest.raises(ValueError,match="unqualified"):
        with LiveFlexibleStereoScope(owner,snapshot_from_exchange=broken,present=lambda _: log.append("present")):
            owner.control("original")
    assert log[-1][0] == "persist"
    assert owner.control == old_control and owner.ledger.emit == old_emit
    assert "present" not in log

def test_no_physical_event_never_reuses_previous_frame():
    log=[]
    owner=Owner(log)
    with LiveFlexibleStereoScope(owner,snapshot_from_exchange=lambda row: row["n"],present=log.append):
        assert owner.control("first") == "result"
        owner.ledger.emit("unrelated", {})
        assert owner.control("no step") == "no step"
    assert log.count(2)==1
