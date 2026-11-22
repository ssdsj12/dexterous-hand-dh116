"""CPU evidence durability; no native lease, clock or PPO qualification."""
import json
from types import SimpleNamespace as NS
import pytest
from go2_pvcnn.control.dh116.tabletop_native_teacher_actual_provider import _record_actual_wall_budget
from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal


@pytest.mark.parametrize("changed,value,reason",[
    ("remaining_s",-0.25,"wall_deadline_exhausted"),
    ("parent_deadline_s",181.,"parent_deadline_mismatch"),
    ("local_cap_s",179.,"external_cap_mismatch"),
    ("episode_cap_s",1.,"episode_cap_mismatch"),
])
def test_rejected_wall_readback_is_durable_before_exception(tmp_path,changed,value,reason):
    sample=dict(started_s=1.,observed_s=180.25,parent_deadline_s=180.,
                remaining_s=0.25,local_cap_s=180.)
    context=NS(external_cap_s=180.,episode_cap_s=.5333333333333333)
    if changed=="episode_cap_s":context.episode_cap_s=value
    else:sample[changed]=value
    original=dict(sample)
    with NativeJournalForTest(tmp_path/"events.jsonl") as journal:
        collector=NS(owner=NS(contract=NS(runtime=NS(wall_readback=lambda:dict(sample)),
            config=NS(episode_cap_s=.5333333333333333)),context=context,
            episode_uid="cpu-only",sim=NS(current_time_step_index=66)),journal=journal)
        with pytest.raises(ValueError,match="live actual Teacher wall budget"):
            _record_actual_wall_budget(collector,NS(deadline_monotonic_s=180.),"after_update")
        rows=[json.loads(line) for line in journal.path.read_text().splitlines()]
        assert len(rows)==1
        row=rows[0]
        assert row["operation"]=="fresh_update_wall_budget_rejected"
        assert row["payload"]["rejection_reasons"]==[reason]
        assert row["payload"]["phase"]=="after_update" and row["payload"]["sdk_step"]==66
        assert row["payload"]["native_qualification"] is row["payload"]["teacher_admission"] is False
        assert all(row["payload"][key]==value for key,value in original.items())
        assert sample==original


class NativeJournalForTest(NativeJournal):
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def test_failed_evidence_write_preserves_budget_rejection():
    sample=dict(started_s=1.,observed_s=181.,parent_deadline_s=180.,
                remaining_s=-1.,local_cap_s=180.)
    calls=[]
    def failed_write(operation,payload):
        calls.append((operation,payload))
        raise OSError("disk unavailable")
    owner=NS(contract=NS(runtime=NS(wall_readback=lambda:sample),
        config=NS(episode_cap_s=.5)),context=NS(external_cap_s=180.,episode_cap_s=.5),
        episode_uid="cpu-only",sim=NS(current_time_step_index=66))
    with pytest.raises(ValueError,match="wall budget.*evidence write failed") as caught:
        _record_actual_wall_budget(NS(owner=owner,journal=NS(emit=failed_write)),
            NS(deadline_monotonic_s=180.),"after_update")
    assert len(calls)==1
    assert isinstance(caught.value.__cause__,OSError)
    assert calls[0][1]["rejection_reasons"]==["wall_deadline_exhausted"]
