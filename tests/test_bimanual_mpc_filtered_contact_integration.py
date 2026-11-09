"""CPU-only fresh metadata sidecar boundaries; no sensor or force qualification."""
from dataclasses import replace
import importlib
import sys
from types import SimpleNamespace as NS

import pytest

from test_bimanual_mpc_delivery_integration import Harness, api, rows


class FilteredHarness(Harness):
    def __init__(self,path):
        super().__init__(path,mode=None)
        self.metadata_error=None
        self.wrong_clock=False
        self.reads=[]

    def components(self,*,clock_reader):
        scene,pairs,metadata=super().components(clock_reader=clock_reader)
        owner=self
        class Provider:
            enabled=True
            def capture(self,*,clocks,previous=None):
                owner.reads.append((clocks,previous))
                if owner.metadata_error:raise owner.metadata_error
                f0=importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_filtered_contact_provenance')
                if owner.wrong_clock:
                    clocks=tuple(replace(c,policy_step=c.policy_step+1) for c in clocks)
                return f0.FilteredPairSnapshot(metadata_binding_valid=False,
                    errors=('CPU fixture has no actual backend metadata',),pairs=(),clocks=clocks,
                    raw={'source_scope':'synthetic fixture, not Native'},binding_fingerprint=None)
        self.metadata_provider=Provider()
        return api().AuxCaptureComponents(scene,pairs,metadata,
            filtered_metadata_provider=self.metadata_provider)


def test_opt_in_captures_one_global_snapshot_without_an_extra_physics_step(tmp_path):
    h=FilteredHarness(tmp_path/'capture')
    capture=h.install(capture_filtered_contact_metadata=True)
    h.submit();h.step()
    summary=capture.finish()
    assert len(h.reads)==1 and h.events.count('original_advance')==1 and h.source.polls==1
    packet=rows(h.path/'aux_480hz.jsonl.gz')[0]
    snapshot=packet['filtered_pair_metadata']
    assert snapshot['clocks']==packet['clocks']
    assert snapshot['metadata_binding_valid'] is False
    assert snapshot['pairs']==[] and all(v is False for v in snapshot['qualification'].values())
    assert summary['filtered_contact_metadata']['snapshots']==1
    assert summary['filtered_contact_metadata']['valid_snapshots']==0
    assert summary['filtered_contact_metadata']['invalid_snapshots']==1
    assert h.env.joint_replay_observer is h.prior and h.world.advance==h.original


def test_default_off_preserves_legacy_components_and_does_not_write_metadata_key(tmp_path):
    h=Harness(tmp_path/'capture',mode=None)
    capture=h.install();h.submit();h.step();summary=capture.finish()
    assert 'filtered_pair_metadata' not in rows(h.path/'aux_480hz.jsonl.gz')[0]
    assert 'filtered_contact_metadata' not in summary


def test_provider_cannot_be_supplied_without_explicit_flag(tmp_path):
    h=FilteredHarness(tmp_path/'capture')
    with pytest.raises(ValueError,match='explicit|metadata'):
        h.install()
    assert h.world.advance==h.original and h.env.joint_replay_observer is h.prior


def test_enabled_capture_requires_provider_and_strict_bool(tmp_path):
    h=Harness(tmp_path/'missing',mode=None)
    with pytest.raises(ValueError,match='metadata|provider'):
        h.install(capture_filtered_contact_metadata=True)
    with pytest.raises(ValueError,match='bool'):
        Harness(tmp_path/'truthy',mode=None).install(capture_filtered_contact_metadata=1)


def test_contact_delivery_and_metadata_arms_cannot_be_combined(tmp_path):
    h=Harness(tmp_path/'capture',mode='post_step_poll')
    with pytest.raises(ValueError,match='independent|separate'):
        h.install(capture_filtered_contact_metadata=True)
    assert h.source.callback is None


@pytest.mark.parametrize('kind',['exception','wrong_clock'])
def test_failed_metadata_keeps_first_terminal_and_never_reuses_an_old_snapshot(tmp_path,kind):
    h=FilteredHarness(tmp_path/'capture')
    capture=h.install(capture_filtered_contact_metadata=True);h.submit()
    failure=RuntimeError('actual metadata getter failed')
    if kind=='exception':h.metadata_error=failure
    else:h.wrong_clock=True
    with pytest.raises((RuntimeError,ValueError)) as caught:h.step(terminal=(True,False))
    if kind=='exception':assert caught.value is failure
    summary=capture.finish(error=str(caught.value))
    assert summary['terminal_count']==1 and capture.terminals['0']['complete_measurement'] is False
    terminal=capture.terminals['0']
    assert terminal['filtered_pair_metadata'] is None
    if kind=='wrong_clock':assert terminal['provider_partial_filtered_metadata'] is not None
    assert h.world.advance==h.original and h.env.joint_replay_observer is h.prior


def test_missing_snapshot_module_preserves_acquisition_error_and_terminal(tmp_path,monkeypatch):
    h=FilteredHarness(tmp_path/'capture')
    capture=h.install(capture_filtered_contact_metadata=True);h.submit()
    monkeypatch.setitem(sys.modules,
        'go2_pvcnn.control.dh116.tabletop_bimanual_mpc_filtered_contact_provenance',None)
    with pytest.raises(ModuleNotFoundError):h.step(terminal=(True,False))
    summary=capture.finish(error='CPU fixture module unavailable')
    assert summary['terminal_count']==1
    assert capture.terminals['0']['provider_partial_filtered_metadata'] is None


@pytest.mark.parametrize('missing_module',[False,True])
def test_secondary_archive_error_cannot_replace_original_metadata_error(tmp_path,monkeypatch,missing_module):
    h=FilteredHarness(tmp_path/'capture')
    capture=h.install(capture_filtered_contact_metadata=True);h.submit()
    primary=RuntimeError('original metadata failure')
    if missing_module:
        monkeypatch.setitem(sys.modules,
            'go2_pvcnn.control.dh116.tabletop_bimanual_mpc_filtered_contact_provenance',None)
    else:h.metadata_error=primary
    stream=capture._stream
    class BrokenWrite:
        def write(self,value):raise OSError('secondary archive failure')
        def close(self):stream.close()
    capture._stream=BrokenWrite()
    with pytest.raises(ModuleNotFoundError if missing_module else RuntimeError) as caught:
        h.step(terminal=(True,False))
    if not missing_module:assert caught.value is primary
    summary=capture.finish(error=repr(caught.value))
    assert summary['terminal_count']==1 and summary['raw_write_errors']
    assert summary['filtered_contact_metadata']['archive_errors']
    assert h.world.advance==h.original and h.env.joint_replay_observer is h.prior
