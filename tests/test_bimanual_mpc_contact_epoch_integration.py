"""Real CPU F0/F1 providers through the Aux observer; no Native qualification."""
from copy import deepcopy
from dataclasses import replace
import importlib

import pytest

from test_bimanual_mpc_delivery_integration import Harness, api, rows
from test_bimanual_mpc_filtered_contact_values import fixture as values_fixture


class EpochHarness(Harness):
    def __init__(self,path,*,values=False):
        super().__init__(path,mode=None)
        self.with_values=values
        self.fixture=values_fixture()

    def components(self,*,clock_reader):
        scene,pairs,metadata=super().components(clock_reader=clock_reader)
        module=importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_filtered_contact_measurements')
        f=self.fixture
        clock_batch=lambda:tuple(clock_reader(i) for i in range(self.env.n))
        self.metadata_provider=module.FilteredPairMetadataProvider(sensors=f.sensors,
            wire_root_view=type('Wire',(),{'prim_paths':[f'/World/envs/env_{i}/Wire' for i in range(2)]})(),
            clock_reader=clock_batch,source_reader=lambda:deepcopy(f.source),enabled=True)
        f.provider.clock_reader=clock_batch
        return api().AuxCaptureComponents(scene,pairs,metadata,
            filtered_metadata_provider=self.metadata_provider,
            filtered_values_provider=f.provider if self.with_values else None)

    def install_epoch(self):
        return self.install(capture_filtered_contact_metadata=True,capture_filtered_contact_epoch=True,
            capture_filtered_contact_values=self.with_values)

    def physical(self,*,live=(True,True),terminal=(False,False),done=None):
        self.world.advance(None,None,None)
        self.env.task.done=list(terminal if done is None else done)
        self.env.joint_replay_observer(self.env,{},list(live),list(terminal))


def test_epoch_keeps_first_terminal_and_other_live_env_across_actual_generation_change(tmp_path):
    h=EpochHarness(tmp_path/'capture');capture=h.install_epoch();h.submit()
    h.physical(terminal=(True,False))
    first=capture.terminals['0']
    for _ in range(7):h.physical(live=(False,True),done=(True,False))
    h.env.diagnostic_episode_ids[0]=1;h.env.task.done=[False,False]
    capture.submit([[0.]*24]*2,policy_step=1);h.physical()
    summary=capture.finish();packets=rows(h.path/'aux_480hz.jsonl.gz')
    assert summary['records_per_env']==[1,9] and summary['terminal_count']==1
    assert len(packets)==9 and h.events.count('original_advance')==9
    assert capture.terminals['0']==first
    assert packets[1]['filtered_contact_epoch']['active_pairs_per_env']==[0,34]
    last=packets[-1]['filtered_contact_epoch']
    assert last['first_terminals'][0]['episode_uid'].endswith(':0')
    assert last['generation_receipts'][0]['after']['reset_generation']==1
    assert last['generation_receipts'][0]['eligibility_granted'] is False
    assert summary['filtered_contact_epoch']['snapshots']==9
    assert summary['filtered_contact_epoch']['last_receipt_sha256']==last['receipt_sha256']
    assert h.world.advance==h.original and h.env.joint_replay_observer is h.prior


def test_values_use_same_frame_and_retired_env_never_receives_value_credit(tmp_path):
    h=EpochHarness(tmp_path/'capture',values=True);capture=h.install_epoch();h.submit()
    h.physical(terminal=(True,False));h.physical(live=(False,True),done=(True,False))
    summary=capture.finish();packets=rows(h.path/'aux_480hz.jsonl.gz')
    latest=packets[-1]['filtered_pair_values']
    assert latest['capture_valid'] is True and latest['clocks']==packets[-1]['clocks']
    for item in latest['pairs']:
        active=item['metadata_pair']['clock']['env_id']==1
        assert item['normal_sensor_side_world']['valid'] is active
        assert item['friction_sensor_side_world']['valid'] is active
    counts=summary['filtered_contact_values']
    assert counts['snapshots']==2 and counts['valid_snapshots']==2
    assert counts['active_normal_valid']==counts['active_friction_valid']==102
    assert not counts['active_normal_unavailable'] and not counts['active_friction_unavailable']
    assert all(v is False for v in latest['qualification'].values())


def test_legacy_f0_without_epoch_keeps_its_strict_previous_terminal_refusal(tmp_path):
    h=EpochHarness(tmp_path/'capture')
    capture=h.install(capture_filtered_contact_metadata=True);h.submit()
    h.physical(terminal=(True,False))
    with pytest.raises(ValueError):h.physical(live=(False,True),done=(True,False))
    summary=capture.finish(error='expected strict v1 refusal')
    assert summary['terminal_count']==1 and 'filtered_contact_epoch' not in summary


def test_invalid_values_keeps_partial_raw_and_first_terminal_without_committing_epoch(tmp_path):
    h=EpochHarness(tmp_path/'capture',values=True);capture=h.install_epoch();h.submit()
    h.fixture.provider.units_reader=lambda:{'stage_identity':'wrong','meters_per_unit':1.,'kilograms_per_unit':1.}
    with pytest.raises(ValueError,match='values|F1'):h.physical(terminal=(True,False))
    summary=capture.finish(error='expected failed F1 transaction')
    terminal=capture.terminals['0']
    assert terminal['provider_partial_filtered_values']['capture_valid'] is False
    assert terminal['provider_partial_filtered_metadata']['metadata_binding_valid'] is True
    assert terminal['provider_partial_contact_epoch'] is not None
    assert capture._filtered_epoch is None and summary['terminal_count']==1
    assert summary['filtered_contact_epoch']['snapshots']==0
    assert summary['filtered_contact_values']['snapshots']==0


def test_scene_failure_does_not_commit_provisional_epoch(tmp_path):
    h=EpochHarness(tmp_path/'capture');capture=h.install_epoch();h.submit()
    failure=RuntimeError('original scene error');h.provider_error=failure
    with pytest.raises(RuntimeError) as caught:h.physical(terminal=(True,False))
    assert caught.value is failure and capture._filtered_epoch is None
    summary=capture.finish(error=repr(failure))
    assert summary['terminal_count']==1 and summary['filtered_contact_epoch']['snapshots']==0


def test_values_pair_clock_cannot_be_reassigned_under_a_correct_batch_clock(tmp_path):
    h=EpochHarness(tmp_path/'capture',values=True);capture=h.install_epoch();h.submit()
    original=h.fixture.provider.capture
    def corrupt(*args,**kwargs):
        snapshot=original(*args,**kwargs);item=snapshot.pairs[0]
        wrong=replace(item,metadata_pair=replace(item.metadata_pair,sensor_row=99))
        return replace(snapshot,pairs=(wrong,)+snapshot.pairs[1:])
    h.fixture.provider.capture=corrupt
    with pytest.raises(ValueError,match='values|F1'):h.physical(terminal=(True,False))
    capture.finish(error='expected invalid pair binding')
    assert capture._filtered_epoch is None


def test_failed_stream_write_keeps_first_terminal_and_does_not_commit_epoch(tmp_path):
    h=EpochHarness(tmp_path/'capture',values=True);capture=h.install_epoch();h.submit();original=capture._stream
    class Broken:
        def write(self,value):raise OSError('failed raw stream')
        def close(self):original.close()
    capture._stream=Broken()
    with pytest.raises(OSError):h.physical(terminal=(True,False))
    summary=capture.finish(error='expected writer failure')
    assert capture._filtered_epoch is None and summary['terminal_count']==1
    assert summary['filtered_contact_epoch']['snapshots']==0 and summary['raw_write_errors']
    assert summary['filtered_contact_epoch']['archive_errors'] and summary['filtered_contact_values']['archive_errors']


@pytest.mark.parametrize('changes',[
    {'capture_filtered_contact_epoch':True},
    {'capture_filtered_contact_values':True,'capture_filtered_contact_metadata':True},
    {'capture_filtered_contact_epoch':1,'capture_filtered_contact_metadata':True},
    {'capture_filtered_contact_values':1,'capture_filtered_contact_metadata':True,'capture_filtered_contact_epoch':True}])
def test_new_flags_require_explicit_boolean_dependencies_before_install(tmp_path,changes):
    h=EpochHarness(tmp_path/'capture')
    with pytest.raises(ValueError,match='epoch|values|bool|metadata'):h.install(**changes)
    assert h.world.advance==h.original and h.env.joint_replay_observer is h.prior


def test_batch_clock_matches_existing_per_env_readback_including_terminal_and_generation(tmp_path):
    h=EpochHarness(tmp_path/'capture');capture=h.install_epoch()
    for done,ids in (([False,False],[0,0]),([True,False],[0,0]),([False,False],[1,0])):
        h.env.task.done=done;h.env.diagnostic_episode_ids=ids
        assert capture.clock_batch()==tuple(capture.clock_for_env(i) for i in range(2))
    capture.finish()
