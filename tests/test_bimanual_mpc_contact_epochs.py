"""Synthetic first-cohort eligibility only; never Native/contact qualification."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import importlib
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from test_bimanual_mpc_filtered_contact_provenance import fixture, clock
from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_filtered_contact_provenance import resolve_filtered_sensor_pairs

MODULE='go2_pvcnn.control.dh116.tabletop_bimanual_mpc_contact_epochs'


def api():
    assert (ROOT/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_contact_epochs.py').is_file(), 'explicit contact epoch contract is missing'
    return importlib.import_module(MODULE)


def frame(step=10,policy=1,terminal=(),generations=(0,0),change_source=False):
    data=fixture()
    clocks=[clock(i,world_step=step,world_time_s=step/480,sdk_step=1000+step,
        sdk_time_s=5+step/480,policy_step=policy,terminal=i in terminal,
        reset_generation=generations[i],episode_uid=f'ep:{i}:{generations[i]}') for i in (0,1)]
    data.update(clocks_before=clocks,clocks_after=clocks)
    if change_source:
        for key in ('before','after'):data['source_identity'][key]['stage_identity']='different-stage'
    result=resolve_filtered_sensor_pairs(**data)
    assert result.metadata_binding_valid,result.errors
    return result


def advance(snapshot=None,previous=None,live=(True,True),terminal=(False,False),cohort=None,receipts=()):
    return api().advance_contact_epoch(frame() if snapshot is None else snapshot,
        previous=previous,actual_live=live,actual_terminal=terminal,
        cohort_live=live if cohort is None else cohort,
        retired_generation_receipts=receipts,
        liveness_source='synthetic actual pre-reset observer masks')


class ContactEpochTests(unittest.TestCase):
    def ended(self):
        return advance(frame(terminal=(0,)),terminal=(True,False))

    def test_normal_frames_have_one_history_owner_without_relabeling_f0(self):
        first=advance();second=advance(frame(11),first)
        self.assertEqual(second.version,'dh116-filtered-contact-first-cohort-v2')
        self.assertEqual(second.active_pairs_per_env,(34,34))
        self.assertEqual(second.previous_receipt_sha256,first.receipt_sha256)
        self.assertEqual(second.metadata.version,'dh116-filtered-pair-metadata-f0-v1')
        self.assertFalse(any(second.qualification.values()))

    def test_terminal_then_inactive_same_episode_keeps_other_env_live(self):
        first=self.ended()
        second=advance(frame(11,terminal=(0,)),first,live=(False,True),cohort=(False,True))
        self.assertEqual(second.active_pairs_per_env,(0,34))
        self.assertEqual(second.retired,(True,False))
        self.assertEqual(second.first_terminals[0],first.first_terminals[0])
        self.assertTrue(all(p.clock.env_id==1 for p in second.active_pairs))
        self.assertEqual(len(second.metadata.pairs),68)

    def test_retired_generation_change_requires_explicit_readback_receipt(self):
        first=self.ended();new=frame(11,policy=2,generations=(1,0))
        with self.assertRaisesRegex(ValueError,'receipt'):
            advance(new,first,cohort=(False,True))
        receipt=api().RetiredGenerationReceipt(first.metadata.clocks[0],new.clocks[0],
            source='synthetic actual diagnostic generation readback')
        second=advance(new,first,cohort=(False,True),receipts=(receipt,))
        self.assertEqual(second.active_pairs_per_env,(0,34))
        self.assertEqual(second.first_terminals[0],first.first_terminals[0])
        self.assertEqual(second.generation_receipts,(receipt,))
        self.assertFalse(receipt.eligibility_granted)

    def test_closed_slot_cannot_resurrect_or_clear_terminal_in_old_episode(self):
        first=self.ended()
        with self.assertRaisesRegex(ValueError,'cohort'):
            advance(frame(11,terminal=(0,)),first,cohort=(True,True))
        with self.assertRaisesRegex(ValueError,'terminal'):
            advance(frame(11),first,cohort=(False,True))

    def test_open_slot_cannot_change_episode_or_be_silently_removed(self):
        first=advance()
        with self.assertRaisesRegex(ValueError,'episode'):
            advance(frame(11,policy=2,generations=(0,1)),first)
        with self.assertRaisesRegex(ValueError,'open'):
            advance(frame(11),first,live=(False,True))

    def test_source_or_view_binding_and_clock_discontinuity_are_refused(self):
        first=advance()
        with self.assertRaisesRegex(ValueError,'binding'):
            advance(frame(11,change_source=True),first)
        for step in (10,12):
            with self.subTest(step=step),self.assertRaisesRegex(ValueError,'clock'):
                advance(frame(step),first)

    def test_all_closed_is_no_sample_and_never_overwrites_terminal(self):
        first=advance(frame(terminal=(0,1)),terminal=(True,True))
        before=first.to_dict()
        with self.assertRaisesRegex(ValueError,'no live'):
            advance(frame(11,terminal=(0,1)),first,live=(False,False),cohort=(False,False))
        self.assertEqual(before,first.to_dict())

    def test_data_are_owned_and_immutable(self):
        state=self.ended();data=state.to_dict()
        data['first_terminals'][0]['episode_uid']='forged'
        self.assertEqual(state.first_terminals[0].episode_uid,'ep:0:0')
        with self.assertRaises((FrozenInstanceError,AttributeError)):state.retired=(False,False)
        with self.assertRaises(TypeError):state.qualification['planning_eligible']=True

    def test_modified_previous_cannot_discard_retirement_history(self):
        first=self.ended()
        forged=replace(first,retired=(False,False),first_terminals=(None,None))
        with self.assertRaisesRegex(ValueError,'history'):
            advance(frame(11),forged)

    def test_receipt_for_open_slot_or_wrong_current_clock_is_refused(self):
        first=self.ended();new=frame(11,policy=2,generations=(1,0))
        unrelated=api().RetiredGenerationReceipt(first.metadata.clocks[1],
            replace(new.clocks[1],reset_generation=1,episode_uid='ep:1:1'),source='synthetic unrelated')
        with self.assertRaisesRegex(ValueError,'receipt'):
            advance(new,first,cohort=(False,True),receipts=(unrelated,))

    def test_retired_post_reset_metadata_can_advance_without_credit(self):
        first=self.ended();new=frame(11,policy=2,generations=(1,0))
        receipt=api().RetiredGenerationReceipt(first.metadata.clocks[0],new.clocks[0],source='synthetic actual generation')
        second=advance(new,first,cohort=(False,True),receipts=(receipt,))
        third=advance(frame(12,policy=2,generations=(1,0)),second,cohort=(False,True))
        self.assertEqual(third.active_pairs_per_env,(0,34))
        self.assertEqual(third.first_terminals,first.first_terminals)

    def test_forged_f0_typed_pairs_cannot_replace_actual_raw_binding(self):
        snapshot=frame()
        forged=replace(snapshot,pairs=(replace(snapshot.pairs[0],sensor_row=99),)+snapshot.pairs[1:])
        with self.assertRaisesRegex(ValueError,'DTO/raw'):
            advance(forged)

    def test_previous_metadata_raw_cannot_be_replaced_under_old_fingerprint(self):
        first=advance();raw=first.metadata.to_dict()['raw']
        raw['sensor_sources']['sensors'][0]['sensor_paths'][0]='/forged'
        forged=replace(first,metadata=replace(first.metadata,raw=raw))
        with self.assertRaisesRegex(ValueError,'DTO/raw'):
            advance(frame(11),forged)

    def test_history_binds_non_pair_configuration_and_acquisition_raw_too(self):
        first=advance()
        for defect in ('config','acquisition'):
            raw=first.metadata.to_dict()['raw']
            if defect=='config':raw['sensor_sources']['sensors'][0]['configured_filter_expressions']=['FORGED_AFTER_HISTORY']
            else:raw['acquisition']={'forged':'AFTER_HISTORY'}
            forged=replace(first,metadata=replace(first.metadata,raw=raw))
            with self.subTest(defect=defect),self.assertRaisesRegex(ValueError,'history'):
                advance(frame(11),forged)

    def test_actual_masks_are_strict_and_first_terminal_clock_must_match(self):
        for changes in ({'live':(1,True)},{'live':(True,)},
                        {'terminal':(True,False)},{'terminal':(True,False),'live':(False,True)}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):advance(**changes)

    def test_generation_receipt_cannot_jump_generation_or_change_source_time(self):
        first=self.ended()
        for changes in ({'reset_generation':2,'episode_uid':'ep:0:2','policy_step':2},
                        {'reset_generation':1,'episode_uid':'ep:0:1','policy_step':1},
                        {'reset_generation':1,'episode_uid':'ep:0:1','policy_step':2,'sdk_step':1012}):
            current=frame(11).clocks[0]
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                api().RetiredGenerationReceipt(first.metadata.clocks[0],replace(current,**changes),source='synthetic')


if __name__=='__main__':unittest.main()
