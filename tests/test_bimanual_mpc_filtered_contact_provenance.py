"""Synthetic F0 metadata checks only; no contacts, SDK or Native qualification."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import importlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import SampleClock


def api():
    source=ROOT/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_filtered_contact_provenance.py'
    assert source.is_file(), 'pure F0 filtered-pair metadata resolver is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_filtered_contact_provenance')


def clock(env=0, **changes):
    values=dict(run_uuid='synthetic-only',env_id=env,episode_uid=f'ep:{env}:0',
        world_step=10,world_time_s=10/480,sdk_step=1010,sdk_time_s=5+10/480,
        policy_step=1,reset_generation=0)
    values.update(changes)
    return SampleClock(**values)


def fixture(envs=(1,0)):
    rows=[]
    for side in ('left','right'):
        for body in range(17):
            paths=[f'/World/envs/env_{i}/{side}Hand/finger_{body}_link' for i in envs]
            rows.append(dict(sensor_index=len(rows),source='synthetic.ContactSensor',
                sensor_paths=paths,body_prim_paths=list(paths),filter_count=1,
                contact_view_identity=dict(python_runtime_id=1000+len(rows),python_type='synthetic.RigidContactView'),
                configured_filter_expressions=['THIS CONFIG IS NEVER READ'],
                filter_paths=dict(available=True,
                    value=[[f'/World/envs/env_{i}/Wire'] for i in envs],
                    source='actual contact_physx_view.filter_paths getter',resolved_semantics='unknown')))
    identity=dict(stage_identity='synthetic-stage:13',contact_view_api=dict(
        path='/synthetic/omni/physics/tensors/impl/api.py',sha256='a'*64,
        python_type='synthetic.RigidContactView'))
    return dict(sensor_sources=dict(available=True,sensors=rows,per_contact_partner_id_recorded=False),
        wire_body_paths=[f'/World/envs/env_{i}/Wire' for i in reversed(envs)],
        clocks_before=[clock(i) for i in reversed(envs)],clocks_after=[clock(i) for i in reversed(envs)],
        source_identity=dict(before=identity,after=deepcopy(identity)))


class FilteredPairMetadataTests(unittest.TestCase):
    def resolve(self,data=None,**changes):
        data=fixture() if data is None else data
        data.update(changes)
        return api().resolve_filtered_sensor_pairs(**data)

    def test_resolves_actual_rows_by_env_not_array_position_or_config(self):
        result=self.resolve()
        self.assertTrue(result.metadata_binding_valid)
        self.assertEqual(result.errors,())
        self.assertEqual(len(result.pairs),68)
        first=result.pairs[0]
        self.assertEqual((first.sensor_index,first.sensor_row,first.filter_column),(0,0,0))
        self.assertEqual((first.clock.env_id,first.hand,first.body_name),(1,'left','finger_0_link'))
        self.assertEqual(first.sensor_path,'/World/envs/env_1/leftHand/finger_0_link')
        self.assertEqual(first.filter_path,'/World/envs/env_1/Wire')
        self.assertEqual(first.pair_basis,'SDK_resolved_sensor_filter_paths_only')

    def test_never_issues_numeric_partner_force_support_or_native_qualification(self):
        result=self.resolve()
        self.assertTrue(all(p.partner_body_id is None for p in result.pairs))
        self.assertTrue(all(p.actor_id is None and p.collider_id is None for p in result.pairs))
        self.assertFalse(any(result.qualification.values()))
        self.assertFalse(result.contact_values_audited)
        self.assertIn('count/start',result.contact_values_missing_reason)

    def test_results_and_nested_raw_are_owned_and_immutable(self):
        data=fixture();result=self.resolve(data)
        data['sensor_sources']['sensors'][0]['sensor_paths'][0]='/changed'
        data['source_identity']['before']['contact_view_api']['sha256']='b'*64
        self.assertNotEqual(result.pairs[0].sensor_path,'/changed')
        self.assertEqual(result.raw['source_identity']['before']['contact_view_api']['sha256'],'a'*64)
        with self.assertRaises((FrozenInstanceError,AttributeError)):
            result.pairs[0].sensor_path='/changed'
        with self.assertRaises(TypeError):result.raw['new']='forged'
        with self.assertRaises(TypeError):result.raw['sensor_sources']['sensors'][0]['sensor_paths'][0]='/changed'
        plain=result.to_dict();plain['raw']['source_identity']['before']['contact_view_api']['sha256']='c'*64
        self.assertEqual(result.raw['source_identity']['before']['contact_view_api']['sha256'],'a'*64)

    def test_unavailable_actual_filter_cannot_use_correct_config_as_fallback(self):
        for defect in ('missing','unavailable','wrong_source','config_only'):
            data=fixture();row=data['sensor_sources']['sensors'][0]
            row['configured_filter_expressions']=[f'/World/envs/env_{i}/Wire' for i in (1,0)]
            if defect=='missing':del row['filter_paths']
            elif defect=='unavailable':row['filter_paths']['available']=False
            elif defect=='wrong_source':row['filter_paths']['source']='configuration'
            else:row['filter_paths']['value']=None
            with self.subTest(defect=defect):
                result=self.resolve(data)
                self.assertFalse(result.metadata_binding_valid)
                self.assertEqual(result.pairs,())
                self.assertTrue(result.errors)

    def test_actual_filter_must_be_nested_singleton_exact_same_env_wire(self):
        for value in ([['/World/envs/env_.*/Wire']]*2,
                      [['/World/envs/env_0/Wire'],['/World/envs/env_1/Wire']],
                      ['/World/envs/env_1/Wire','/World/envs/env_0/Wire'],
                      [['/World/envs/env_1/Wire','/extra'],['/World/envs/env_0/Wire']],
                      [['/World/envs/env_1/Table'],['/World/envs/env_0/Wire']]):
            data=fixture();data['sensor_sources']['sensors'][0]['filter_paths']['value']=value
            with self.subTest(value=value):self.assertFalse(self.resolve(data).metadata_binding_valid)

    def test_sensor_body_paths_and_optional_counts_must_agree(self):
        for key,value in (('body_prim_paths',['/wrong']*2),('sensor_paths',['/wrong']*2),
                          ('filter_count',True),('filter_count',2),('sensor_count',3)):
            data=fixture();data['sensor_sources']['sensors'][0][key]=value
            with self.subTest(key=key,value=value):self.assertFalse(self.resolve(data).metadata_binding_valid)
        data=fixture();data['sensor_sources']['sensors'][0]['sensor_count']=2
        self.assertTrue(self.resolve(data).metadata_binding_valid)

    def test_exact_34_unique_sensors_and_17_unique_bodies_per_hand_required(self):
        for defect in ('missing','duplicate_index','duplicate_body','body_drift','side_drift'):
            data=fixture();rows=data['sensor_sources']['sensors']
            if defect=='missing':rows.pop()
            elif defect=='duplicate_index':rows[-1]['sensor_index']=0
            elif defect=='duplicate_body':rows[1]=deepcopy(rows[0]);rows[1]['sensor_index']=1
            else:
                path=rows[0]['sensor_paths'][1]
                path=path.replace('finger_0_link','finger_other_link') if defect=='body_drift' else path.replace('leftHand','rightHand')
                rows[0]['sensor_paths'][1]=path;rows[0]['body_prim_paths'][1]=path
            with self.subTest(defect=defect):self.assertFalse(self.resolve(data).metadata_binding_valid)

    def test_same_body_basename_under_different_paths_is_not_a_unique_hand_mapping(self):
        data=fixture();record=data['sensor_sources']['sensors'][1]
        for key in ('sensor_paths','body_prim_paths'):
            record[key]=[p.replace('finger_1_link','subtree/finger_0_link') for p in record[key]]
        self.assertFalse(self.resolve(data).metadata_binding_valid)

    def test_wire_paths_must_come_from_actual_unique_root_view_and_match_all_envs(self):
        for paths in ([], ['/World/envs/env_0/Wire']*2,
                      ['/World/envs/env_0/Wire','/World/envs/env_2/Wire'],
                      ['/World/envs/env_0/Wire/geometry','/World/envs/env_1/Wire']):
            with self.subTest(paths=paths):self.assertFalse(self.resolve(wire_body_paths=paths).metadata_binding_valid)

    def test_getter_bracket_cannot_cross_frame_episode_reset_or_source_clock(self):
        for change in (dict(world_step=11),dict(episode_uid='other'),dict(reset_generation=1),
                       dict(sdk_time_s=9.),dict(policy_step=2),dict(terminal=True),dict(reset=True)):
            data=fixture();data['clocks_after'][0]=replace(data['clocks_after'][0],**change)
            with self.subTest(change=change):self.assertFalse(self.resolve(data).metadata_binding_valid)

    def test_clock_batch_must_have_unique_envs_and_one_actual_physics_frame(self):
        for defect in ('duplicate','missing','frame','policy','run','not_clock'):
            data=fixture();clocks=list(data['clocks_before'])
            if defect=='duplicate':clocks[1]=clocks[0]
            elif defect=='missing':clocks.pop()
            elif defect=='frame':clocks[1]=replace(clocks[1],world_step=11)
            elif defect=='policy':clocks[1]=replace(clocks[1],policy_step=2)
            elif defect=='run':clocks[1]=replace(clocks[1],run_uuid='another')
            else:clocks[0]={'not':'SampleClock'}
            data['clocks_before']=clocks;data['clocks_after']=list(clocks)
            with self.subTest(defect=defect):self.assertFalse(self.resolve(data).metadata_binding_valid)

    def test_source_api_identity_is_required_and_never_inferred_from_type_label(self):
        for defect in ('stage','path','sha','type','available'):
            data=fixture();identity=data['source_identity']['before']
            if defect=='stage':identity['stage_identity']=None
            elif defect=='available':identity['contact_view_api']=None
            elif defect=='sha':identity['contact_view_api']['sha256']='not-a-sha'
            else:identity['contact_view_api']['path' if defect=='path' else 'python_type']=''
            with self.subTest(defect=defect):self.assertFalse(self.resolve(data).metadata_binding_valid)

    def test_terminal_can_be_archived_but_never_continued_and_reset_requires_rebuild(self):
        data=fixture()
        clocks=[replace(c,terminal=True) for c in data['clocks_before']]
        terminal=self.resolve(data,clocks_before=clocks,clocks_after=clocks)
        self.assertTrue(terminal.metadata_binding_valid)
        self.assertFalse(any(terminal.qualification.values()))
        later=[replace(c,world_step=11,sdk_step=1011,world_time_s=11/480,sdk_time_s=5+11/480) for c in clocks]
        self.assertFalse(self.resolve(clocks_before=later,clocks_after=later,previous=terminal).metadata_binding_valid)
        reset=[replace(c,reset=True) for c in fixture()['clocks_before']]
        self.assertFalse(self.resolve(clocks_before=reset,clocks_after=reset).metadata_binding_valid)

    def test_previous_snapshot_refuses_stale_source_or_cross_episode_retimestamp(self):
        previous=self.resolve()
        next_clocks=[replace(c,world_step=11,sdk_step=1011,world_time_s=11/480,sdk_time_s=5+11/480)
                     for c in fixture()['clocks_before']]
        data=fixture();data.update(clocks_before=next_clocks,clocks_after=next_clocks,previous=previous)
        self.assertTrue(self.resolve(data).metadata_binding_valid)
        for defect in ('repeat','source','stage','episode','reset','sdk_delta','policy_regress','paths','view'):
            changed=fixture();changed.update(clocks_before=list(next_clocks),clocks_after=list(next_clocks),previous=previous)
            if defect=='repeat':changed['clocks_before']=fixture()['clocks_before'];changed['clocks_after']=fixture()['clocks_after']
            elif defect in ('source','stage'):
                if defect=='source':changed['source_identity']['before']['contact_view_api']['sha256']='b'*64
                else:changed['source_identity']['before']['stage_identity']='another-stage'
                changed['source_identity']['after']=deepcopy(changed['source_identity']['before'])
            elif defect=='view':changed['sensor_sources']['sensors'][0]['contact_view_identity']['python_runtime_id']=99999
            elif defect=='paths':
                for field in ('sensor_paths','body_prim_paths'):
                    changed['sensor_sources']['sensors'][0][field]=[p.replace('finger_0_link','renamed_link') for p in changed['sensor_sources']['sensors'][0][field]]
            else:
                values=dict(episode_uid='other') if defect=='episode' else dict(reset_generation=1) if defect=='reset' else dict(sdk_step=1012) if defect=='sdk_delta' else dict(policy_step=0)
                changed['clocks_before'][0]=replace(next_clocks[0],**values);changed['clocks_after'][0]=changed['clocks_before'][0]
            with self.subTest(defect=defect):self.assertFalse(self.resolve(changed).metadata_binding_valid)

    def test_invalid_metadata_preserves_raw_reason_and_never_partial_pair_credit(self):
        data=fixture();data['sensor_sources']['sensors'][-1]['filter_count']=None
        result=self.resolve(data)
        self.assertFalse(result.metadata_binding_valid);self.assertEqual(result.pairs,())
        self.assertIsNone(result.raw['sensor_sources']['sensors'][-1]['filter_count'])
        self.assertTrue(result.errors);self.assertFalse(any(result.qualification.values()))

    def test_current_stage_api_bracket_and_actual_view_lifecycle_are_required(self):
        for defect in ('source_after','stage_change','api_change','missing_view','duplicate_view','type_change'):
            data=fixture();row=data['sensor_sources']['sensors'][0]
            if defect=='source_after':del data['source_identity']['after']
            elif defect=='stage_change':data['source_identity']['after']['stage_identity']='new'
            elif defect=='api_change':data['source_identity']['after']['contact_view_api']['sha256']='b'*64
            elif defect=='missing_view':del row['contact_view_identity']
            elif defect=='duplicate_view':row['contact_view_identity']=deepcopy(data['sensor_sources']['sensors'][1]['contact_view_identity'])
            else:row['contact_view_identity']['python_type']='different.Type'
            with self.subTest(defect=defect):self.assertFalse(self.resolve(data).metadata_binding_valid)


if __name__=='__main__':unittest.main()
