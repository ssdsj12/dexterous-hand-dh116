"""CPU examples check routing and shared native-buffer ownership only."""
import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_teacher_wire_tensor_probe as probe


class ContactView:
    sensor_count = 2
    filter_count = 2
    max_contact_data_count = 8
    sensor_paths = ['/env0/Wire', '/env1/Wire']
    filter_paths = [['/env0/a', '/env0/b'], ['/env1/a', '/env1/b']]

    def __init__(self):
        self.counts = torch.tensor([[1, 1], [1, 0]], dtype=torch.int32)
        self.starts = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)
        self.matrices = torch.zeros(2,2,3)
        self.matrices[:,:,2] = torch.tensor([[1., 2.], [3., 0.]])

    def check(self):
        return True

    def get_contact_force_matrix(self, dt):
        assert dt == 1/480
        return self.matrices

    def get_contact_data(self, dt):
        assert dt == 1/480
        loads = torch.arange(1, 9).reshape(8,1).float()
        normals = torch.tensor([[0.,0.,1.]]).repeat(8,1)
        return loads, torch.zeros(8,3), normals, torch.zeros(8,1), self.counts, self.starts

    def get_friction_data(self, dt):
        assert dt == 1/480
        # Installed API reuses these exact metadata buffers.
        self.counts.fill_(0)
        self.starts.fill_(0)
        self.counts[0, 1] = 1
        force = torch.zeros(8,3)
        force[0, 0] = .25
        return force, torch.zeros(8,3), self.counts, self.starts


def test_exact_order_and_copies_before_getter_alias_reuse():
    view = ContactView()
    result = probe.read_filtered(view, view.sensor_paths, view.filter_paths, dt=1/480)
    assert result['normal_pair_n'][0] == [[0.,0.,1.], [0.,0.,2.]]
    assert result['friction_pair_n'][0] == [[0.,0.,0.], [.25,0.,0.]]
    assert result['normal_cells'][0][0]['count'] == 1
    assert result['friction_cells'][0][0]['count'] == 0


def test_filter_binding_cannot_be_accepted_by_slot_count_alone():
    view = ContactView()
    wrong = list(reversed(view.filter_paths))
    with pytest.raises(ValueError, match='filter routing'):
        probe.read_filtered(view, view.sensor_paths, wrong, dt=1/480)


@pytest.mark.parametrize('fault', ['overlap', 'saturated', 'matrix'])
def test_bad_slots_or_inconsistent_native_matrix_cannot_be_reduced(fault):
    view = ContactView()
    if fault == 'overlap':
        view.starts[1,0] = 1
    elif fault == 'saturated':
        view.counts[0,0] = 6
    else:
        view.matrices[0,0,2] = 2.
    with pytest.raises(ValueError, match='slot|matrix|capacity'):
        probe.read_filtered(view, view.sensor_paths, view.filter_paths, dt=1/480)


def test_total_net_residual_keeps_both_candidates_and_unverified_semantics():
    view = ContactView()
    a = probe.read_filtered(view, view.sensor_paths, view.filter_paths, dt=1/480)
    b = probe.net_residual([[.25,0.,4.],[0.,0.,3.]], {'hand':a})
    assert b['net_minus_filtered_normal_n'] == [[.25,0.,1.],[0.,0.,0.]]
    assert b['net_minus_filtered_normal_and_friction_n'] == [[0.,0.,1.],[0.,0.,0.]]
    assert b['unknown_partner_coverage_qualified'] is False
    assert b['upstream_overflow_excluded'] is False
    assert b['net_contains_friction'] is None


def test_instance_proxy_collision_routes_via_exact_nearest_rigid_actor():
    from types import SimpleNamespace
    collision, rigid, force, joint, proxy = object(), object(), object(), object(), object()
    class Prim:
        def __init__(self, path, apis=(), parent=None, instance=False, enabled=True):
            self.path,self.apis,self.parent,self.instance,self.enabled=path,apis,parent,instance,enabled
        def HasAPI(self, api): return api in self.apis
        def GetParent(self): return self.parent
        def GetPath(self): return self.path
        def IsInstanceProxy(self): return self.instance
        def GetAppliedSchemas(self): return []
        def GetPrimStack(self): return []
        def GetTypeName(self): return 'Mesh'
        def IsA(self, t): return False
    actor=Prim('/env0/hand',(rigid,))
    proxy_shape=Prim('/env0/hand/shape',(collision,),actor,instance=True)
    unknown=Prim('/global/shape',(collision,),None,instance=True)
    disabled=Prim('/global/disabled',(collision,),None,instance=True,enabled=False)
    def traverse(predicate):
        assert predicate is proxy
        return [actor,proxy_shape,unknown,disabled]
    collision_api=type('API',(),{'__call__':lambda self,prim:SimpleNamespace(
        GetCollisionEnabledAttr=lambda:SimpleNamespace(Get=lambda:prim.enabled))})()
    proxy_shape.apis=(collision_api,);unknown.apis=(collision_api,);disabled.apis=(collision_api,)
    evidence=probe.inspect_instance_proxy_collisions(SimpleNamespace(Traverse=traverse),
        usd=SimpleNamespace(TraverseInstanceProxies=lambda:proxy),
        physics=SimpleNamespace(CollisionAPI=collision_api,RigidBodyAPI=rigid,Joint=joint),
        physx=SimpleNamespace(PhysxForceAPI=force), covered_actors={'/env0/hand'}, covered_colliders=set())
    assert evidence['enabled_or_unknown_unlisted_collision_prim_paths'] == ['/global/shape']
    assert evidence['collision_prims'][0]['actor_path']=='/env0/hand'
    assert evidence['collision_prims'][0]['instance_proxy']
    assert evidence['external_source_complete'] is False


def test_force_tap_observes_original_arguments_and_same_step_only():
    from types import SimpleNamespace
    force=torch.tensor([[0.,0.,4.]])
    torque=torch.tensor([[0.,1.,0.]])
    indices=torch.tensor([0],dtype=torch.int32)
    state=[(1, 18, .0375)]
    calls=[]
    def apply(force_data,torque_data,position_data,indices,is_global):
        calls.append((force_data,torque_data,position_data,indices,is_global))
        return 17
    view=SimpleNamespace(prim_paths=['/env0/Wire'], apply_forces_and_torques_at_position=apply)
    tap=probe.WireRootForceTap(view,lambda:state[0])
    with tap.install():
        assert view.apply_forces_and_torques_at_position(force, torque, None, indices, True)==17
        assert tap.observed_step((0,17,.035))['associated_calls']==[]
        result=tap.observed_step((1,19,.039583333333333))
        assert len(result['associated_calls'])==1
        call=result['associated_calls'][0]
        assert call['before']==call['after']==[1,18,.0375]
        assert call['force_data']['values']==[[0.,0.,4.]]
        assert call['is_global'] and call['target_actor_paths']==['/env0/Wire']
        assert result['external_wrench_absence_proven'] is False
    assert view.apply_forces_and_torques_at_position is apply
    assert len(calls)==1 and calls[0][0] is force and calls[0][3] is indices


def test_force_tap_preserves_failed_original_write_and_unmatched_calls():
    from types import SimpleNamespace
    def apply(*args):
        raise RuntimeError('SDK rejected fixture')
    view=SimpleNamespace(prim_paths=['/env0/Wire'], apply_forces_and_torques_at_position=apply)
    tap=probe.WireRootForceTap(view,lambda:(5,22,.04))
    with tap.install():
        with pytest.raises(RuntimeError,match='SDK rejected'):
            view.apply_forces_and_torques_at_position(None,None,None,torch.tensor([0]),False)
    assert tap.unassociated_calls[0]['outcome']=='raised'
    assert tap.unassociated_calls[0]['error']=='RuntimeError: SDK rejected fixture'
    assert tap.observed_step((6,24,.05))['unmatched_older_calls'][0]['outcome']=='raised'
