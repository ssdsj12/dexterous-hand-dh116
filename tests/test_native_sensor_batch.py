"""Read-only native sensor parity on CPU; no Isaac imports or process."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


MODULE = Path(__file__).parents[1] / 'src/go2_pvcnn/control/dh116/tabletop_native_sensor_batch.py'


def test_public_reader_exists():
    assert MODULE.exists(), 'read_native_sensors sidecar is missing'


@pytest.fixture
def reader():
    assert MODULE.exists(), 'read_native_sensors sidecar is missing'
    spec = importlib.util.spec_from_file_location(
        'go2_pvcnn.control.dh116.tabletop_native_sensor_batch', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_native_sensors


class Sensor:
    def __init__(self, n, capacity, friction_capacity):
        self._timestamp = torch.arange(n, dtype=torch.float64)
        self._timestamp_last_update = self._timestamp - .1
        self.calls = []
        contact = [torch.full((capacity, 1), float('nan')),
                   torch.full((capacity, 3), float('nan')),
                   torch.full((capacity, 3), float('nan')),
                   torch.full((capacity, 1), float('nan')),
                   torch.zeros(n, 1, dtype=torch.int32),
                   torch.zeros(n, 1, dtype=torch.int32)]
        friction = [torch.full((friction_capacity, 3), float('nan')),
                    torch.full((friction_capacity, 3), float('nan')),
                    torch.zeros(n, 1, dtype=torch.int32),
                    torch.zeros(n, 1, dtype=torch.int32)]
        view = NS(filter_count=1, matrix=torch.zeros(n, 1, 3), contact=contact, friction=friction)
        for name, field in (('get_contact_force_matrix', 'matrix'),
                            ('get_contact_data', 'contact'), ('get_friction_data', 'friction')):
            def read(dt, name=name, field=field):
                self.calls.append((name, dt))
                value = getattr(view, field)
                return value if field == 'matrix' else tuple(value)
            setattr(view, name, read)
        self.contact_physx_view = view

    def update(self, *args, **kwargs):
        raise AssertionError('sensor.update must not be called')

    @property
    def data(self):
        raise AssertionError('sensor.data must not be read')


def world(n=2, capacity=12, friction_capacity=9):
    return NS(n=n, dt=1/480, device='cpu',
              sensors=[Sensor(n, capacity, friction_capacity) for _ in range(34)])


def put(w, sensor=0, env=0, slot=0, forces=(1., 9.), friction_slot=4):
    view=w.sensors[sensor].contact_physx_view
    k=len(forces)
    view.contact[0][slot:slot+k, 0]=torch.tensor(forces)
    view.contact[1][slot:slot+k]=torch.arange(k*3).reshape(k, 3).float()+sensor+env*10
    view.contact[2][slot:slot+k]=torch.tensor([0., 1., 0.])
    view.contact[3][slot:slot+k]=-.001
    view.contact[4][env]=k; view.contact[5][env]=slot
    # Independent counts AND starts: friction is not indexed by contact owners.
    view.friction[0][friction_slot]=torch.tensor([sensor+.25, -.5, env+1.])
    view.friction[1][friction_slot]=torch.tensor([1., 2., 3.])
    view.friction[2][env]=1; view.friction[3][env]=friction_slot
    # Deliberately unrelated to scalar*normal: this matrix must be returned as-is.
    view.matrix[env, 0]=torch.tensor([sensor+.125, env+.75, -1.25])


def snapshot(w):
    return [([v.clone() for v in (s.contact_physx_view.matrix,
             *s.contact_physx_view.contact, *s.contact_physx_view.friction)],
             s._timestamp.clone(), s._timestamp_last_update.clone()) for s in w.sensors]


def unchanged(w, before):
    for s, (buffers, clock, last) in zip(w.sensors, before):
        for v, old in zip((s.contact_physx_view.matrix, *s.contact_physx_view.contact,
                           *s.contact_physx_view.friction), buffers):
            torch.testing.assert_close(v, old, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(s._timestamp, clock, rtol=0, atol=0)
        torch.testing.assert_close(s._timestamp_last_update, last, rtol=0, atol=0)


def reference(w):
    vectors=[]; friction=[]; points=[]
    for sensor in w.sensors:
        v=sensor.contact_physx_view
        vectors.append(v.matrix.reshape(w.n, 3).double())
        ff=[]; pp=[]
        for e in range(w.n):
            k, start=int(v.contact[4][e]), int(v.contact[5][e])
            pp.append(v.contact[1][start:start+k].mean(0))
            k, start=int(v.friction[2][e]), int(v.friction[3][e])
            ff.append(v.friction[0][start:start+k].sum(0))
        points.append(torch.stack(pp).double()); friction.append(torch.stack(ff).double())
    return {key:torch.stack(value, 1).reshape(w.n, 2, 17, 3)
            for key,value in (('vectors',vectors),('friction',friction),('contact_positions',points))}


@pytest.mark.parametrize('wrapped', [False, True])
def test_all_34_unsorted_buffers_match_old_aggregates_and_are_read_only(reader, wrapped):
    w=world()
    for i in range(33):
        put(w, i, env=0, slot=6, friction_slot=2)
        put(w, i, env=1, slot=1, forces=(-.002, 0., 3.), friction_slot=7)
    expected=reference(w); before=snapshot(w)
    out=reader(NS(n=w.n, dt=w.dt, w=w) if wrapped else w)
    assert set(out)=={'raw_contacts','vectors','friction','contact_positions','bad_env'}
    assert len(out['raw_contacts'])==34 and all(isinstance(v,tuple) for v in out['raw_contacts'])
    for current,s in zip(out['raw_contacts'],w.sensors):
        assert all(a is b for a,b in zip(current[:4],s.contact_physx_view.contact[:4]))
        for actual,expected_count in zip(current[4:],s.contact_physx_view.contact[4:]):
            torch.testing.assert_close(actual,expected_count,rtol=0,atol=0)
    for key,value in expected.items():
        assert out[key].dtype==torch.float64
        torch.testing.assert_close(out[key],value,rtol=0,atol=0,equal_nan=True)
    assert not out['bad_env'].any()
    assert torch.isnan(out['contact_positions'][:,1,16]).all()
    assert not out['friction'][:,1,16].any()
    for s in w.sensors:
        assert sorted(name for name,_ in s.calls)==sorted([
            'get_contact_force_matrix','get_contact_data','get_friction_data'])
        assert all(dt==w.dt for _,dt in s.calls)
    unchanged(w,before)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_physx_contact_and_friction_share_metadata_buffers_but_not_owners(reader,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    w=world();w.device=device
    put(w,env=0,slot=1,forces=(1.,9.),friction_slot=0)
    put(w,env=1,slot=6,forces=(2.,3.),friction_slot=4)
    for sensor in w.sensors:
        view=sensor.contact_physx_view
        view.matrix=view.matrix.to(device)
        view.contact=[x.to(device) for x in view.contact]
        view.friction=[x.to(device) for x in view.friction]
    expected=reference(w)
    view=w.sensors[0].contact_physx_view
    shared_count=view.contact[4].clone();shared_start=view.contact[5].clone()
    def read_contact(dt):
        shared_count.copy_(view.contact[4]);shared_start.copy_(view.contact[5])
        return (*view.contact[:4],shared_count,shared_start)
    def read_friction(dt):
        shared_count.copy_(view.friction[2]);shared_start.copy_(view.friction[3])
        return (*view.friction[:2],shared_count,shared_start)
    view.get_contact_data=read_contact;view.get_friction_data=read_friction
    result=reader(w)
    assert not result['bad_env'].any()
    for key,value in expected.items():
        torch.testing.assert_close(result[key],value,rtol=0,atol=0,equal_nan=True)
    read_friction(w.dt)
    torch.testing.assert_close(result['raw_contacts'][0][4],view.contact[4],rtol=0,atol=0)
    torch.testing.assert_close(result['raw_contacts'][0][5],view.contact[5],rtol=0,atol=0)


def test_empty_zero_capacity_and_nan_padding(reader):
    for capacity in (0,12):
        w=world(capacity=capacity,friction_capacity=capacity)
        out=reader(w)
        assert not out['bad_env'].any()
        assert torch.isnan(out['contact_positions']).all()
        assert not out['friction'].any() and not out['vectors'].any()


@pytest.mark.parametrize('field', ['count','start','overrun','count_nan','start_nan','fraction'])
@pytest.mark.parametrize('kind', ['contact','friction'])
def test_invalid_interval_isolates_only_its_environment(reader, kind, field):
    w=world(n=4,capacity=24,friction_capacity=24)
    for e in range(4): put(w,env=e,slot=e*4,friction_slot=e*4)
    v=getattr(w.sensors[0].contact_physx_view,kind)
    ci,si=(4,5) if kind=='contact' else (2,3)
    if field=='count': v[ci][1]=-1
    elif field=='start': v[si][1]=-1
    elif field=='overrun': v[ci][1]=25
    else:
        i=si if field=='start_nan' else ci
        v[i]=v[i].float(); v[i][1]=1.5 if field=='fraction' else float('nan')
    before=snapshot(w); out=reader(w)
    assert out['bad_env'].tolist()==[False,True,False,False]
    assert torch.isfinite(out['contact_positions'][2,0,0]).all()
    assert torch.isfinite(out['friction'][2,0,0]).all()
    unchanged(w,before)


@pytest.mark.parametrize('kind', ['contact','friction'])
@pytest.mark.parametrize('nested', [False,True])
def test_overlap_marks_all_conflicting_owners_without_poisoning_other_envs(reader,kind,nested):
    w=world(n=4,capacity=24,friction_capacity=24)
    v=getattr(w.sensors[0].contact_physx_view,kind)
    ci,si=(4,5) if kind=='contact' else (2,3)
    for value in v[:ci]: value.zero_()
    v[ci][:,0]=torch.tensor([10 if nested else 3,1,1,0])
    v[si][:,0]=torch.tensor([0,2,5 if nested else 15,24])
    out=reader(w)
    assert out['bad_env'].tolist()==[True,True,nested,False]


@pytest.mark.parametrize('kind,index', [('contact',0),('contact',1),('contact',2),('contact',3),
                                     ('friction',0),('friction',1)])
def test_only_owned_nonfinite_values_fault_and_release_points_are_not_force_weighted(reader,kind,index):
    w=world()
    put(w,env=0,slot=6,forces=(-1e-12,0.),friction_slot=7)
    put(w,env=1,slot=1,friction_slot=2)
    expected=reference(w)
    out=reader(w)
    assert not out['bad_env'].any()
    torch.testing.assert_close(out['contact_positions'],expected['contact_positions'],equal_nan=True)
    buf=getattr(w.sensors[0].contact_physx_view,kind)
    buf[index][6 if kind=='contact' else 7,0]=float('nan')
    out=reader(w)
    assert out['bad_env'].tolist()==[True,False]
    assert torch.isfinite(out['contact_positions'][1,0,0]).all()


def test_friction_uses_its_own_counts_and_slot_owners(reader):
    w=world()
    put(w,env=0,slot=6,friction_slot=1)
    put(w,env=1,slot=1,friction_slot=6)
    v=w.sensors[0].contact_physx_view
    v.friction[0][7]=torch.tensor([4.,5.,6.]); v.friction[1][7]=0
    v.friction[2][1]=2
    expected=reference(w);out=reader(w)
    torch.testing.assert_close(out['friction'],expected['friction'],rtol=0,atol=0)
    assert out['friction'][1,0,0,0]==4.25


def test_nonfinite_force_matrix_and_reduction_overflow_are_isolated(reader):
    w=world()
    put(w,env=0,forces=(1.,2.));put(w,env=1,slot=7,friction_slot=2)
    v=w.sensors[0].contact_physx_view
    v.matrix[0,0,0]=float('inf')
    out=reader(w)
    assert out['bad_env'].tolist()==[True,False]
    assert torch.isinf(out['vectors'][0,0,0,0])
    v.matrix.zero_();v.contact[1][:2]=torch.finfo(torch.float32).max
    assert reader(w)['bad_env'].tolist()==[True,False]


@pytest.mark.parametrize('unsupported',['sensor_count','capacity','friction_capacity','point_shape',
    'friction_point_shape','count_shape','matrix_shape','filters','tuple_length','n'])
def test_unsupported_layout_returns_none_without_touching_clock_or_buffers(reader,unsupported):
    w=world();v=w.sensors[0].contact_physx_view
    if unsupported=='sensor_count': w.sensors.pop()
    elif unsupported=='capacity':
        for i in range(4):v.contact[i]=v.contact[i][:-1]
    elif unsupported=='friction_capacity':
        for i in range(2):v.friction[i]=v.friction[i][:-1]
    elif unsupported=='point_shape':v.contact[1]=v.contact[1][:,:2]
    elif unsupported=='friction_point_shape':v.friction[1]=v.friction[1][:,:2]
    elif unsupported=='count_shape':v.contact[4]=v.contact[4][:-1]
    elif unsupported=='matrix_shape':v.matrix=v.matrix[:,:,:2]
    elif unsupported=='filters':v.filter_count=2
    elif unsupported=='tuple_length':v.contact.pop()
    else:w.n=0
    before=snapshot(w)
    assert reader(w) is None
    unchanged(w,before)
