"""Native pad readback must keep hand/body/environment correspondence."""
from types import SimpleNamespace
import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_multifinger_runtime import MultifingerLiftControl


def setup_control(mixed=False):
    hands=[SimpleNamespace(body_names=['wrist_fixed','finger14_link','finger23_link']) for _ in range(2)]
    encoded=[[-10000.,-20000.],[0.,0.],[-20000.,-10000.],[-10000.,-10000.]]
    sensors=[]
    for i,values in enumerate(encoded):
        data=torch.tensor([[[1.,.8,v]] for v in values],dtype=torch.float64)
        if mixed and i==0:
            hard=data.clone();hard[:,:,2]=0.;data=torch.cat((data,hard),1)
        sensors.append(SimpleNamespace(body_physx_view=SimpleNamespace(get_material_properties=lambda data=data:data)))
    control=object.__new__(MultifingerLiftControl)
    control.device='cpu';control.n=2
    control.w=SimpleNamespace(hands=hands,t=lambda value:torch.as_tensor(value,dtype=torch.float64),
        w=SimpleNamespace(sensor_links=['finger14_link','finger23_link'],sensors=sensors))
    return control


def test_native_compliance_mapping_keeps_body_and_environment_order():
    result=setup_control()._read_body_compliance()
    torch.testing.assert_close(result[0],torch.tensor([[0.,.0001,0.],[0.,.00005,0.]],dtype=torch.float64))
    torch.testing.assert_close(result[1],torch.tensor([[0.,.00005,.0001],[0.,.0001,.0001]],dtype=torch.float64))


def test_mixed_native_stiffness_on_one_body_is_rejected_explicitly():
    with pytest.raises(RuntimeError,match='mixed native collider stiffness'):
        setup_control(mixed=True)._read_body_compliance()
