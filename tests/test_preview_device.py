import pytest
from go2_pvcnn.control.dh116.tabletop_preview_contract import freeze, verify
from go2_pvcnn.control.dh116.tabletop_preview_device import validate_device

def test_gpu0_pci_binding_accepts_same_physical_device():
    assert validate_device(0,mask="0",physical_bus="00000000:31:00.0",
                           logical_bus="0000:31:00.0",logical_count=1) == 0

@pytest.mark.parametrize("kwargs",[
    dict(mask="3"),dict(logical_bus="0000:32:00.0"),dict(logical_count=2)
])
def test_gpu0_binding_rejects_wrong_device(kwargs):
    args=dict(mask="0",physical_bus="0000:31:00.0",logical_bus="0000:31:00.0",logical_count=1)
    args.update(kwargs)
    with pytest.raises(ValueError): validate_device(0,**args)

def test_contract_binds_selected_gpu0(tmp_path):
    for name in ("src","scripts","configs","docs/contracts"):
        (tmp_path/name).mkdir(parents=True)
    kit=tmp_path/"render.kit";kit.write_text("kit")
    contract=freeze(tmp_path,kit,physical_gpu_index=0)
    assert contract["physical_gpu_index"]==0
    verify(tmp_path,contract)
    contract["physical_gpu_index"]=True
    with pytest.raises(ValueError):verify(tmp_path,contract)
