import numpy as np
import pytest
from scripts.solve_dh116_right_pinch_owned_surfaces import owned_targets

def fixture():
    nodes=np.stack([np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.305)],axis=1)
    p=np.array([[.437,-.2008,.309],[.424,-.2102,.311]])
    return p,nodes

def test_actual_right_station_order_is_preserved_not_reflected():
    p,nodes=fixture()
    r=owned_targets(p,nodes,-.2)
    assert r["node_indices"]==[4,3]
    np.testing.assert_allclose(r["normals_world"][:,2],-.2)
    assert r["normals_world"][0,0]<0<r["normals_world"][1,0]
    np.testing.assert_allclose(r["points_world_m"]+.00485*r["normals_world"],nodes[[4,3]])
    assert r["native_contact_observed"] is False and r["reference_only"] is True

@pytest.mark.parametrize("damage",["same_side","same_node","bent","normal","unknown"])
def test_owned_targets_fail_closed_for_ambiguous_geometry(damage):
    p,nodes=fixture();nz=-.2
    if damage=="same_side":p[1,0]=.437
    if damage=="same_node":p[1,1]=p[0,1]
    if damage=="bent":nodes[10,0]+=.001
    if damage=="normal":nz=-1.
    if damage=="unknown":p[0,0]=np.nan
    with pytest.raises(ValueError):owned_targets(p,nodes,nz)
