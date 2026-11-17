"""Raw measured tracking is distinct from legal SDK targets."""
import numpy as np
import pytest
from scripts.audit_dh116_right_owned_evidence import measured_tracking,stat
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand

def hand():
 q=expand([.3,.4,.6,1.,1.,1.])
 names=["wrist_x","wrist_y","wrist_z","wrist_roll","wrist_pitch","wrist_yaw"]+list(JOINTS)
 a=np.r_[np.zeros(6),q][None]
 return dict(joint_names=names,active_indices=[names.index(n) for n in ACTIVE],
 q=a.copy(),qd=np.zeros_like(a),backend_targets=a.copy(),
 measured_source="physx_backend_get_dof_positions/get_dof_velocities",
 backend_source="physx_backend_get_dof_position_targets")

def test_actual_error_not_hidden_by_legal_target():
 h=hand();i=h["joint_names"].index("finger12");h["q"][0,i]-=.03
 r=measured_tracking(h)
 assert r["active_error_rad"][1]==pytest.approx(-.03)
 assert r["active_targets_rad"][1]==pytest.approx(.4)
 assert r["active_positions_rad"][1]==pytest.approx(.37)
 assert r["passive_coupling_error_rad"]>0

@pytest.mark.parametrize("damage",["source","indices","nan","shape"])
def test_invalid_measurement_rejected(damage):
 h=hand()
 if damage=="source":h["measured_source"]="command"
 elif damage=="indices":h["active_indices"]=h["active_indices"][::-1]
 elif damage=="nan":h["q"][0,0]=np.nan
 elif damage=="shape":h["q"]=h["q"][0]
 with pytest.raises(ValueError):measured_tracking(h)

def test_no_samples_are_unknown_not_zero():
 assert stat([]) is None

def test_statistics_keep_signed_negative_displacement():
 s=stat([-.003,-.002,-.001])
 assert s["max"]==-.001 and s["p50"]==-.002
