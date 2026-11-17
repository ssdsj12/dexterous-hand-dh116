"""Independent same-episode bilateral Pinch plan; original one-hand scopes read-only."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from scripts import prepare_dh116_right_pinch_owned_contact as right
from scripts import prepare_dh116_pinch_margin_contact as left

VERSION="dh116-bilateral-owned-pinch-plan-v1"
ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/dh116_tasks/flexible_bilateral_owned_pinch_v1.json"
SETUP_STEPS,MEASURED_STEPS,TOTAL_STEPS=2,232,234

def derive_pair():
    pair=deepcopy(right.load_pair())
    if pair["left"]!=left.load_pair()["left"]:
        raise ValueError("original independent left pose required")
    pair.update(version=VERSION,scope="bilateral_same_episode_contact_candidate_not_teacher",
        limitations=["Same episode contact requires NEW native evidence.",
            "Historical single-hand episodes must not be combined as bilateral evidence.",
            "Table remains present; 234 steps is not 5 second Hold or grasp qualification."])
    return pair

def load_pair(project=ROOT):
    value=json.loads((Path(project)/CONFIG).read_text())
    if value!=derive_pair():raise ValueError("frozen bilateral pair differs")
    return value

def action_schedule(packet=None):
    from go2_pvcnn.control.dh116.tabletop_surface_contract import validate_actions
    pair=derive_pair() if packet is None else packet
    if pair!=derive_pair():raise ValueError("frozen bilateral schedule input differs")
    value=left.action_schedule()+right.action_schedule(pair)
    validate_actions(value)
    return value

def expected_boundary(packet,ordinal):
    value=right.expected_boundary(packet,ordinal)
    c=packet["left"]["candidate"]
    value["left_q_rad"]=np.asarray(c["active_q_rad"])+value["fraction"]*np.asarray(c["active_delta_rad"])
    return value
