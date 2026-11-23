"""Measured mounted-hand inputs and per-lane smoke references for shared owners."""
from copy import deepcopy
from dataclasses import asdict
import math
from time import perf_counter
import uuid

import numpy as np
import torch

from .tabletop_native_teacher_context import (
    ContextIdentity, ContextMetadata, TaskContextPacket, CONTEXT_FIELDS,
    SMOKE_EXCLUDED, validate_context)
from .tabletop_teacher_mounted_wrist_source import (
    SEED_VERSION, SEED_SOURCE, capture_mounted_hand, adapt_mounted_hands)
from .tabletop_native_teacher_profile import digest
from .tabletop_surface_contract import validate_actions


def lane_clock(source, lane, *, policy_step=0, substep=0):
    b = source._boundary()
    return dict(run_uuid=b.run_uuid,env_id=lane,episode_uid=b.episode_uids[lane],
        reset_id=b.reset_ids[lane],rod_uid=b.rod_uids[lane],physics_step=b.sdk_step,
        rod_tick=b.rod_ticks[lane],timestamp_s=b.policy_time_s,solver_time_s=b.solver_time_s,
        clock_valid=True,policy_step=policy_step,substep=substep)


def read_live_common(source, hand_bindings, seed, *, policy_step):
    """Use original SDK-vs-cache verification, never targets as observations."""
    common = []
    for lane, bindings in enumerate(hand_bindings):
        clock = lambda lane=lane: dict(lane_clock(source,lane,policy_step=policy_step),
            phase="pre_action_before_target_write")
        c = clock()
        root = np.array(seed["authored_root_seed_w"],dtype=float,copy=True)
        root[:,:3] += source.origins[lane] - np.asarray(seed["origin"])
        state_seed = dict(version=SEED_VERSION,source=SEED_SOURCE,identity=c,
            authored_hand_roots_w=deepcopy(seed["hand_roots"]),
            environment_origin_w=source.origins[lane].tolist(),
            root_xy=deepcopy(seed["root_xy"]),root_seed_w=root.tolist())
        packets = []
        for binding in bindings:
            container = binding.hand.root_physx_view.link_paths[0][binding.mount.base_id].rsplit("/",1)[0]
            packets.append(capture_mounted_hand(binding.hand,binding.mount,side=binding.side,
                read_identity=clock,data_env_index=0,
                expected_articulation_path=binding.env_paths[lane],
                expected_hand_container_path=container,reset_seed=state_seed))
        common.append(adapt_mounted_hands(packets,expected_identity=c))
    return common


class SharedSmokeReferences:
    """Explicit observed reset references; excludes unavailable contact/hold sources."""

    def __init__(self, source, *, grasp, episode_cap_s, wall_cap_s):
        if (grasp not in ("pinch","tripod","power") or any(
                not math.isfinite(x) or x <= 0 for x in (episode_cap_s,wall_cap_s))):
            raise ValueError("grasp and finite source-owned episode/wall caps required")
        self.source,self.grasp=source,grasp
        self.episode_cap_s,self.wall_cap_s=float(episode_cap_s),float(wall_cap_s)
        self.started=perf_counter()
        self.owner_uid=uuid.uuid4().hex
        self.rows={}

    def reset_lane(self, lane, *, distance_mean_m):
        if (not math.isfinite(distance_mean_m) or distance_mean_m < 0
                or self.source.runtime.ticks[lane] != 0):
            raise ValueError("actual fresh Reset nodes and measured distance required")
        c = ContextIdentity.from_clock(lane_clock(self.source,lane))
        p = torch.as_tensor(self.source.runtime.solver.positions())[lane]
        rest = self.source.runtime.solver.parameter_snapshot()["rest_lengths_m"][lane]
        if p.shape != (41,3) or not torch.isfinite(p).all():
            raise ValueError("actual complete reset node positions required")
        self.rows[lane]=dict(identity=c,reset_z=(float(p[:,2].median()),float(p[:,2].min())),
            rest=float(np.asarray(rest).sum()),distance=float(distance_mean_m),
            previous=torch.zeros(24),token="measured-reset:"+c.episode_uid)

    def set_previous_request(self, actions, token):
        validate_actions(actions)
        if actions.shape != (len(self.source.runtime.ticks),24) or not isinstance(token,str) or not token:
            raise ValueError("one actually recorded normalized request per row and token required")
        for i,row in self.rows.items():
            row["previous"]=actions[i].detach().cpu().float().clone()
            row["token"]=token

    def contexts(self):
        result=[]
        b=self.source._boundary()
        remaining=max(0.,self.wall_cap_s-(perf_counter()-self.started))/self.wall_cap_s
        for i in range(b.parent_num_envs):
            row=self.rows[i];c=ContextIdentity.from_clock(lane_clock(self.source,i))
            if c.epoch != row["identity"].epoch:
                raise ValueError("new native Reset epoch needs fresh lane references")
            values=torch.full((1,50),torch.nan,dtype=torch.float32)
            valid=torch.zeros_like(values,dtype=torch.bool)
            def setv(index,value):
                values[:,index]=torch.as_tensor(value,dtype=torch.float32);valid[:,index]=True
            setv(slice(0,4),[1.,0.,0.,0.])
            setv(slice(10,12),[0.,0.])
            setv(slice(12,36),row["previous"])
            d=row["distance"]
            setv(slice(36,43),[*row["reset_z"],max(d,.005),d,.03,5.,.0005])
            setv(43,(c.timestamp_s-row["identity"].timestamp_s)/self.episode_cap_s)
            setv(44,remaining);setv(45,0.);setv(slice(48,50),[0.,0.])
            excluded=(*SMOKE_EXCLUDED,8,9,46)
            applicable=tuple(i not in excluded for i in range(50))
            metadata=ContextMetadata(self.owner_uid,(c,),self.source.profile.digest,self.source.scope,
                "approach_wiring_smoke_v1",tuple(f["source"] for f in CONTEXT_FIELDS),applicable,
                tuple(None if a else "unqualified smoke source or no material anchor" for a in applicable),
                (("target_material",row["identity"].episode_uid+":material41"),),row["token"],self.grasp)
            packet=TaskContextPacket(values,valid,metadata)
            validate_context(packet,backend_profile=self.source.profile,identities=(c,))
            result.append(packet)
        return result
