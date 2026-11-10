"""Opt-in fixed-wrist, one-control-horizon nonlinear geometric proposals.

No simulator calls, force commands, budget writes or measured-contact claims.
The response model is an explicit approximation, not a validated drive model.
Every submitted post-guard command requires a new ``audit_commands`` call at
the actual same-clock snapshot. The runner still owns native provenance.
"""
from copy import deepcopy
import math
import time

import numpy as np

from .linear_coupling import ACTIVE, JOINTS, coupling_matrix
from .tabletop_surface_contract import ACTION_CONTRACT
from .tabletop_tripod_nonlinear_geometry import audit_pad_region

DT = 1 / 480
ACTIVE_IDS = [JOINTS.index(v) for v in ACTIVE]
DIGITS = ('thumb', 'index', 'middle')
JOINT_GROUPS = ((0, 1), (2,), (3,))
RATE = ACTION_CONTRACT['joint_rate_rad_s']
LEAD = .1  # tabletop_free_wrist_grasp.CONTRACT, unchanged native projection.
APPROACH_SPEED_M_S = .001  # Existing acquisition limit, not a drive override.


def _has_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return True
    if isinstance(v, np.ndarray):
        return v.dtype.kind == 'b' or (v.dtype.kind == 'O' and any(_has_bool(x) for x in v.flat))
    return isinstance(v, (list, tuple)) and any(_has_bool(x) for x in v)


def _finite(v, shape, name, dtype=np.float64):
    if _has_bool(v):
        raise ValueError(name+' must be numeric, not boolean')
    if hasattr(v, 'detach'):
        v = v.detach().cpu().numpy()
    a = np.asarray(v, dtype=dtype)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name+' requires finite '+str(shape))
    return a.copy()


def _substeps(n):
    if type(n) is not int or not 1 <= n <= 8:
        raise ValueError('explicit one to eight physics substeps required')
    return n


def _logical_command(value, name='logical command'):
    if hasattr(value,'detach'):
        value=value.detach().cpu().numpy()
    dtype=np.asarray(value).dtype
    if dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise ValueError(name+' requires explicit float32/float64 native task dtype')
    return _finite(value,(2,6),name,dtype)


def decode_velocity_action24(current_command, actual_active_q, action24,
                             active_lower, active_upper, *, substeps,
                             response_per_control=.5):
    """Decode native task-dtype target ramps with separate float32 SDK values.

    Mirrors the policy-owned, redistribution-disabled finger path *before*
    native table/pressure/recovery guards. Those guards must be audited after
    execution permission is obtained. A held endpoint is never substituted for
    eight ramped SDK targets. Actual follower residuals are applied separately.
    """
    n = _substeps(substeps)
    if type(response_per_control) not in (int, float) or not 0 < response_per_control <= 1:
        raise ValueError('explicit response in (0,1] required')
    command = _logical_command(current_command)
    dtype=command.dtype.type
    q0 = _finite(actual_active_q, (2, 6), 'actual active q')
    action = _finite(action24, (24,), 'canonical action24', np.float32)
    low = _finite(active_lower, (2, 6), 'active lower', dtype)
    high = _finite(active_upper, (2, 6), 'active upper', dtype)
    if np.any(np.abs(action) > 1):
        raise ValueError('canonical normalized action range required')
    if np.any(action[12:] != 0):
        raise ValueError('fixed wrist planner forbids translational/rotational actions')
    alpha = 1 - (1-float(response_per_control))**(1/8)
    q = q0.astype(np.float64); commands=[]; logical=[]; deltas=[]
    step = dtype(RATE*DT)
    for _ in range(n):
        # Multiplication order matches actions*rate*dt in the existing Torch path.
        target = command+action[:12].reshape(2, 6).astype(dtype)*dtype(RATE)*dtype(DT)
        actual = q.astype(dtype)
        target = np.minimum(np.maximum(target, actual-dtype(LEAD)), actual+dtype(LEAD))
        target = np.minimum(np.maximum(target, low), high)
        target = command+np.clip(target-command, -step, step)
        command = np.minimum(np.maximum(target, low), high).astype(dtype)
        encoded=command.astype(np.float32)
        q = q+alpha*(encoded.astype(np.float64)-q)
        logical.append(command.copy());commands.append(encoded); deltas.append(q-q0.astype(np.float64))
    return dict(action24=action, command_targets=np.asarray(commands, dtype=np.float32),
                logical_command_targets=np.asarray(logical,dtype=dtype),
                predicted_deltas=np.asarray(deltas), response_per_control=float(response_per_control),
                dynamics='unvalidated_first_order_actual_q_tracking_approximation',
                native_contact_verified=False)


def _command_path_bounds(model, snapshot, before, after):
    """Arc-length upper bound of every vertex of each designated source pad.

    Unlike a chord/Hausdorff displacement, sum(radius*abs(angle)) bounds the
    complete commanded joint path. It is charged only by the external ledger
    after actual command permission/submission, never by this planner.
    """
    angles = np.abs(coupling_matrix()@(np.asarray(after)-np.asarray(before)))
    extents = [2*np.linalg.norm(e['t0'][:3, 3])+np.linalg.norm(b[:3, 3])
               for e, b in zip(model.edges, snapshot['relative'])]
    out=[]
    for digit in DIGITS:
        patch = model.patches[digit]
        shape = next(s for s in model.shapes if s['body'] == patch['body'])
        path=model.paths[patch['body']]; bound=0.
        for k, edge_id in enumerate(path):
            edge=model.edges[edge_id]
            if edge['axis'] is None:
                continue
            radius=(np.linalg.norm(edge['t0'][:3, 3])+sum(extents[j] for j in path[k:])
                    +np.linalg.norm(shape['vertices'], axis=-1).max())
            bound += radius*angles[JOINTS.index(edge['name'])]
        out.append(float(bound))
    return np.asarray(out)


class TripodCandidatePlanner:
    """Finite right-pad candidates with a complete, conservative bimanual audit.

    H=one 60-Hz control action, 1..8 explicit 480-Hz substeps. Both actual hands
    track their absolute commands in the approximate prediction. A single
    whole-trajectory joint-box certificate covers all intermediate poses; a
    failed conservative certificate rejects the candidate, never falls back
    to sampled endpoint acceptance. Conservative false negatives are expected.
    """
    def __init__(self, geometry, *, active_lower, active_upper, full_lower, full_upper,
                 limit_sources, response_per_control=.5, max_candidates=4):
        if not isinstance(limit_sources, dict) or any(not limit_sources.get(k)
                for k in ('active', 'full', 'lead', 'response', 'command')):
            raise ValueError('explicit original limit sources required')
        self.low=_finite(active_lower,(2,6),'active lower')
        self.high=_finite(active_upper,(2,6),'active upper')
        self.flow=_finite(full_lower,(2,11),'full lower')
        self.fhigh=_finite(full_upper,(2,11),'full upper')
        if np.any(self.low > self.high) or np.any(self.flow > self.fhigh):
            raise ValueError('ordered original joint limits required')
        if type(max_candidates) is not int or not 1 <= max_candidates <= 4:
            raise ValueError('one to four finite candidates required')
        if type(response_per_control) not in (int,float) or not 0 < response_per_control <= 1:
            raise ValueError('explicit response in (0,1] required')
        if len(geometry.models) != 2 or any(len(m.shapes) != 17 for m in geometry.models):
            raise ValueError('all 34 source hulls required')
        if any(not set(DIGITS).issubset(m.patches) for m in geometry.models):
            raise ValueError('three independently defined source pad regions required')
        self.geometry=geometry; self.limit_sources=deepcopy(limit_sources)
        self.response=float(response_per_control);self.max_candidates=max_candidates

    def _snapshot(self, snapshots, full_qd, snapshot_identity):
        if (not isinstance(snapshot_identity,dict)
                or any(type(snapshot_identity.get(k)) is not int or snapshot_identity[k] < 0
                       for k in ('env_id','episode_id','physics_step'))
                or not isinstance(snapshot_identity.get('source_snapshot_id'),str)
                or not snapshot_identity['source_snapshot_id']):
            raise ValueError('same-clock snapshot identity required')
        if len(snapshots) != 2:
            raise ValueError('ordered bimanual actual snapshots required')
        qd=_finite(full_qd,(2,11),'same-clock actual full velocity')
        errors=[]; full=[]
        for h,(model,snapshot) in enumerate(zip(self.geometry.models,snapshots)):
            if snapshot.get('measurement_identity') != snapshot_identity:
                raise ValueError('snapshot measurement identity mismatch')
            if not np.array_equal(_finite(snapshot.get('full_qd'),(11,),'snapshot full velocity'),qd[h]):
                raise ValueError('snapshot velocity identity mismatch')
            full.append(_finite(snapshot['full_q'],(11,),'actual full q'))
            predicted=model.predict(snapshot,np.zeros(6))
            if len(predicted['hulls']) != 17:
                raise ValueError('all seventeen source hulls required')
            errors.extend(float(np.max(np.abs(predicted['poses'][body]-pose)))
                          for body,pose in snapshot['poses'].items())
        return np.asarray(full),dict(passed=max(errors,default=math.inf)<1e-12,
            maximum_pose_error=max(errors,default=None),source_hulls=[17,17],
            native_kinematics_validated=False,
            scope='zero-increment measured-frame rebasing consistency only')

    def _audit_path(self, snapshots, full_qd, current_command, snapshot_identity,
                    bearing_mask, bearing_provenance, commands, deltas, action24=None,
                    logical_commands=None):
        full,calibration=self._snapshot(snapshots,full_qd,snapshot_identity)
        current=_logical_command(current_command)
        commands=np.asarray(commands,dtype=np.float32);deltas=np.asarray(deltas,dtype=np.float64)
        if commands.ndim != 3 or commands.shape[1:] != (2,6) or not 1<=len(commands)<=8:
            raise ValueError('one to eight bimanual command targets required')
        if deltas.shape != commands.shape or not np.isfinite(deltas).all() or not np.isfinite(commands).all():
            raise ValueError('finite command/state trajectory required')
        if logical_commands is None:
            if current.dtype != np.float32:
                raise ValueError('double native task requires its actual logical commands')
            logical_commands=commands
        logical_commands=_finite(logical_commands,commands.shape,'logical command trajectory',current.dtype)
        if not np.array_equal(logical_commands.astype(np.float32),commands):
            raise ValueError('logical command does not match encoded SDK command')
        predicted_full=full[None]+np.einsum('jk,shk->shj',coupling_matrix(),deltas)
        actual=full[:,ACTIVE_IDS]
        predicted_actual=actual[None]+deltas
        starts=np.concatenate((actual[None],predicted_actual[:-1]))
        previous_commands=np.concatenate((current[None],logical_commands[:-1]))
        dtype=current.dtype.type
        # Compare native logical codec endpoints exactly in its real dtype.
        # SDK rounding is checked independently, never used to enlarge slew.
        slew_low=previous_commands-dtype(RATE*DT)
        slew_high=previous_commands+dtype(RATE*DT)
        lead_low=starts.astype(dtype)-dtype(LEAD)
        lead_high=starts.astype(dtype)+dtype(LEAD)
        checks=dict(actual_active_limits=bool(np.all((actual>=self.low)&(actual<=self.high))),
            actual_full_limits=bool(np.all((full>=self.flow)&(full<=self.fhigh))),
            current_command_limits=bool(np.all((current>=self.low)&(current<=self.high))),
            command_limits=bool(np.all((logical_commands>=self.low)&(logical_commands<=self.high))),
            command_slew=bool(np.all((logical_commands>=slew_low)&(logical_commands<=slew_high))),
            command_lead=bool(np.all((logical_commands>=lead_low)&(logical_commands<=lead_high))),
            predicted_active_limits=bool(np.all((predicted_actual>=self.low)&(predicted_actual<=self.high))),
            predicted_full_limits=bool(np.all((predicted_full>=self.flow)&(predicted_full<=self.fhigh))),
            zero_rebase_consistency=calibration['passed'])
        reasons=[k.upper() for k,v in checks.items() if not v]
        result=dict(accepted=False,motion_safe=False,audit_complete=False,
            action24=None if action24 is None else np.asarray(action24,dtype=np.float32).copy(),
            first_command=commands[0].copy(),command_targets=commands.copy(),
            first_logical_command=logical_commands[0].copy(),logical_command_targets=logical_commands.copy(),
            predicted_deltas=deltas.copy(),predicted_full_q=predicted_full,
            approach_command_m=sum((_command_path_bounds(self.geometry.models[1],snapshots[1],a[1],b[1])
                for a,b in zip(np.concatenate((current.astype(np.float32)[None],commands[:-1])),commands)),np.zeros(3)),
            distance_kind='source_pad_command_path_arclength_upper_bound_not_actual_motion',
            joint_audit=checks,zero_rebase_calibration=calibration,
            coverage=dict(physics_substeps=len(commands),control_horizon=1,source_hulls=[17,17],
                          swept_method='whole_trajectory_joint_box_certificate'),
            snapshot_identity=deepcopy(snapshot_identity),native_contact_verified=False,
            native_candidate_eligible=False,budget_consumed_m=0,
            evidence_kind='geometric_proposal_not_native_contact',
            dynamics='unvalidated_first_order_actual_q_tracking_approximation',
            response_per_control=self.response,limit_sources=deepcopy(self.limit_sources),
            decoder_guards_scope='policy-owned targets, redistribution disabled; post-native-guard reaudit required')
        if reasons:
            return dict(result,rejection_reasons=reasons)
        # Each active coordinate is monotone on every first-order substep.
        # Include start + ALL states in the box; this covers nonmonotone q
        # histories as well. coupling_matrix entries are all nonnegative.
        all_q=np.concatenate((np.zeros((1,2,6)),deltas))
        lo=all_q.min(axis=0);hi=all_q.max(axis=0);middle=(lo+hi)/2
        bounds=[model.motion_bounds(snapshots[h],lo[h],hi[h]) for h,model in enumerate(self.geometry.models)]
        normals=[model.normal_motion_bounds(lo[h],hi[h]) for h,model in enumerate(self.geometry.models)]
        swept=self.geometry.audit_state(snapshots,middle[1],left_delta=middle[0],
            motion_bounds=bounds,normal_motion_bounds=normals,
            bearing_mask=bearing_mask,bearing_provenance=bearing_provenance)
        result.update(swept=swept,rejection_reasons=swept['rejection_reasons'],
            audit_complete=swept['audit_complete'],motion_safe=swept['motion_safe'],
            accepted=bool(swept['motion_safe'] and swept['audit_complete']))
        return result

    def audit_action(self, *, action24, snapshots, full_qd, current_command,
                     snapshot_identity, bearing_mask, bearing_provenance, substeps=8):
        full, _=self._snapshot(snapshots,full_qd,snapshot_identity)
        path=decode_velocity_action24(current_command,full[:,ACTIVE_IDS],action24,
            self.low,self.high,substeps=substeps,response_per_control=self.response)
        return self._audit_path(snapshots,full_qd,current_command,snapshot_identity,bearing_mask,
            bearing_provenance,path['command_targets'],path['predicted_deltas'],path['action24'],
            logical_commands=path['logical_command_targets'])

    def audit_commands(self, *, final_next_command, snapshots, full_qd, current_command,
                       snapshot_identity, bearing_mask, bearing_provenance,final_logical_command=None):
        """Audit exact float32 post-native-guard SDK target, one physical step.

        Final commands must already be the actual float32 values to be sent to
        SDK. A float64 optimizer value is rejected instead of silently recast.
        """
        if hasattr(final_next_command,'detach'):
            final_next_command=final_next_command.detach().cpu().numpy()
        if np.asarray(final_next_command).dtype != np.float32:
            raise ValueError('encoded SDK final command must be float32')
        command=_finite(final_next_command,(2,6),'encoded final command',np.float32)
        current=_logical_command(current_command)
        if final_logical_command is None:
            if current.dtype!=np.float32:
                raise ValueError('double task requires actual final logical command')
            final_logical_command=command
        logical=_logical_command(final_logical_command,'final logical command')
        if logical.dtype != current.dtype:
            raise ValueError('final logical command native task dtype mismatch')
        if not np.array_equal(logical.astype(np.float32),command):
            raise ValueError('final logical command does not match encoded SDK command')
        full,_=self._snapshot(snapshots,full_qd,snapshot_identity)
        alpha=1-(1-self.response)**(1/8)
        delta=alpha*(command.astype(float)-full[:,ACTIVE_IDS])
        return self._audit_path(snapshots,full_qd,current_command,snapshot_identity,bearing_mask,
            bearing_provenance,command[None],delta[None],logical_commands=logical[None])

    def _regions(self,snapshot,delta):
        model=self.geometry.models[1];prediction=model.predict(snapshot,delta);rows=[]
        for digit in DIGITS:
            patch=model.patches[digit];shape=next(s for s in model.shapes if s['body']==patch['body'])
            rows.append(audit_pad_region(shape['planes'],patch,prediction['poses'][patch['body']],
                self.geometry.scene['rod'],contact_distance_m=self.geometry.thresholds.contact_distance_m,
                normal_cosine_min=self.geometry.thresholds.normal_cosine_min))
        return rows

    @staticmethod
    def _score(row):
        w=row.get('witness')
        if not row.get('complete') or w is None:
            return math.inf
        return abs(w['distance_m'])+.001*max(0.,.8-w['facing_cosine'])

    def plan(self, *, snapshots, full_qd, current_command, missing_right,
             remaining_approach_m, snapshot_identity, bearing_mask, bearing_provenance,
             substeps=8):
        started=time.perf_counter();n=_substeps(substeps)
        missing=np.asarray(missing_right)
        if missing.shape!=(3,) or missing.dtype!=np.bool_:
            raise ValueError('independent right missing-pad bool mask required')
        remaining=_finite(remaining_approach_m,(3,),'remaining approach budget')
        if np.any(remaining<0) or np.any(remaining>.001+1e-12):
            raise ValueError('original remaining per-episode approach budget required')
        full,_=self._snapshot(snapshots,full_qd,snapshot_identity)
        command=_logical_command(current_command)
        baseline=dict(accepted=False,action24=np.zeros(24,np.float32),reason=None,
            budget_consumed_m=0,native_contact_verified=False,native_candidate_eligible=False,
            evidence_kind='geometric_proposal_not_native_contact',
            rod_snapshot=deepcopy(self.geometry.scene['rod']),snapshot_identity=deepcopy(snapshot_identity))
        eligible=missing&(remaining>0)
        if not eligible.any():
            return dict(baseline,reason='NO_ELIGIBLE_RECOVERY_BUDGET',elapsed_s=time.perf_counter()-started)
        # Fail closed at the actual starting SDK pose before proposing any move.
        actual=self.geometry.audit_state(snapshots,np.zeros(6),left_delta=np.zeros(6),
            bearing_mask=bearing_mask,bearing_provenance=bearing_provenance)
        if not actual['motion_safe'] or not actual['audit_complete']:
            return dict(baseline,reason='ACTUAL_SOURCE_GEOMETRY_UNCERTIFIED',audit=actual,
                        rejection_reasons=actual['rejection_reasons'],elapsed_s=time.perf_counter()-started)
        basepath=decode_velocity_action24(command,full[:,ACTIVE_IDS],np.zeros(24,np.float32),
            self.low,self.high,substeps=n,response_per_control=self.response)
        rows=self._regions(snapshots[1],basepath['predicted_deltas'][-1,1])
        targets=[];increments=np.zeros((3,6));step=.0001
        for d,(digit,group) in enumerate(zip(DIGITS,JOINT_GROUPS)):
            witness=rows[d].get('witness')
            targets.append(None if witness is None else dict(digit=digit,
                object_surface_point_w_m=np.asarray(witness['object_surface_point_w_m']).copy(),
                pad_witness_w_m=np.asarray(witness['point_w_m']).copy(),
                object_pose_w=deepcopy(self.geometry.scene['rod']['pose_w']),
                predicted_geometry_only=True))
            if not eligible[d] or witness is None or abs(witness['distance_m'])>.010:
                continue
            gradient=np.zeros(6)
            for k in group:
                scores=[]
                for sign in (-1,1):
                    delta=basepath['predicted_deltas'][-1,1].copy();delta[k]+=sign*step
                    scores.append(self._score(self._regions(snapshots[1],delta)[d]))
                gradient[k]=(scores[1]-scores[0])/(2*step)
            if not np.isfinite(gradient).all() or max(abs(gradient))<1e-12:
                continue
            increments[d]=-gradient/max(abs(gradient))*(RATE*DT*n)
        proposal_specs=[('combined',sum(increments,np.zeros(6)))]+[(DIGITS[d],increments[d]) for d in range(3)]
        ranked=[];seen=set()
        for name,increment in proposal_specs:
            if not np.any(increment):
                continue
            # Arclength, not displacement/gap decrease, bounds the proposed
            # command for EACH independently missing pad including thumb2DOF.
            pathbound=_command_path_bounds(self.geometry.models[1],snapshots[1],command[1],command[1]+increment)
            allowed=np.where(eligible,np.minimum(remaining,APPROACH_SPEED_M_S*DT*n),0.)
            ratio=min([1.]+[allowed[d]/pathbound[d] for d in range(3) if pathbound[d]>0])
            increment=increment*ratio
            action=np.zeros(24,np.float32);action[6:12]=np.clip(increment/(RATE*DT*n),-1.,1.)
            # Floating-point SDK rounding can expand a micrometre-scale cap.
            # Shrink the normalized action and REDECODE; never expand budgets.
            for _ in range(12):
                path=decode_velocity_action24(command,full[:,ACTIVE_IDS],action,self.low,self.high,
                    substeps=n,response_per_control=self.response)
                per_step=np.asarray([_command_path_bounds(self.geometry.models[1],snapshots[1],a[1],b[1])
                    for a,b in zip(np.concatenate((command.astype(np.float32)[None],path['command_targets'][:-1])),path['command_targets'])])
                charged=per_step.sum(axis=0)
                per_step_allowed=np.where(eligible,np.minimum(remaining,APPROACH_SPEED_M_S*DT),0.)
                if np.all(charged<=allowed+1e-15) and np.all(per_step<=per_step_allowed+1e-15):
                    break
                action[6:12]*=np.float32(.8)
            else:
                continue
            if not np.any(path['command_targets'][:,1]-command[1]):
                continue
            key=path['command_targets'].tobytes()
            if key in seen:
                continue
            seen.add(key)
            candidate_rows=self._regions(snapshots[1],path['predicted_deltas'][-1,1])
            before=sum(self._score(rows[d]) for d in range(3) if eligible[d])
            after=sum(self._score(candidate_rows[d]) for d in range(3) if eligible[d])
            if not math.isfinite(after) or not after<before:
                continue
            ranked.append((after-before,name,action,path,charged,candidate_rows))
        ranked.sort(key=lambda item:item[0]);audits=[]
        for improvement,name,action,path,charged,candidate_rows in ranked[:self.max_candidates]:
            audit=self._audit_path(snapshots,full_qd,command,snapshot_identity,bearing_mask,bearing_provenance,
                path['command_targets'],path['predicted_deltas'],action,
                logical_commands=path['logical_command_targets'])
            audits.append(dict(name=name,score_change_m=improvement,accepted=audit['accepted'],audit=audit))
            if audit['accepted']:
                return dict(audit,reason='BOUNDED_GEOMETRIC_PROPOSAL',rod_snapshot=deepcopy(self.geometry.scene['rod']),
                    targets_current_rod=targets,eligible_right=eligible.copy(),
                    per_pad_final_regions=candidate_rows,candidate_audits=audits,
                    elapsed_s=time.perf_counter()-started)
        return dict(baseline,reason='NO_CERTIFIED_IMPROVING_CANDIDATE',targets_current_rod=targets,
            candidate_audits=audits,rejection_reasons=[r for a in audits for r in a['audit']['rejection_reasons']],
            elapsed_s=time.perf_counter()-started)
