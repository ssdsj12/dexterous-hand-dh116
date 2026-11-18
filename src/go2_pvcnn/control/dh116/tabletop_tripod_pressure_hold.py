"""Fresh actual-pressure references, with independent right-pad FK proposals.

Training-only simulated force feedback; no force prediction or hardware claim.
Each call rebuilds measured material points, directions and nonlinear FK from
the supplied same-clock snapshot. No failure counter, fixed Jacobian/normal or
reacquisition ledger is stored. A failed pad can be tried at the next sample.
"""
from copy import deepcopy
import math
import time

import numpy as np

from .tabletop_tripod_candidate_planner import (
    ACTIVE_IDS, DIGITS, JOINT_GROUPS, DT, APPROACH_SPEED_M_S,
    _logical_command, _substeps, _command_path_bounds, _has_bool, decode_velocity_action24)

GOALS_N = (1.5, .75, .75)
CONTRACT = dict(version='tripod-fresh-pressure-hold-v1', goals_n=list(GOALS_N),
    gain_m_per_n_s=.0004,max_speed_m_s=.001,retreat_depth_m=.0006,
    rejected_contact_relief_above_n=.05,bearing_above_n=.05,
    reference_source='tabletop_teacher_pressure_reference.py; tabletop_vision_teacher_pressure_mpc.py',
    force_prediction=False,hardware_calibrated=False,reacquisition_budget_writes=False)


def _numeric(value, shape, name):
    if _has_bool(value):
        raise ValueError(name+' cannot contain boolean measurements')
    if hasattr(value,'detach'):
        value=value.detach().cpu().numpy()
    raw=np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.shape!=shape:
        raise ValueError(name+' requires numeric '+str(shape))
    return raw.astype(np.float64,copy=True)


def pressure_references(*, effective_pad_loads_n,effective_pad_vectors_w,
                        effective_pad_points_w_m,effective_pad_valid,
                        effective_pad_depth_m,rejected_contact_loads_n):
    """Original velocity-reference law, only for actual valid right bearing.

    Null/nonfinite/negative inputs are unknown. A missing pad receives no
    invented force direction and no automatic acquisition velocity.
    """
    loads=_numeric(effective_pad_loads_n,(2,5),'native pad loads')
    vectors=_numeric(effective_pad_vectors_w,(2,5,3),'native pad vectors on hand')
    points=_numeric(effective_pad_points_w_m,(2,5,3),'native pad points')
    depth=_numeric(effective_pad_depth_m,(2,5),'native pad depth')
    rejected=_numeric(rejected_contact_loads_n,(2,),'native rejected hand loads')
    if hasattr(effective_pad_valid,'detach'):
        effective_pad_valid=effective_pad_valid.detach().cpu().numpy()
    valid=np.asarray(effective_pad_valid)
    if valid.shape!=(2,5) or valid.dtype!=np.bool_:
        raise ValueError('native pad validity must be bool 2x5')
    hand_ok=bool(np.isfinite(loads[1]).all() and (loads[1]>=0).all()
        and np.isfinite(rejected[1]) and rejected[1]>=0)
    rows=[]
    for d,digit in enumerate(DIGITS):
        force=vectors[1,d];length=np.linalg.norm(force);point=points[1,d]
        known=bool(hand_ok and np.isfinite(force).all() and np.isfinite(length)
            and np.isfinite(point).all() and np.isfinite(depth[1,d]) and depth[1,d]>=0)
        bearing=bool(known and loads[1,d]>.05 and valid[1,d] and length>0)
        reason=('eligible' if bearing else 'no_actual_bearing' if known and loads[1,d]<=.05
                else 'unknown_measurement')
        speed=float(np.clip(.0004*(GOALS_N[d]-loads[1,d]),-.001,.001)) if bearing else 0.
        relief=bool(bearing and (loads[1,d]>GOALS_N[d] or depth[1,d]>.0006 or rejected[1]>.05))
        if bearing and (depth[1,d]>.0006 or rejected[1]>.05):
            speed=-.001
        rows.append(dict(digit=digit,eligible=bearing,status=reason,
            measured_load_n=float(loads[1,d]) if np.isfinite(loads[1,d]) else None,
            measured_depth_m=float(depth[1,d]) if np.isfinite(depth[1,d]) else None,
            rejected_hand_load_n=float(rejected[1]) if np.isfinite(rejected[1]) else None,
            goal_n=GOALS_N[d],speed_m_s=speed,
            actual_point_w_m=point.copy() if np.isfinite(point).all() else None,
            actual_force_on_hand_w_n=force.copy() if np.isfinite(force).all() else None,
            actual_normal_on_hand_w=force/length if bearing else None,
            inward_w=-force/length if bearing else None,mandatory_relief=relief,
            must_preserve_upstream_relief=False,relief_blocked=False,
            predicted_normal_displacement_m=0.,native_contact_verified=False,
            reference_kind='simulated_measurement_velocity_reference_not_force_prediction'))
    return rows


class TripodPressureHold:
    """Stateless per-pad pressure geometry; full validator owns final admission.

    Only right thumb (two joints), index and middle commands can be proposed.
    Actual left tracking is still included by the bimanual validator. References
    end at the current measured contact point plus a bounded velocity step;
    commanded wrist transport is never added. Caller must post-guard reaudit.
    """
    def __init__(self,planner):
        if not hasattr(planner,'audit_action') or not hasattr(planner,'geometry'):
            raise ValueError('complete bimanual nonlinear command auditor required')
        self.planner=planner

    def _point_path(self,ctx,action,local_points,n):
        snapshots=ctx['snapshots'];full=np.stack([s['full_q'] for s in snapshots])
        path=decode_velocity_action24(ctx['current_command'],full[:,ACTIVE_IDS],action,
            self.planner.low,self.planner.high,substeps=n,response_per_control=self.planner.response)
        model=self.planner.geometry.models[1];points=[];normals=[]
        for delta in path['predicted_deltas'][:,1]:
            predicted=model.predict(snapshots[1],delta)
            points.append({digit:predicted['poses'][model.patches[digit]['body']][:3,:3]@local+
                predicted['poses'][model.patches[digit]['body']][:3,3] for digit,local in local_points.items()})
            # All source facet directions are freshly rotated by full FK. These
            # are geometric normals, never fabricated future contact forces.
            normals.append({digit:[v['normal_w'].copy() for v in predicted['pad_facets'][digit]]
                            for digit in local_points})
        previous=np.concatenate((_logical_command(ctx['current_command']).astype(np.float32)[None],path['command_targets'][:-1]))
        costs=np.asarray([_command_path_bounds(model,snapshots[1],a[1],b[1])
                          for a,b in zip(previous,path['command_targets'])])
        return path,points,normals,costs

    def _bounded(self,ctx,action,local_points,n):
        """Re-encode after shrinking; no budget or speed can be borrowed."""
        action=action.astype(np.float32,copy=True)
        for _ in range(16):
            path,points,normals,costs=self._point_path(ctx,action,local_points,n)
            if np.all(costs<=APPROACH_SPEED_M_S*DT+1e-15):
                return action,path,points,normals,costs
            factor=min(.8,.95*APPROACH_SPEED_M_S*DT/max(float(costs.max()),1e-30))
            action[6:10]*=np.float32(factor)
        return None

    def plan(self,*,snapshots,full_qd,current_command,snapshot_identity,bearing_mask,bearing_provenance,
             effective_pad_loads_n,effective_pad_vectors_w,effective_pad_points_w_m,effective_pad_valid,
             effective_pad_depth_m,rejected_contact_loads_n,upstream_pressure_action24,substeps=8):
        started=time.perf_counter();n=_substeps(substeps)
        ctx=dict(snapshots=snapshots,full_qd=full_qd,current_command=current_command,
            snapshot_identity=snapshot_identity,bearing_mask=bearing_mask,bearing_provenance=bearing_provenance)
        self.planner._snapshot(snapshots,full_qd,snapshot_identity)
        _logical_command(current_command)
        refs=pressure_references(effective_pad_loads_n=effective_pad_loads_n,
            effective_pad_vectors_w=effective_pad_vectors_w,effective_pad_points_w_m=effective_pad_points_w_m,
            effective_pad_valid=effective_pad_valid,effective_pad_depth_m=effective_pad_depth_m,
            rejected_contact_loads_n=rejected_contact_loads_n)
        upstream=_numeric(upstream_pressure_action24,(24,),'upstream pressure action').astype(np.float32)
        if not np.isfinite(upstream).all() or (abs(upstream)>1).any():
            raise ValueError('finite canonical upstream action24 required')
        result=dict(accepted=False,action24=np.zeros(24,np.float32),per_pad=refs,audit=None,
            candidate_audits=[],budget_consumed_m=0.,reacquisition_budget_untouched=True,
            force_prediction=False,native_contact_verified=False,native_candidate_eligible=False,
            evidence_kind='fresh_pressure_geometric_proposal_not_native_contact',
            contract=deepcopy(CONTRACT),snapshot_identity=deepcopy(snapshot_identity),
            original_upstream_pressure_action24=upstream.copy(),
            per_substep_command_path_m=np.zeros((n,3)),post_native_guard_reaudit_required=True)
        mask=bearing_mask.detach().cpu().numpy() if hasattr(bearing_mask,'detach') else np.asarray(bearing_mask)
        if mask.shape!=(2,5) or mask.dtype!=np.bool_:
            raise ValueError('same-clock native bearing mask bool 2x5 required')
        provenance_ok=(isinstance(bearing_provenance,dict)
            and bearing_provenance.get('native_accepted_contacts') is True
            and type(bearing_provenance.get('physics_step')) is int
            and bearing_provenance['physics_step']==snapshot_identity['physics_step']
            and bearing_provenance.get('snapshot_identity') in
                (snapshot_identity,snapshot_identity['source_snapshot_id']))
        for d,r in enumerate(refs):
            if r['eligible'] and (not mask[1,d] or not provenance_ok):
                r.update(eligible=False,status='bearing_provenance_mismatch',
                         relief_blocked=r['mandatory_relief'])
        if not any(r['eligible'] for r in refs):
            return dict(result,reason='MANDATORY_PRESSURE_RELIEF_BLOCKED' if any(r['relief_blocked'] for r in refs)
                        else 'NO_ACTUAL_BEARING_PRESSURE_INPUT',elapsed_s=time.perf_counter()-started)
        model=self.planner.geometry.models[1];snapshot=snapshots[1];local_points={}
        for r in refs:
            if r['eligible']:
                pose=snapshot['poses'][model.patches[r['digit']]['body']]
                local_points[r['digit']]=pose[:3,:3].T@(r['actual_point_w_m']-pose[:3,3])
        baseline_action=np.zeros(24,np.float32)
        _,base_points,_,_=self._point_path(ctx,baseline_action,local_points,n)
        # Preserve the *actual material-point* direction of any upstream relief,
        # not a guessed sign of a DH116 joint. Never alter the legacy controller.
        for r,group in zip(refs,JOINT_GROUPS):
            if not r['eligible']:
                continue
            only=np.zeros(24,np.float32);indices=np.asarray(group)+6;only[indices]=upstream[indices]
            _,u_points,_,_=self._point_path(ctx,only,local_points,n)
            relative=u_points[-1][r['digit']]-base_points[-1][r['digit']]
            projected=float(relative@r['inward_w'])
            if projected < -1e-15:
                r['must_preserve_upstream_relief']=True
                r['mandatory_relief']=True
                if r['speed_m_s']>=0:
                    r['speed_m_s']=-min(.001,abs(projected)/(n*DT))
            r['target_point_w_m']=r['actual_point_w_m']+r['inward_w']*(r['speed_m_s']*n*DT)
        actual=self.planner.geometry.audit_state(snapshots,np.zeros(6),left_delta=np.zeros(6),
            bearing_mask=bearing_mask,bearing_provenance=bearing_provenance)
        if not actual['motion_safe'] or not actual['audit_complete']:
            for r in refs:
                if r['eligible']:
                    r['status']='actual_source_geometry_uncertified'
                    r['relief_blocked']=r['mandatory_relief']
            return dict(result,reason='ACTUAL_SOURCE_GEOMETRY_UNCERTIFIED',audit=actual,
                rejection_reasons=actual['rejection_reasons'],elapsed_s=time.perf_counter()-started)
        action=np.zeros(24,np.float32);any_accepted=False;last_audit=None
        for r,group in zip(refs,JOINT_GROUPS):
            if not r['eligible']:
                continue
            digit=r['digit'];indices=np.asarray(group)+6
            if r['speed_m_s']==0:
                r['status']='pressure_goal_hold';continue
            # Fresh local FK derivatives are only seeds, never certificates.
            # They are recomputed each call and the resulting point trajectory
            # is evaluated by complete nonlinear FK, then the full hull auditor.
            _,current_points,_,_=self._point_path(ctx,action,local_points,n)
            jac=[]
            for j in indices:
                minus=action.copy();plus=action.copy();minus[j]=-.05;plus[j]=.05
                _,pm,_,_=self._point_path(ctx,minus,local_points,n)
                _,pp,_,_=self._point_path(ctx,plus,local_points,n)
                jac.append((pp[-1][digit]-pm[-1][digit])/.1)
            J=np.asarray(jac).T
            desired=r['target_point_w_m']-current_points[-1][digit]
            if not np.isfinite(J).all() or np.linalg.norm(J)<1e-14:
                r['status']='fresh_fk_no_pressure_authority';r['relief_blocked']=r['mandatory_relief'];continue
            seed=action.copy();seed[indices]=np.clip(np.linalg.lstsq(J,desired,rcond=1e-8)[0],-1.,1.).astype(np.float32)
            specs=[]
            if r['must_preserve_upstream_relief']:
                old=action.copy();old[indices]=upstream[indices];specs.append(('original_upstream_relief',old))
            specs.append(('fresh_nonlinear_pressure',seed))
            scored=[]
            for name,candidate in specs:
                bounded=self._bounded(ctx,candidate,local_points,n)
                if bounded is None:
                    continue
                proposed,path,points,normals,costs=bounded
                if not np.any(proposed[indices]):
                    continue
                displacement=np.asarray([p[digit]-r['actual_point_w_m'] for p in points])
                normal=displacement@r['inward_w']
                # Preserve the current-measurement relief direction over the
                # ENTIRE predicted material-point path; not merely the endpoint.
                if r['mandatory_relief'] and np.any(normal>1e-15):
                    continue
                if r['speed_m_s']>0 and normal[-1]<=0:
                    continue
                if r['speed_m_s']<0 and normal[-1]>=0:
                    continue
                if np.max(np.linalg.norm(displacement,axis=1))>.001*n*DT+1e-15:
                    continue
                distance=float(np.linalg.norm(points[-1][digit]-r['target_point_w_m']))
                original=float(np.linalg.norm(current_points[-1][digit]-r['target_point_w_m']))
                if distance>=original:
                    continue
                scored.append((0 if name=='original_upstream_relief' else 1,distance,name,proposed,path,points,normals,costs,normal))
            scored.sort(key=lambda v:(v[0],v[1]))
            if not scored:
                r['status']='no_direction_safe_pressure_candidate';r['relief_blocked']=r['mandatory_relief'];continue
            _,_,name,proposed,path,points,normals,costs,normal=scored[0]
            audit=self.planner.audit_action(action24=proposed,substeps=n,**ctx)
            result['candidate_audits'].append(dict(digit=digit,name=name,audit=audit))
            if not audit['accepted']:
                r['status']='pressure_geometry_rejected_current_attempt_only'
                r['relief_blocked']=r['mandatory_relief'];continue
            action=proposed;last_audit=audit;any_accepted=True
            r.update(status='certified_geometric_pressure_proposal',
                predicted_normal_displacement_m=float(normal[-1]),
                predicted_material_point_path_w_m=np.asarray([p[digit] for p in points]),
                predicted_source_facet_normals_w=[v[digit] for v in normals],
                command_indices=indices.copy(),proposal_name=name,
                direction_scope='fresh_current_measured_force_only; future normals are geometric FK')
        # A later pad's action cannot silently invalidate an earlier pad's
        # pressure direction. Finger trees are independent but recheck the final
        # full nonlinear material paths rather than assuming that property.
        _,points,_,costs=self._point_path(ctx,action,local_points,n)
        blocked=any(r['relief_blocked'] for r in refs)
        for r in refs:
            if r['eligible'] and r['status']=='certified_geometric_pressure_proposal':
                projection=np.asarray([p[r['digit']]-r['actual_point_w_m'] for p in points])@r['inward_w']
                if r['mandatory_relief'] and np.any(projection>1e-15):
                    blocked=True;r['status']='final_combined_direction_rejected';r['relief_blocked']=True
        if blocked:
            # No partial accepted=True may hide a mandatory relief failure.
            action[:]=0.;any_accepted=False
            result['withdrawn_candidate_audit']=last_audit;last_audit=None
            _,_,_,costs=self._point_path(ctx,action,local_points,n)
            for r in refs:
                if r['status']=='certified_geometric_pressure_proposal':
                    r['status']='withdrawn_due_mandatory_pressure_relief_blocked'
        result.update(action24=action,audit=last_audit,accepted=bool(any_accepted and not blocked),
            per_substep_command_path_m=costs,
            reason='MANDATORY_PRESSURE_RELIEF_BLOCKED' if blocked else
                'FRESH_GEOMETRIC_PRESSURE_PROPOSAL' if any_accepted else 'HOLD_CURRENT_PRESSURE_ATTEMPT',
            elapsed_s=time.perf_counter()-started)
        return result
