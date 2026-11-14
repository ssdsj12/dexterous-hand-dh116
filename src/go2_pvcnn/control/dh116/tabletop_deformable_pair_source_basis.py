"""Recomputable contact-direction and material-grid diagnostics, not admission.

Normals point out of the obstacle; forces act on the rod. A pair point is the
node-sphere surface point x - R*n, not a measured continuous material anchor.
"""
import hashlib
from copy import deepcopy
from itertools import combinations, product
import json
import math
from pathlib import Path

import numpy as np

from .tabletop_deformable_contact_diagnostics import MIN_NORMAL_FORCE_N
from .tabletop_deformx_pair_capture import validate_pair_capture

VERSION="dh116-flexible-pair-source-basis-v1"
CONTACT_SOURCE_SHA256="8d90d9d18c853fbc3931a69435000726f49d36c887e62debe10225e8aa77d2bb"
PADS=("finger14_link","finger23_link","finger33_link","finger43_link","finger53_link")
NAMES={f"{side}.{pad}" for side in ("left","right") for pad in PADS}|{"table"}
DT=1/480


def verify_contact_source(path):
    """Read the exact compiled-tree header; never load or execute its extension."""
    path=Path(path).resolve(strict=True)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=CONTACT_SOURCE_SHA256:
        raise ValueError("contact source differs from this diagnostic convention")
    return dict(path=str(path),source_sha256=sha,
        outward_normal="obstacle_SDF_gradient_rotated_to_world",
        force="force_on_rod_N_times_n_plus_regularized_friction",
        point="pos_plus_dx_minus_radius_times_n",
        normal_source_lines=[94,108],force_source_lines=[27,45],
        readout_source_lines=[165,206],binary_loaded_or_native_qualified=False)


def _source(value):
    if not isinstance(value,str) or not value.strip():
        raise ValueError("explicit nonempty source label required")
    return value


def _radius(sample):
    if not isinstance(sample,dict) or set(sample)!={"value","source"}:
        raise ValueError("explicit radius sample, including unknown, required")
    value=sample["value"]
    if value is None:return None
    _source(sample["source"])
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError("finite positive radius required when known")
    return float(value)


def material_station_resolution(rest_lengths_m,*,source):
    """Exact discrete stations and a conditional nearest-node quantization bound.

    h_max/2 bounds distance from an arbitrary s in [0,L] to its nearest station.
    Capture does not implement this assignment of continuous physical contact,
    so this theorem supplies no error bound on its material anchor or slip.
    """
    rest=np.asarray(rest_lengths_m,dtype=float)
    if rest.ndim!=1 or len(rest)<1 or not np.isfinite(rest).all() or np.any(rest<=0):
        raise ValueError("finite positive one-dimensional rest segments required")
    stations=np.r_[0.,np.cumsum(rest)]
    if not np.isfinite(stations).all():raise ValueError("finite accumulated station range required")
    return dict(source=_source(source),node_count=len(stations),rest_length_m=float(stations[-1]),
        stations_m=stations.tolist(),min_station_gap_m=float(rest.min()),
        max_station_gap_m=float(rest.max()),nearest_station_quantization_bound_m=float(rest.max()/2),
        bound_assumption="arbitrary_s_in_[0,L]_assigned_to_its_nearest_rest_station",
        conditional_bound_applies_to_captured_contact=False,
        physical_material_anchor_error_bound_m=None,tangential_slip_error_bound_m=None,
        material_anchor_slip=None,sub_node_material_anchor_measured=False)


def compare_contacts(a,b,*,radius_m=None):
    """Signed scalar geometry for two validated contacts in the same substep."""
    if a["substep"]!=b["substep"] or a["mesh_index"]==b["mesh_index"]:
        raise ValueError("same-substep distinct-mesh contacts required")
    pa,pb,na,nb,fa,fb=[np.asarray(p[key],dtype=float) for p,key in (
        (a,"contact_point_world_m"),(b,"contact_point_world_m"),
        (a,"outward_normal_world"),(b,"outward_normal_world"),
        (a,"force_on_rod_world_n"),(b,"force_on_rod_world_n"))]
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (pa,pb,na,nb,fa,fb)):
        raise ValueError("finite world vectors required")
    if not all(math.isclose(math.hypot(*n),1.,abs_tol=2e-5,rel_tol=0) for n in (na,nb)):
        raise ValueError("unit obstacle outward normals required")
    chord=pb-pa
    separation=math.hypot(*chord)
    ma,mb=math.hypot(*fa),math.hypot(*fb)
    ndot=float(na@nb)
    force_cos=None if ma==0 or mb==0 else float((fa/ma)@(fb/mb))
    loads=[float(fa@na),float(fb@nb)]
    inward_n=inward_f=None
    if separation>0:
        axis=chord/separation
        inward_n=[float(na@axis),float(-nb@axis)]
        inward_f=[float(fa@axis),float(-fb@axis)]
    loaded=all(f>MIN_NORMAL_FORCE_N for f in loads)
    candidate=None
    if loaded and separation>0 and force_cos is not None:
        candidate=bool(ndot<0 and force_cos<0 and min(inward_n)>0 and min(inward_f)>0)
    center_residual=None
    if radius_m is not None and a["node_index"]==b["node_index"]:
        if not math.isfinite(radius_m) or radius_m<=0:raise ValueError("positive explicit radius required")
        center_residual=math.hypot(*(pa+radius_m*na-pb-radius_m*nb))
    result=dict(pair_ids=[[a["substep"],a["node_index"],a["mesh_index"]],
                         [b["substep"],b["node_index"],b["mesh_index"]]],
        contact_points_world_m=[pa.tolist(),pb.tolist()],obstacle_normals_world=[na.tolist(),nb.tolist()],
        forces_on_rod_world_n=[fa.tolist(),fb.tolist()],
        contact_point_separation_m=separation,normal_dot=ndot,force_cosine=force_cos,
        normal_loads_n=loads,inward_normal_cosines=inward_n,inward_force_projections_n=inward_f,
        both_above_numerical_load_floor=loaded,geometric_opposition_sign_candidate=candidate,
        same_node_reconstructed_center_residual_m=center_residual,
        numerical_direction_uncertainty_bound=None,effective_opposed_contact=None,material_anchor_slip=None)
    # Finite inputs can still overflow intermediate arithmetic: never serialize
    # nonfinite diagnostics or silently substitute a direction or zero.
    json.dumps(result,allow_nan=False)
    return result


def analyze_pair_capture(raw,*,mesh_names,rest_lengths_m,rest_source,radius_sample,
                         capture_source,max_comparisons=65536):
    """Reconcile original atomic raw pairs, then compare within each side/substep.

    Mesh labels and source labels are explicit caller provenance, not new SDK
    path attestation. All comparisons are retained within a declared CPU bound;
    exceeding it rejects the packet instead of truncating evidence.
    """
    source=_source(capture_source)
    if (isinstance(mesh_names,str) or not mesh_names or len(set(mesh_names))!=len(mesh_names)
            or not set(mesh_names)<=NAMES):
        raise ValueError("explicit distinct original hand-pad/table mesh names required")
    if type(max_comparisons) is not int or not 1<=max_comparisons<=1000000:
        raise ValueError("bounded comparison count must be 1..1000000")
    material=material_station_resolution(rest_lengths_m,source=rest_source)
    radius=_radius(radius_sample)
    result=dict(version=VERSION,scope="per_substep_signed_geometry_and_force_diagnostic_only",
        capture_source=source,source_labels_are_caller_provenance_not_native_attestation=True,
        mesh_names=list(mesh_names),radius_sample=dict(radius_sample),material_resolution=material,
        pair_count=None,table_pair_count=None,comparison_count=None,substeps=None,
        min_normal_force_n=MIN_NORMAL_FORCE_N,max_comparisons=max_comparisons,
        effective_opposed_contact=None,material_anchor_slip=None,approach_distance_m=None,
        native_qualified=False,teacher_admission=False,training_reward=None,
        physical_steps_executed=0,optimizer_updates=0,
        unavailable_sources=dict(
            effective_opposed_contact="signed_vectors_do_not_establish_physical_force_closure_or_grasp_quality",
            material_anchor_slip="node_sphere_contacts_are_not_continuous_fixed_material_anchors",
            approach_distance_m="penetration_only_pair_samples_do_not_measure_separated_pad_to_target_distance"))
    if raw is None:
        result["unavailable_reason"]="pair_capture_unavailable_aggregates_cannot_supply_pair_direction"
        return result
    if raw["mesh_count"]!=len(mesh_names):
        raise ValueError("mesh names differ from original captured slot count")
    if len(raw["node_forces_on_rod_world_n"])!=material["node_count"]:
        raise ValueError("rest-segment material identity differs from captured nodes")
    # JSON cannot retain the (0,3)/(0,12) shapes without explicit reconstruction.
    packet=dict(raw,indices=np.asarray(raw["indices"],dtype=np.int64).reshape(-1,3),
                vectors=np.asarray(raw["vectors"],dtype=float).reshape(-1,12))
    # Reject integer coercion rather than silently rounding source node/slot IDs.
    original=np.asarray(raw["indices"])
    if original.size and (original.dtype.kind not in "iu" or not np.array_equal(original,packet["indices"])):
        raise ValueError("original integer substep/node/mesh identities required")
    clean=validate_pair_capture(packet,dt_s=DT,node_forces=raw["node_forces_on_rod_world_n"],
                                mesh_wrenches=raw["mesh_wrenches_on_mesh_world"])
    groups={(sub,side):[] for sub in range(raw["substeps"]) for side in ("left","right")}
    table_count=0
    for p in clean["pairs"]:
        name=mesh_names[p["mesh_index"]]
        if name=="table":table_count+=1;continue
        groups[p["substep"],name.split(".")[0]].append(p)
    # Count with arithmetic first: never enumerate a quadratic rejected case.
    count=0
    for group in groups.values():
        by_mesh={}
        for pair in group:by_mesh[pair["mesh_index"]]=by_mesh.get(pair["mesh_index"],0)+1
        n=len(group)
        count+=n*(n-1)//2-sum(k*(k-1)//2 for k in by_mesh.values())
    if count>max_comparisons:raise ValueError("pair comparisons exceed complete-evidence CPU bound")
    stations=material["stations_m"]
    subrows=[]
    for sub in range(raw["substeps"]):
        comparisons=[]
        for side in ("left","right"):
            mesh_groups={}
            for pair in groups[sub,side]:mesh_groups.setdefault(pair["mesh_index"],[]).append(pair)
            for left,right in combinations(mesh_groups.values(),2):
                for a,b in product(left,right):
                    item=compare_contacts(a,b,radius_m=radius)
                    item.update(side=side,pad_names=[mesh_names[p["mesh_index"]] for p in (a,b)],
                        material_node_stations_m=[stations[p["node_index"]] for p in (a,b)],
                        material_station_gap_m=abs(stations[a["node_index"]]-stations[b["node_index"]]))
                    comparisons.append(item)
        subrows.append(dict(substep=sub,comparisons=comparisons))
    result.update(pair_count=len(clean["pairs"]),table_pair_count=table_count,
                  comparison_count=count,substeps=subrows,substep_dt_s=clean["substep_dt_s"],
                  pair_aggregate_reconciliation=True)
    return result


def analyze_rod_pair_basis(row,*,grasp,radius_sample,max_comparisons=65536):
    """Consume the current dual-hand runtime row without changing its contact gate."""
    from .tabletop_deformable_contact_diagnostics import summarize_contact_pairs
    identity_keys=("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step",
                   "rod_tick","timestamp_s","solver_time_s","clock_valid")
    identity={k:row[k] for k in identity_keys}
    if (identity["clock_valid"] is not True
            or any(not isinstance(identity[k],str) or not identity[k] for k in ("run_uuid","episode_uid","rod_uid"))
            or any(type(identity[k]) is not int or identity[k]<0
                   for k in ("env_id","reset_id","physics_step","rod_tick"))
            or any(type(identity[k]) not in (int,float) or not math.isfinite(identity[k]) or identity[k]<0
                   for k in ("timestamp_s","solver_time_s"))
            or not math.isclose(identity["timestamp_s"],identity["physics_step"]*DT,rel_tol=0,abs_tol=1e-9)):
        raise ValueError("complete actual runtime boundary clock identity required")
    counts=summarize_contact_pairs(row,grasp=grasp)
    packet=row.get("rod_mesh_contact_pairs")
    raw=None
    if packet is not None:
        pairs=packet["pairs"]
        raw=dict(substeps=counts["substeps"],mesh_count=len(row["mesh_names"]),
            capacity=max(1,len(pairs)),attempted=len(pairs),overflow=False,
            indices=[[p["substep"],p["node_index"],p["mesh_index"]] for p in pairs],
            vectors=[[*p["contact_point_world_m"],*p["outward_normal_world"],
                      *p["force_on_rod_world_n"],*p["torque_on_mesh_about_origin_world_nm"]] for p in pairs],
            node_forces_on_rod_world_n=np.asarray(row["distributed_contact"])[0],
            mesh_wrenches_on_mesh_world=[row["named_wrenches"][n] for n in row["mesh_names"]])
    result=analyze_pair_capture(raw,mesh_names=row["mesh_names"],rest_lengths_m=row["rest_lengths_m"],
        rest_source="same_current_runtime_rest_lengths_m",radius_sample=radius_sample,
        capture_source="same_current_runtime_pair_packet_revalidated_by_original_diagnostics",
        max_comparisons=max_comparisons)
    result.update(boundary_identity=identity,original_pair_diagnostics=counts,
        reconstructed_raw_capacity_is_exact_stored_row_count_not_original_buffer_capacity=True)
    return result


class ContactTimeline:
    """Bounded same-episode observation of actual pair packets.

    Candidate signs are geometric diagnostics. No R0 label, physical stability,
    force-closure qualification or fixed-material slip is manufactured here.
    """

    def __init__(self, *, grasp, radius_sample, max_physics_rows=240):
        from .deformx_dh116_colliders import collider_plan
        if type(max_physics_rows) is not int or not 1 <= max_physics_rows <= 240:
            raise ValueError("bounded contact timeline requires 1..240 physics rows")
        _radius(radius_sample)
        self.radius_sample = dict(radius_sample)
        self.grasp = grasp
        self.required = {side: tuple(spec["bodies"]) for side, spec in
                         zip(("left", "right"), collider_plan(grasp)["specs"])}
        self.max_physics_rows = max_physics_rows
        self.rows = []
        self._last = None
        self._substeps = None

    def record(self, row, *, action_identity):
        from .tabletop_deformable_solver_clock import solver_clock_matches
        if self._last is not None and self._last["physics_rows"] >= self.max_physics_rows:
            raise ValueError("contact timeline physics-row budget exhausted")
        # Original validator reopens all raw vectors and reconciles independent
        # aggregate buffers BEFORE any timeline state is committed.
        basis = analyze_rod_pair_basis(row, grasp=self.grasp, radius_sample=self.radius_sample)
        identity = basis["boundary_identity"]
        if (not isinstance(action_identity, dict)
                or any(action_identity.get(k) != v for k, v in identity.items())
                or any(type(action_identity.get(k)) is not int or action_identity[k] < 0
                       for k in ("policy_step", "substep"))
                or action_identity["substep"] >= 8):
            raise ValueError("actual action and physical identity must agree")
        if self._last is not None:
            last = self._last
            for key in ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid"):
                if identity[key] != last[key]:
                    raise ValueError("new episode requires independent timeline")
            if (identity["physics_step"] != last["physics_step"] + 1
                    or not solver_clock_matches(last["solver_time_s"], identity["solver_time_s"], steps=1)
                    or (action_identity["policy_step"], action_identity["substep"]) !=
                       (last["policy_step"] + (last["substep"] == 7), (last["substep"] + 1) % 8)):
                raise ValueError("non-contiguous actual action/physics/solver clock")
        substeps = (basis["original_pair_diagnostics"]["substeps"]
                    if basis["substeps"] is not None else self._substeps)
        # An absent first packet cannot establish solver sampling resolution.
        if substeps is None:
            raise ValueError("first timeline packet must establish solver substeps")
        if (self._substeps is not None and substeps != self._substeps
                or identity["rod_tick"] < substeps
                or (self._last is not None and identity["rod_tick"] != self._last["rod_tick"] + substeps)):
            raise ValueError("solver tick interval changed")
        slots = row["mesh_names"]
        packet = row.get("rod_mesh_contact_pairs")
        groups = [[] for _ in range(substeps)]
        if packet is not None:
            for p in packet["pairs"]:
                groups[p["substep"]].append(p)
        output = []
        for sub in range(substeps):
            pairs = groups[sub]
            comparisons = None if basis["substeps"] is None else basis["substeps"][sub]["comparisons"]
            pads = None
            if packet is not None:
                pads = {}
                for name in slots:
                    selected = [p for p in pairs if p["mesh_name"] == name]
                    loaded = [p for p in selected if
                              np.dot(p["force_on_rod_world_n"], p["outward_normal_world"]) > MIN_NORMAL_FORCE_N]
                    pads[name] = dict(
                        loaded=bool(loaded), raw_pairs=len(selected),
                        loaded_node_ids=sorted(p["node_index"] for p in loaded),
                        normal_load_n=float(sum(np.dot(p["force_on_rod_world_n"], p["outward_normal_world"])
                                                for p in loaded)),
                        force_on_rod_world_n=np.sum([p["force_on_rod_world_n"] for p in selected],
                                                   axis=0).tolist() if selected else [0., 0., 0.])
            hands = {}
            for side in ("left", "right"):
                required = self.required[side]
                all_loaded = None if pads is None else all(pads[side+"."+p]["loaded"] for p in required)
                sign_candidate = None
                if all_loaded is False:
                    sign_candidate = False
                elif all_loaded is True:
                    # A thumb/opposing-finger graph in this SAME substep.
                    # Different node stations remain in comparisons, never
                    # silently interpreted as one material cross-section.
                    results = []
                    for digit in required[1:]:
                        pair_names = {side+"."+required[0], side+"."+digit}
                        flags = [c["geometric_opposition_sign_candidate"] for c in comparisons
                                 if c["side"] == side and set(c["pad_names"]) == pair_names]
                        results.append(True if any(f is True for f in flags) else
                                       None if not flags or any(f is None for f in flags) else False)
                    sign_candidate = False if any(f is False for f in results) else (
                        True if all(f is True for f in results) else None)
                hands[side] = dict(required_pads=list(required), all_required_loaded=all_loaded,
                                   opposition_sign_candidate=sign_candidate)
            event_identity = dict(identity, policy_step=action_identity["policy_step"],
                substep=action_identity["substep"], solver_substep=sub,
                solver_tick=identity["rod_tick"]-substeps+sub+1,
                # This is an explicit ideal clock derived from the source SDK
                # interval; raw float solver_time_s remains a boundary getter.
                reconstructed_sdk_substep_end_s=(identity["physics_step"]-1+(sub+1)/substeps)*DT)
            output.append(dict(identity=event_identity, substep_dt_s=DT/substeps,
                source_scope="validated_pair_signed_direction_diagnostic_only",
                pads=pads, hands=hands, comparisons=comparisons,
                table_loaded=None if pads is None else pads["table"]["loaded"],
                effective_opposed_contact=None, stable=None, material_anchor_slip=None))
        # Commit a complete, successfully validated interval atomically.
        self.rows.extend(output)
        self._substeps = substeps
        self._last = dict(action_identity, physics_rows=1 if self._last is None else self._last["physics_rows"]+1)
        return deepcopy(output)

    def summary(self):
        def stats(values):
            first = last = None
            count = unknown = longest = current = intervals = 0
            for value, event in zip(values, self.rows):
                if value is True:
                    count += 1
                    if current == 0:
                        intervals += 1
                    current += 1
                    longest = max(longest, current)
                    first = first or event["identity"]
                    last = event["identity"]
                else:
                    current = 0
                    unknown += value is None
            return dict(true_substeps=count, unknown_substeps=unknown,
                        false_substeps=len(values)-count-unknown,
                        observed_true_duration_s=count*DT/self._substeps if self._substeps else 0.,
                        longest_interval_s=longest*DT/self._substeps if self._substeps else 0.,
                        true_interval_count=intervals, interval_count_includes_open_tail=True,
                        first_true=deepcopy(first), last_true=deepcopy(last))
        metrics = {}
        for side in ("left", "right"):
            for key in ("all_required_loaded", "opposition_sign_candidate"):
                metrics[side+"."+key] = stats([r["hands"][side][key] for r in self.rows])
            for pad in PADS:
                metrics[side+"."+pad] = stats([None if r["pads"] is None else
                                               r["pads"][side+"."+pad]["loaded"] for r in self.rows])
        metrics["table_loaded"] = stats([r["table_loaded"] for r in self.rows])
        both = []
        for row in self.rows:
            vals = [row["hands"][side]["all_required_loaded"] for side in ("left", "right")]
            both.append(False if any(v is False for v in vals) else
                        True if all(v is True for v in vals) else None)
        metrics["bilateral_required_loaded"] = stats(both)
        return dict(version="dh116-actual-pair-contact-timeline-v1", grasp=self.grasp,
            physics_rows=0 if self._last is None else self._last["physics_rows"],
            solver_substep_rows=len(self.rows), metrics=metrics,
            comparison_count=sum(len(r["comparisons"] or []) for r in self.rows),
            effective_opposed_contact=None, stable=None, material_anchor_slip=None,
            teacher_admission=False, native_qualified=False, new_physics_steps=0,
            ppo_updates=0, training_reward=None, actual_hold_success=None)
