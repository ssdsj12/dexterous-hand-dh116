"""Independent disk audit of native parameter packets; all previous gates retained."""
from pathlib import Path
import numpy as np
from scripts import validate_dh116_dual_pinch_director_interval as previous
from scripts import validate_dh116_dual_pinch_interval as original
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from go2_pvcnn.control.dh116 import tabletop_flexible_parameter_readback as reader
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned

AUDIT_NAME="dual-pinch-parameter-post-exit-audit.json"
SCOPE="exact_2_setup_plus_16_parameter_provenance_endpoints_only"
require=original.require

def inspect_parameters(path,*,complete=True):
    path=Path(path)
    prior=previous.inspect_directors(path,complete=complete)
    rows=original.rows(path)
    reset=next(decode_owned(x["payload"])["reset"]["rod_reset"] for x in rows[1:] if x["operation"]=="reset")
    nodes=reset["node_positions"][0]
    saved=original.read(path.parent/"director_reset.json")
    first=decode_owned(saved["payload"])
    samples=[(first,reset)] if first is not None else []
    for x in rows[1:]:
        if x["operation"]!="physical":continue
        exchange=decode_owned(x["payload"])["exchange"]
        if exchange["material_director"] is not None:
            samples.append((exchange["material_director"],exchange["rod"]))
    summaries=[]
    for director,identity in samples:
        # Use the original boundary clock, not a producer's invented identity.
        from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import _clock
        summaries.append(reader.validate_packet(director["parameter_readback"],
            identity=_clock(lambda:identity),constructor_nodes=nodes))
    require(len(samples)==prior["valid_packets"],"every valid director must have same-clock parameters")
    require(not complete or len(samples)==17,"complete seventeen parameter endpoint packets required")
    return dict(version=reader.VERSION,valid_packets=len(samples),
        measured_physics_rows=prior["measured_physics_rows"],
        complete_endpoint_sequence=prior["complete_endpoint_sequence"],
        configured_radius_m=summaries[0]["configured_radius_m"] if summaries else None,
        mass_kg=summaries[0]["mass_kg"] if summaries else None,
        source_basis=reader.source_basis(),native_qualification=False,teacher_admission=False,
        material_identified=False,configuration_is_native_measurement=False)

def audit_outcome(output,outer_exit,cleanup,*,profile_sha256=None):
    from scripts import prepare_dh116_dual_pinch_director_interval as director
    from scripts import prepare_dh116_dual_pinch_parameter_interval as operation
    output=Path(output)
    with bindings((director,"VERSION",operation.VERSION),(director,"inspect_interval",operation.inspect_interval)):
        result=previous.audit_outcome(output,outer_exit,cleanup,profile_sha256=profile_sha256)
    try:
        checked=inspect_parameters(output/"physical.jsonl",complete=result["lifecycle_qualified"])
        if result["lifecycle_qualified"]:
            require(result["journal"]["parameter_audit"]==checked,
                    "producer parameter summary differs from independent disk replay")
        result["parameter_audit"]=checked
        result["parameter_audit_error"]=None
    except previous.ERRORS as exc:
        result["lifecycle_qualified"]=False
        result["parameter_audit"]=None
        result["parameter_audit_error"]=type(exc).__name__+": "+str(exc)
        if not result.get("reason"):result["reason"]=result["parameter_audit_error"]
    result.update(qualification_scope=SCOPE,
        parameter_readback_qualified=result["lifecycle_qualified"],
        endpoint_director_readback_qualified=result["lifecycle_qualified"],
        material_identified=False,teacher_admission=False,grasp_qualified=False)
    result["available_artifact_sha256"]={p.name:original.sha(p) for p in sorted(output.glob("*"))
        if p.is_file() and not p.is_symlink() and p.name!=AUDIT_NAME}
    return result
