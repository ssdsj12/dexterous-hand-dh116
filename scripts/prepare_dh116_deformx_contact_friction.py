"""Prepare a separate, source-locked raw friction-displacement observation.

Generated DeformX/YarnBall delta: GPL-3.0-only. No existing binary, source,
kernel equation, controller, material, clock or checkpoint is replaced.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
from scripts.prepare_dh116_deformx_mesh_observations import build_commands

ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = source.BUILD
OLD_SOURCE = OLD_ROOT / "source"
OLD_BACKEND = source.BACKEND
VERSION = "dh116-deformx-contact-friction-observation-v1"
RECORD_PATCHES = (
    ("third_party/yarnball/KittenEngine/YarnBall/YarnBall.h",
     "float gap, normalForce;",
     "float gap, normalForce;\n\t\t\tvec3 frictionDisplacement; // raw surfaceReadout dxRel, world metres"),
    ("third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh",
     "P.pos, P.q, gap, nativeNormalForce};",
     "P.pos, P.q, gap, nativeNormalForce, dxRel};"),
)


def _payload(value):
    return value if isinstance(value, bytes) else value.encode()


def _replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("exact single source anchor required")
    return text.replace(before, after)


def locked_sources(project):
    """Verify original complete source and binary before copying any bytes."""
    if Path(project).resolve() != source.ROOT.resolve():
        raise ValueError("the current source-locked project is required")
    source.source_basis()
    receipt = json.loads(source.PREPARATION.read_text())
    inventory = receipt["prepared_source_sha256"]
    result = {}
    for name, digest in inventory.items():
        path = OLD_SOURCE / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(OLD_SOURCE):
            raise ValueError("regular locked source file required")
        payload = source._read(path, digest)
        result[name] = payload if path.suffix == ".png" else payload.decode()
    return result


def planned_sources(project):
    """Only append raw diagnostic storage and a new independently named getter."""
    result = locked_sources(project)
    for name, before, after in RECORD_PATCHES:
        result[name] = _replace_once(result[name], before, after)
    name = "csrc/bindings.cpp"
    original = result[name]
    start = original.index('        .def("contact_pair_samples_v2",')
    stop = original.index('        .def("mesh_state_snapshot",', start)
    getter = original[start:stop]
    replacements = (
        ('contact_pair_samples_v2', 'contact_friction_samples_v1'),
        ('dh116.sdf_pair_samples.v2', 'dh116.sdf_pair_friction_samples.v1'),
        ('py::array_t<float> quaternions({size, py::ssize_t(4)});',
         'py::array_t<float> quaternions({size, py::ssize_t(4)});\n'
         '                 py::array_t<float> displacement({size, py::ssize_t(3)});'),
        ('auto q = quaternions.mutable_unchecked<2>();',
         'auto q = quaternions.mutable_unchecked<2>();\n'
         '                 auto d = displacement.mutable_unchecked<2>();'),
        ('p(i, j) = r.meshPosition[j];',
         'p(i, j) = r.meshPosition[j];\n'
         '                         d(i, j) = r.frictionDisplacement[j];'),
        ('out["surface_gap_m"] = gaps; out["normal_force_n"] = normal_forces;',
         'out["surface_gap_m"] = gaps; out["normal_force_n"] = normal_forces;\n'
         '                 out["solver_friction_displacement_world_m"] = displacement;\n'
         '                 out["friction_displacement_semantics"] = "raw_surfaceReadout_dxRel_before_tangent_projection";\n'
         '                 out["slip_uncertainty_m"] = py::none();\n'
         '                 out["slip_qualified"] = false;'),
        ('Owned native pose/gap/normal-force samples; substep indices have no measured timestamp.',
         'Owned raw solver friction displacement; not calibrated material or tangential slip.'),
    )
    for before, after in replacements:
        getter = _replace_once(getter, before, after)
    result[name] = original[:start] + getter + original[start:]
    return result


def prepare(project, destination):
    """Copy into a new tree and emit explicit build commands, without running them."""
    destination = Path(destination).resolve()
    project = Path(project).resolve()
    if (destination.exists() or destination.is_relative_to(OLD_ROOT)
            or OLD_ROOT.is_relative_to(destination) or destination.is_relative_to(project)):
        raise ValueError("a new empty independent destination outside the project is required")
    original = locked_sources(project)
    prepared = planned_sources(project)
    changed = {n: hashlib.sha256(_payload(v)).hexdigest()
               for n, v in prepared.items() if v != original[n]}
    expected = {p[0] for p in RECORD_PATCHES} | {"csrc/bindings.cpp"}
    if set(changed) != expected:
        raise ValueError("only the reviewed three observation files may change")
    for name, value in prepared.items():
        path = destination / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_payload(value))
        if name.startswith("ropesim/") and name.endswith(".py"):
            facade = destination / "build/python" / name
            facade.parent.mkdir(parents=True, exist_ok=True)
            facade.write_bytes(_payload(value))
    build = destination / "build"
    build.mkdir(exist_ok=True)
    delta = "".join("".join(difflib.unified_diff(
        original[name].splitlines(keepends=True), prepared[name].splitlines(keepends=True),
        fromfile="a/"+name, tofile="b/"+name)) for name in sorted(changed))
    (build/"observation-only.patch").write_text(delta)
    result = dict(version=VERSION, upstream_commit="96b8514c7c55c724f1abba1fbf6ed6642b7db72b",
        original_source_tree_sha256=source.TREE_SHA,
        original_backend_path=str(OLD_BACKEND), original_backend_sha256=source.BACKEND_SHA,
        prepared_source_sha256={n: hashlib.sha256(_payload(v)).hexdigest() for n, v in prepared.items()},
        changed_sources=changed, delta_sha256=hashlib.sha256(delta.encode()).hexdigest(),
        commands=build_commands(destination), compiled=False, solver_executed=False,
        isaac_executed=False, slip_qualified=False, native_qualification=False, teacher_admission=False)
    (build/"preparation.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    locked_sources(project)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(ROOT, args.destination), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
