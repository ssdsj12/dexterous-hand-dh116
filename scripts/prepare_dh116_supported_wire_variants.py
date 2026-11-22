"""Prepare independent variable-diameter and length geometry on CPU."""
import argparse
import json
from pathlib import Path

from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import build_variant, group_wire_specs


ROOT = Path(__file__).resolve().parents[1]


def prepare(scene_path, scene_sha256, catalog_path, catalog_sha256, stations):
    scene = json.loads(checked_file(scene_path, scene_sha256))
    catalog = json.loads(checked_file(catalog_path, catalog_sha256))
    if (catalog["version"] != "dh116-supported-wire-generalization-v1"
            or catalog["material_scaling"] != "same_homogeneous_circular_candidate_material"):
        raise ValueError("known opt-in circular wire candidate catalog required")
    if scene["fixed_rod_nodes"] != [] or scene["physics_qualified"] is not False:
        raise ValueError("free supported scenario geometry required")
    source = catalog["reference_rod"]
    rod = json.loads(checked_file(ROOT / source["path"], source["sha256"]))["rod"]
    groups = group_wire_specs(catalog["variants"], rod)
    output = {}
    for split, candidates in groups.items():
        output[split] = {}
        for row in candidates:
            result = build_variant(scene, rod, radius_m=row["rod"]["radius_m"], length_m=row["length_m"],
                                   grasp_material_ids=stations)
            if result["spec_sha256"] != row["spec_sha256"]:
                raise ValueError("geometry and evaluation split must bind the same physical specification")
            output[split][row["id"]] = result
    return dict(version="dh116-wire-variant-scene-candidates-v1", catalog_sha256=catalog_sha256,
                source_scene_sha256=scene_sha256, grasp_material_ids=stations, groups=output,
                runtime_qualified=False, teacher_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--source-scene-sha256", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--left-material-id", type=int, required=True)
    parser.add_argument("--right-material-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source_scene, args.source_scene_sha256, args.catalog, args.catalog_sha256,
                     {"left": args.left_material_id, "right": args.right_material_id})
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: len(group) for key, group in result["groups"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
