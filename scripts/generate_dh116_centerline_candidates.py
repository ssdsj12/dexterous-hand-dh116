#!/usr/bin/env python3
"""Generate CPU-only DH116 geometry/static proposals from explicit frozen inputs.

Example: PYTHONPATH=src:. python scripts/generate_dh116_centerline_candidates.py
  --scene scene.json --seed seed.json --output new-report.json --positions 3
This module does not import SimulationApp, start Kit or write any scene/reset.
Frozen input arithmetic/local hashes are not native physical certification;
the independent native frozen-scene/reset/candidate admission gate is pending.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--positions', type=int, default=3)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--wall-seconds', type=float, default=120.)
    parser.add_argument('--azimuths-rad', type=float, nargs='+', default=[0., 1.5707963267948966, 3.141592653589793])
    parser.add_argument('--styles', choices=('pinch', 'tripod'), nargs='+', default=['pinch', 'tripod'])
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 2 <= args.positions <= 12 or not 0 <= args.steps <= 200 or not 0 < args.wall_seconds <= 3600:
        raise ValueError('positions=2..12, steps=0..200, wall-seconds=(0,3600] required')
    # Heavy CPU/USD imports occur only after all output/argument guards.
    import numpy as np
    import torch
    from go2_pvcnn.control.dh116.tabletop_surface_kinematics import SurfaceHandModel
    from go2_pvcnn.control.dh116.tabletop_centerline_grasp import (
        validate_frozen_inputs, sample_centerline_frames, build_source_pad_atlas, generate_candidates)
    scene = json.loads(args.scene.read_text(encoding='utf-8-sig'))
    seed = json.loads(args.seed.read_text(encoding='utf-8-sig'))
    validate_frozen_inputs(scene, seed)
    root = args.root.resolve()
    sources = []
    for side in ('left', 'right'):
        source = (root/scene['source_usd'][side]).resolve()
        try:
            source.relative_to(root/'assets')
        except ValueError as exc:
            raise ValueError('source USD must be a repository assets resource') from exc
        if not source.is_file() or source.suffix.lower() not in ('.usd', '.usda', '.usdc'):
            raise ValueError('existing repository source USD required')
        sources.append(source)
    protected = [Path(__file__).resolve(), args.scene.resolve(), args.seed.resolve(), *sources]
    protected += sorted((root/'src/go2_pvcnn/control/dh116').glob('*.py'))
    before = {str(p): sha256(p) for p in protected}
    torch.set_num_threads(2)
    models = [SurfaceHandModel.from_usd(source, side, device='cpu') for side, source in zip(('left', 'right'), sources)]
    if any(len(m.shapes) != 17 for m in models):
        raise ValueError('all seventeen source collision hulls per hand required')
    atlas = build_source_pad_atlas(models)
    length = sample_centerline_frames(scene['centerline_w_m'], [0])[0]['centerline_length_m']
    config = dict(arc_positions_m=np.linspace(.2*length, .8*length, args.positions).tolist(),
        azimuths_rad=args.azimuths_rad, styles=args.styles, steps=args.steps, wall_seconds=args.wall_seconds,
        wrist_translation_bound_m=.02, wrist_rotation_bound_rad=.3)
    report = generate_candidates(models, atlas, scene, seed, scene['obstacles'], config)
    after = {str(p): sha256(p) for p in protected}
    report.update(source_hashes=dict(zip(('left', 'right'), (before[str(p)] for p in sources))),
        frozen_scene_sha256=before[str(args.scene.resolve())], frozen_seed_sha256=before[str(args.seed.resolve())],
        script_sha256=before[str(Path(__file__).resolve())], protected_source_hashes_before=before,
        protected_source_hashes_after=after, protected_source_hashes_unchanged=before == after,
        pad_atlas=atlas, execution=dict(device='cpu', torch_threads=2, started_isaac=False),
        native_contact_verified=False, native_experiment_eligible=False, native_candidate_gate_implemented=False)
    if before != after:
        report['rejected'].extend(dict(row, accepted=False, geometry_feasible=False,
            native_experiment_eligible=False, static_proposal_ready_for_native_verification=False,
            rejection_reasons=row['rejection_reasons']+['SOURCE_IDENTITY_CHANGED']) for row in report['accepted'])
        report.update(accepted=[], candidates=[], accepted_count=0, geometry_feasible_count=0,
            static_proposal_ready_for_native_verification=False, status='source_identity_changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents both accidental reuse and a check/write race.
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(output=str(args.output), status=report['status'], accepted=report['accepted_count'],
                          rejected=len(report['rejected']), native_contact_verified=False)))
    return report


if __name__ == '__main__':
    main()
