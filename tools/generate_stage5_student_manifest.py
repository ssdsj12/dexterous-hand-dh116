"""Author an unseen Student validation split; this never runs native physics.

The existing authored states are templates, not native reset certificates.
Only the rigid wire's XY position and world-Z orientation are perturbed. All
hand state, velocities, placement labels and camera/physics identities remain
unchanged. Production always uses at least 100 independent states per style.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

from go2_pvcnn.control.dh116.tabletop_evaluation_trials import (
    GRASP_STYLES, state_sha256, validate_manifest,
)


XY_BOUND_M = .00025
WORLD_Z_YAW_BOUND_RAD = .002
MAX_RESAMPLE_ATTEMPTS = 128
PROVENANCE_VERSION = 'dh116-stage5-student-validation-manifest-v1'


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def _check_sha256(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('expected_source_sha256 must be a lowercase SHA256')


def _world_yaw_wxyz(q, angle):
    """Return normalized q_world_z * q, both explicitly wxyz.

    Left multiplication rotates about the world/reset-frame Z axis. It is
    deliberately not q * q_world_z, which rotates about the wire's local Z.
    Environment origins are translation-only placement labels in this schema.
    """
    a, b, c, d = math.cos(angle / 2.), 0., 0., math.sin(angle / 2.)
    w, x, y, z = q
    result = [a*w-b*x-c*y-d*z, a*x+b*w+c*z-d*y,
              a*y-b*z+c*w+d*x, a*z+b*y-c*x+d*w]
    norm = math.hypot(*result)
    if not math.isfinite(norm) or norm == 0.:
        raise ValueError('wire quaternion cannot be normalized')
    return [v / norm for v in result]


def _build_manifest(source_bytes, *, expected_source_sha256, seed,
                    _test_min_trials_per_style=None):
    """Pure authored-state generator; the private reduced mode is test-only.

    The public writer/CLI never exposes the reduced minimum. Production source
    manifests may be tagged test or validation, but require >=100/style in both
    cases. A reduced fixture must be tagged test and remains test. Neither
    source split nor sufficient authored case count provides native evidence.
    """
    _check_sha256(expected_source_sha256)
    if _sha256(source_bytes) != expected_source_sha256:
        raise ValueError('source manifest SHA256 does not match required source hash')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    source = json.loads(source_bytes)
    fixture = _test_min_trials_per_style is not None
    minimum = _test_min_trials_per_style if fixture else 100
    validate_manifest(source, min_trials_per_style=minimum)
    if fixture and source['metadata']['split'] != 'test':
        raise ValueError('reduced internal fixtures must use split=test')

    # Preserve supplied fields/values exactly instead of writing the validator's
    # normalized copy, which may reorder joints or round unrelated numbers.
    result = deepcopy(source)
    generator_sha = _sha256(Path(__file__).read_bytes())
    result['metadata'].update(seed=seed, split='test' if fixture else 'validation',
        validation_status='authored_unverified', generator_sha256=generator_sha,
        source_sha256=expected_source_sha256, template_sha256=expected_source_sha256)
    source_hashes = {row['state_sha256'] for row in source['trials']}
    output_hashes, perturbations = set(), []
    rng = random.Random(seed)
    for i, (old, new) in enumerate(zip(source['trials'], result['trials'])):
        original_pose = old['physical_state']['wire_pose']
        for attempt in range(1, MAX_RESAMPLE_ATTEMPTS + 1):
            dx = rng.uniform(-XY_BOUND_M, XY_BOUND_M)
            dy = rng.uniform(-XY_BOUND_M, XY_BOUND_M)
            yaw = rng.uniform(-WORLD_Z_YAW_BOUND_RAD, WORLD_Z_YAW_BOUND_RAD)
            new['physical_state']['wire_pose'] = [original_pose[0] + dx,
                original_pose[1] + dy, original_pose[2]] + _world_yaw_wxyz(original_pose[3:], yaw)
            digest = state_sha256(new['physical_state'])
            if digest not in source_hashes and digest not in output_hashes:
                break
        else:
            raise ValueError(f'bounded resampling exhausted for source trial {old["trial_id"]!r}')
        new['trial_id'] = f'stage5-validation-seed{seed}:{i:04d}:{old["grasp_style"]}'
        new['state_sha256'] = digest
        output_hashes.add(digest)
        perturbations.append(dict(source_trial_id=old['trial_id'],
            output_trial_id=new['trial_id'], source_state_sha256=old['state_sha256'],
            output_state_sha256=digest, dx_m=dx, dy_m=dy, world_z_yaw_rad=yaw,
            attempts=attempt))
    validate_manifest(result, min_trials_per_style=minimum)
    if source_hashes & output_hashes or len(output_hashes) != len(result['trials']):
        raise ValueError('output physical identities must be unique and source-disjoint')
    counts = {style: sum(row['grasp_style'] == style for row in result['trials'])
              for style in GRASP_STYLES}
    provenance = dict(version=PROVENANCE_VERSION, source_sha256=expected_source_sha256,
        template_sha256=expected_source_sha256, generator_sha256=generator_sha,
        physics_sha256=source['metadata']['physics_sha256'],
        camera_sha256=source['metadata']['camera_sha256'], seed=seed,
        split=result['metadata']['split'], validation_status='authored_unverified',
        scope='same_reset_family_wire_pose_microperturbation', generalization_claims=[],
        limitations=['No native reset/readback or contact validation has been run.',
                     'Not a flexible-shape, length, mass, material or friction generalization test.',
                     'Independent authored float32 write identities do not certify settled physics.'],
        native_runtime_verified=False, counts_as_native_validation=False,
        native_trials=0, model_derivations=0, training_updates=0,
        trials_by_style=counts, source_disjoint=True, output_states_unique=True,
        identity_contract='existing state_sha256 canonical native float32 write state',
        distribution=dict(kind='independent_symmetric_uniform', expected_mean=0.,
            xy_bound_m=XY_BOUND_M, world_z_yaw_bound_rad=WORLD_Z_YAW_BOUND_RAD,
            wire_z_unchanged=True, quaternion='normalized wxyz; q_world_z * q_source',
            max_attempts_per_trial=MAX_RESAMPLE_ATTEMPTS), perturbations=perturbations)
    return result, provenance


def generate_manifest(input_path, output_path, *, expected_source_sha256, seed=116507):
    """Exclusively write a production manifest and its adjacent provenance JSON.

    The source is read-only. On a failed write, only newly created files with
    the same file identity are removed; existing output/sidecar files survive.
    """
    source = Path(input_path).resolve(strict=True)
    output = Path(output_path).absolute()
    sidecar = Path(str(output) + '.provenance.json')
    if source in (output.resolve(), sidecar.resolve()):
        raise ValueError('output and provenance must not alias the source manifest')
    for path in (output, sidecar):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'refusing to overwrite {path}')
    source_bytes = source.read_bytes()
    manifest, provenance = _build_manifest(source_bytes,
        expected_source_sha256=expected_source_sha256, seed=seed)
    output_bytes = _json_bytes(manifest)
    provenance.update(source_path=str(source), output_path=str(output),
                      output_sha256=_sha256(output_bytes))
    if source.read_bytes() != source_bytes:
        raise ValueError('source manifest changed during generation')
    owned = []
    try:
        for path, data in ((output, output_bytes), (sidecar, _json_bytes(provenance))):
            with path.open('xb') as stream:
                identity = os.fstat(stream.fileno())
                owned.append((path, identity.st_dev, identity.st_ino))
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        if source.read_bytes() != source_bytes:
            raise ValueError('source manifest changed while writing outputs')
    except BaseException:
        for path, device, inode in reversed(owned):
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) == (device, inode):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    return dict(manifest_path=str(output), provenance_path=str(sidecar),
                output_sha256=_sha256(output_bytes), trials_by_style=provenance['trials_by_style'],
                validation_status='authored_unverified', native_trials=0)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--input', required=True,
        help='Existing authored test/validation manifest with >=100/style; read-only')
    result.add_argument('--output', required=True, help='New exclusive JSON; sidecar is OUTPUT.provenance.json')
    result.add_argument('--source-sha256', required=True, help='Required SHA256 of exact input file bytes')
    result.add_argument('--seed', type=int, default=116507)
    return result


def main(argv=None):
    options = parser().parse_args(argv)
    print(json.dumps(generate_manifest(options.input, options.output,
        expected_source_sha256=options.source_sha256, seed=options.seed), sort_keys=True))


if __name__ == '__main__':
    main()
