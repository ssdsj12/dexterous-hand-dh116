"""Merge qualified native surface replays into a balanced, auditable library.

Only explicit reset SHA fields may vary. The manifest records immutable source
file identities and supplied native proof references; no simulator is imported.
"""
from collections import Counter
import copy
import hashlib
import io
import json
from pathlib import Path
import pickle

import torch

from .tabletop_surface_imitation import DATASET_VERSION, validate_dataset, validate_reset_family


def _component(label, path):
    if not isinstance(label, str) or not label.strip():
        raise ValueError('component style label must be nonempty')
    try:
        path = Path(path).resolve()
        raw = path.read_bytes()
        data = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    except (OSError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError('component must reference a readable surface dataset file') from error
    validate_dataset(data)
    if 'reset_family' in data:
        raise ValueError('merge original single-reset components, not an already merged library')
    return label, path, data, hashlib.sha256(raw).hexdigest()


def _column(data, name):
    value = data[name]
    n = len(data['observations'])
    shape = (n, data['contract']['observation_size']) if name == 'next_observations' else (n,)
    if (not isinstance(value, torch.Tensor) or value.shape != shape
            or value.layout != torch.strided or value.is_complex()
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f'{name} must be finite and have shape {shape}')
    if name == 'dones':
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError('dones must contain only boolean values')
    elif not value.is_floating_point():
        raise ValueError(f'{name} must be floating point')
    return value


def merge_datasets(components, output):
    """Save a library and output.with_suffix('.manifest.json'); return its data.

    ``components`` is a list of (style_label, native_dataset_path) pairs. Row
    order and optional columns are preserved, episode IDs are globally unique,
    and every label receives equal total sampling weight across all its files.
    Original per-episode grasp styles remain separate from component labels.
    """
    if not isinstance(components, (list, tuple)) or len(components) < 2:
        raise ValueError('a library requires at least two native components')
    if any(not isinstance(item, (list, tuple)) or len(item) != 2 for item in components):
        raise ValueError('components must be (style_label, dataset_path) pairs')
    loaded = [_component(label, path) for label, path in components]
    output = Path(output).resolve()
    manifest_path = output.with_suffix('.manifest.json')
    if output == manifest_path or any(path in (output, manifest_path) for _, path, _, _ in loaded):
        raise ValueError('library output must not overwrite native components or its own manifest')
    family = [dict(label=label, contract=copy.deepcopy(data['contract']),
                   source_dataset_sha256=sha, source_dataset_path=str(path),
                   proof_path=data['qualification']['proof_path'],
                   qualification=copy.deepcopy(data['qualification']))
              for label, path, data, sha in loaded]
    base = loaded[0][2]['contract']
    validate_reset_family(family, base, base)
    columns = ['observations', 'actions']
    for name in ('next_observations', 'rewards', 'dones'):
        present = [name in data for _, _, data, _ in loaded]
        if any(present) and not all(present):
            raise ValueError(f'optional column {name} must be present in every component or none')
        if all(present):
            for _, _, data, _ in loaded:
                _column(data, name)
            columns.append(name)
    for name in columns:
        if len({data[name].dtype for _, _, data, _ in loaded}) != 1:
            raise ValueError(f'inconsistent {name} dtypes across native components')
    merged = dict(version=DATASET_VERSION, contract=copy.deepcopy(base), reset_family=family)
    for name in columns:
        merged[name] = torch.cat([data[name] for _, _, data, _ in loaded])
    totals = Counter()
    for label, _, data, _ in loaded:
        totals[label] += len(data['observations'])
    ids, weights = [], []
    styles, labels, episode_provenance = {}, {}, []
    next_episode = 0
    for label, path, data, sha in loaded:
        original_ids = data['episode_ids'].tolist()
        original_episodes = list(dict.fromkeys(original_ids))
        if len(original_episodes) != data['qualification']['successful_replays']:
            raise ValueError('component successful_replays must match its episode identities')
        remap = {old: next_episode + index for index, old in enumerate(original_episodes)}
        ids.append(torch.tensor([remap[old] for old in original_ids], dtype=torch.long))
        weights.append(torch.full((len(original_ids),), 1. / totals[label], dtype=torch.float64))
        native_styles = data.get('episode_grasp_styles', {})
        if not isinstance(native_styles, dict):
            raise ValueError('episode_grasp_styles must map episode IDs to original styles')
        for old, new in remap.items():
            style = native_styles.get(old, native_styles.get(str(old), label))
            if not isinstance(style, str) or not style.strip():
                raise ValueError('episode grasp styles must be nonempty strings')
            styles[new], labels[new] = style, label
            episode_provenance.append(dict(episode_id=new, original_episode_id=old,
                style_label=label, grasp_style=style, source_dataset_sha256=sha,
                source_dataset_path=str(path)))
        next_episode += len(remap)
    merged.update(episode_ids=torch.cat(ids), sample_weights=torch.cat(weights),
                  episode_grasp_styles=styles, episode_style_labels=labels)
    successes = sum(data['qualification']['successful_replays'] for _, _, data, _ in loaded)
    merged['qualification'] = dict(qualified=True, successful_replays=successes,
                                    proof_path=str(manifest_path))
    manifest = dict(version='dh116-surface-demo-library-manifest-v1', qualified=True,
        successful_replays=successes, dataset_path=str(output), reset_family=family,
        components=[dict(label=label, path=str(path), sha256=sha,
                         qualification=copy.deepcopy(data['qualification']))
                    for label, path, data, sha in loaded],
        episodes=episode_provenance,
        sampling=dict(method='inverse_style_sample_count', replacement=True,
                      style_sample_counts=dict(totals)))
    validate_dataset(merged)
    # Complete all validation/serialization before producing either output.
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_text, encoding='utf-8')
    torch.save(merged, output)
    return merged
