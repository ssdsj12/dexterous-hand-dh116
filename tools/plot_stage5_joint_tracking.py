"""Plot completed T1 Isaac readback with first-terminal and input-SHA checks."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


LABELS = ('Original M2 replay', 'Independent wrist guard', 'Coupled wrist guard')
COLORS = ('#1764a1', '#c65f0a', '#16836c')


def load(path):
    episodes = {}
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError('incomplete input JSONL')
            row = json.loads(line)
            identity = row['identity']
            if not row['live']:
                continue
            key = identity['env_id'], identity['episode_uid']
            rows = episodes.setdefault(key, [])
            if rows and (rows[-1]['terminal'] or identity['physics_step'] <= rows[-1]['identity']['physics_step']
                         or identity['timestamp_s'] <= rows[-1]['identity']['timestamp_s']):
                raise ValueError('closed episode reopened or physical clock regressed')
            rows.append(row)
    return episodes


def episode(data, env):
    found = [rows for (env_id, _), rows in data.items() if env_id == env]
    if len(found) != 1:
        raise ValueError('one explicitly identified first physical episode required')
    return found[0]


def series(rows, field):
    result = np.array([row['right_finger12'][field] for row in rows], dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('missing or nonfinite measured joint field')
    return result


def steps(rows):
    result = np.array([row['identity']['physics_step'] for row in rows])
    if not np.all(np.diff(result) == 1):
        raise ValueError('nonconsecutive first-episode physical samples')
    return result


def same_targets(groups):
    reference = {}
    for rows in groups:
        for row in rows:
            step = row['identity']['physics_step']
            target = row['right_finger12']['sdk_target_rad']
            if step in reference and reference[step] != target:
                raise ValueError('SDK targets differ on the common physical window')
            reference[step] = target
    return reference


def finish(figure, path, hashes):
    if path.exists():
        raise FileExistsError('derived figure already exists; choose an independent output directory')
    description = json.dumps(dict(source='Isaac simulation only; Visual Student M2 replay',
        window_s=256/480, qualified_grasp_successes=0, input_sha256=hashes), sort_keys=True)
    figure.savefig(path, dpi=160, metadata={'Description': description}, facecolor='white')
    plt.close(figure)


def plot(original, independent, coupled, output):
    paths = (original, independent, coupled)
    data = [load(path) for path in paths]
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for env, axis in zip((799, 932), axes):
        groups = [episode(item, env) for item in data]
        uid = groups[0][0]['identity']['episode_uid']
        if any(rows[0]['identity']['episode_uid'] != uid for rows in groups):
            raise ValueError('compared native episode UIDs differ')
        target = same_targets(groups)
        for rows, label, color in zip(groups, LABELS, COLORS):
            axis.plot(steps(rows), series(rows, 'position_rad'), color=color, label=label, lw=1.65)
            if rows[-1]['terminal']:
                axis.scatter(steps(rows)[-1], series(rows, 'position_rad')[-1], marker='x', s=42,
                             linewidth=1.8, color=color, zorder=4)
        ordered = sorted(set.intersection(*(set(steps(rows)) for rows in groups)))
        axis.plot(ordered, [target[step] for step in ordered], color='#222222', ls=':', lw=1.6)
        lower = groups[0][0]['right_finger12']['nominal_lower_rad']
        guard = groups[0][0]['right_finger12']['lower_guard_rad']
        for rows in groups:
            if not np.all(series(rows, 'nominal_lower_rad') == lower) or not np.all(series(rows, 'lower_guard_rad') == guard):
                raise ValueError('joint guard or nominal limit changed between arms')
        axis.axhline(lower, color='#666666', ls='--', lw=1.2)
        axis.axhline(guard, color='#b5232e', ls='-.', lw=1.3)
        axis.set(title=f'env {env}  |  episode {uid}', xlabel='Physics step (480 Hz)', ylabel='Right finger12 angle (rad)',
                 xlim=(1, 256), ylim=(-.02, .055))
        axis.grid(alpha=.18)
    figure.suptitle('Right finger12 tracking under identical source finger requests', fontsize=15, y=.975)
    figure.text(.5, .91, 'Isaac simulation readback | source: Visual Student M2 action tape | interventions: privileged wrist guards',
                ha='center', fontsize=9)
    handles = [Line2D([], [], color=color, lw=2, label=label) for color, label in zip(COLORS, LABELS)]
    handles += [Line2D([], [], color='#222222', ls=':', label='SDK command cache (shared window)'),
        Line2D([], [], color='#666666', ls='--', label='Nominal lower bound'),
        Line2D([], [], color='#b5232e', ls='-.', label='Existing lower guard')]
    figure.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(.5, .055), frameon=False)
    figure.text(.02, .02, 'Window: 256 steps = 0.533 s. Crosses mark first terminal samples. Qualified grasp successes: 0. Simulation only.', fontsize=8)
    figure.subplots_adjust(top=.82, bottom=.25, left=.07, right=.98, wspace=.24)
    first_path = output/'stage5_finger12_tracking.png'
    finish(figure, first_path, hashes)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    groups = [episode(item, 661) for item in data]
    if any(rows[0]['identity']['episode_uid'] != '1:661:0' for rows in groups):
        raise ValueError('env 661 comparison requires the identified first native episode')
    prefix = [[row for row in rows if row['identity']['physics_step'] <= 125] for rows in groups]
    if any(len(rows) != 125 for rows in prefix):
        raise ValueError('env 661 pre-intervention prefix must contain all 125 physical samples')
    for rows in prefix[1:]:
        for base, current in zip(prefix[0], rows):
            if (base['hand_object_load_n'] != current['hand_object_load_n']
                    or base['contacts']['effective_pad_load_n'] != current['contacts']['effective_pad_load_n']):
                raise ValueError('env 661 measured load prefix differs before intervention')
    for rows, label, color in zip(groups, LABELS, COLORS):
        thumb = np.array([row['contacts']['effective_pad_load_n'][1][0]
                          if row['contacts']['effective_pad_measurement_valid'] else np.nan for row in rows])
        hand = np.array([row['hand_object_load_n'][1] for row in rows])
        if not np.isfinite(thumb).all() or not np.isfinite(hand).all():
            raise ValueError('missing env 661 native contact load cannot be replaced with zero')
        axes[0].plot(steps(rows), thumb, color=color, lw=1.65, label=label)
        axes[1].plot(steps(rows), hand, color=color, lw=1.65, label=label)
        if rows[-1]['terminal']:
            for axis, value in zip(axes, (thumb[-1], hand[-1])):
                axis.scatter(steps(rows)[-1], value, marker='x', s=48, linewidth=2, color=color, zorder=4)
    axes[0].axhline(30, color='#b5232e', ls='--', lw=1.3)
    axes[0].text(6, 30.6, 'Original per-link guard: 30 N', color='#b5232e', fontsize=9)
    for axis in axes:
        axis.axvspan(1, 125, color='#777777', alpha=.06)
        axis.axvline(126, color='#666666', ls=':', lw=1)
        axis.axvline(185, color='#c65f0a', ls=':', lw=1)
        axis.set(xlabel='Physics step (480 Hz)', ylabel='Contact load (N)', xlim=(1, 256))
        axis.grid(alpha=.18)
    axes[0].set(title='Right thumb accepted normal load', ylim=(0, 34))
    axes[1].set(title='Right whole-hand object load (sum)', ylim=(0, 58))
    figure.suptitle('env 661 contact loads before and after wrist coordination', fontsize=15, y=.975)
    figure.text(.5, .91, 'Same first episode 1:661:0 | step 126: first left-hand pause | step 185: independent-guard excess_force',
                ha='center', fontsize=9)
    figure.legend(handles=handles[:3], loc='lower center', ncol=3, bbox_to_anchor=(.5, .07), frameon=False)
    figure.text(.02, .035, 'Shaded region: identical pre-intervention loads. The 30 N guard applies per link, while the hand sum is diagnostic.', fontsize=8)
    figure.text(.02, .01, 'Isaac simulation only | source: Visual Student M2 replay; simulated wrist-guard variants | 0.533 s window; no grasp success.', fontsize=8)
    figure.subplots_adjust(top=.82, bottom=.24, left=.07, right=.98, wspace=.24)
    second_path = output/'stage5_env661_contact_loads.png'
    finish(figure, second_path, hashes)
    return dict(figures=[str(first_path.resolve()), str(second_path.resolve())], input_sha256=hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('independent', type=Path)
    parser.add_argument('coupled', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.original, args.independent, args.coupled, args.output_dir), indent=2))
