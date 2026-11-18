"""Bulk host transfer preserves pre-reset collector evidence and first-terminal rules."""
from copy import deepcopy
import json

import pytest
import torch

from test_vision_evaluation import payload
from test_vision_evaluation_trials import manifest
from go2_pvcnn.control.dh116 import tabletop_vision_evaluation as evaluation


def frames():
    first = payload()
    first['done'][:] = False
    first.update(slip_applicable=torch.zeros(2, dtype=torch.bool),
        prior_anchor_latched=torch.zeros(2, dtype=torch.bool),
        slip_reason_code=torch.ones(2, dtype=torch.long),
        slip_reason_codes={'valid': 0, 'no_anchor': 1, 'no_contact': 2,
                           'measurement_invalid': 3, 'anchor_invalid': 4, 'inactive': 5})
    terminal = deepcopy(first)
    terminal['done'][0] = terminal['native_success'][0] = True
    terminal['lift_m'][0] = .035
    terminal['qualified_hold_s'][0] = 5.
    terminal['slip_m'][0] = .0002
    terminal['slip_valid'][0] = terminal['slip_applicable'][0] = True
    terminal['prior_anchor_latched'][0] = True
    terminal['slip_reason_code'][0] = 0
    # A genuinely invalid measurement remains invalid, including its full
    # diagnostic vector. It must not become an eleven-joint valid sample.
    terminal['measurement_valid'][1] = False
    terminal['torque_valid'][1, 11] = False
    terminal['lift_valid'][1] = False
    terminal['lift_m'][1] = float('nan')
    last = deepcopy(first)
    last['done'][:] = True
    last['lift_m'][0] = 99.  # A repeated episode cannot replace first terminal.
    last['bilateral_opposed_contact'][1] = False
    last['action'] = torch.arange(48, dtype=torch.float32).reshape(2, 24) / 100.
    return first, terminal, last


def collect(packets):
    collector = evaluation.EvaluationTelemetryCollector(manifest(),
        checkpoint_sha256='a'*64, min_trials_per_style=1)
    raw = []
    collector.attach_raw_sink(raw.append)
    collector.begin_trials()
    for packet in packets: collector(packet)
    return collector.summary(), raw


def test_host_snapshot_preserves_dtype_shape_values_and_non_tensor_metadata():
    assert hasattr(evaluation, '_cpu_payload_snapshot'), 'bulk payload snapshot is missing'
    packet = frames()[0]
    packet['lift_m'] = torch.tensor([.031, .01], dtype=torch.float64, requires_grad=True)
    packet['extra_labels'] = ['left', 'right']
    before = deepcopy(packet)
    copied = evaluation._cpu_payload_snapshot(packet)
    assert copied is not packet and copied.keys() == packet.keys()
    for key, value in packet.items():
        actual = copied[key]
        if isinstance(value, torch.Tensor):
            assert actual.device.type == 'cpu'
            assert actual.dtype == value.dtype and actual.shape == value.shape
            assert not actual.requires_grad
            torch.testing.assert_close(actual, value.detach(), rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(value, before[key], rtol=0, atol=0, equal_nan=True)
        else:
            assert actual is value  # No enum, timestamp, scalar or list rewriting.
    assert packet['lift_m'].requires_grad


def test_collector_skips_transfer_before_begin_then_takes_one_bulk_snapshot(monkeypatch):
    assert hasattr(evaluation, '_cpu_payload_snapshot'), 'bulk payload snapshot is missing'
    original = evaluation._cpu_payload_snapshot
    calls = []
    def tracked(packet):
        calls.append(packet)
        return original(packet)
    monkeypatch.setattr(evaluation, '_cpu_payload_snapshot', tracked)
    collector = evaluation.EvaluationTelemetryCollector(manifest(),
        checkpoint_sha256='a'*64, min_trials_per_style=1)
    packet = frames()[0]
    collector(packet)
    assert not calls and collector.summary()['completed_count'] == 0
    collector.begin_trials()
    collector(packet)
    assert len(calls) == 1 and calls[0] is packet


def test_bulk_snapshot_keeps_existing_collector_raw_json_and_terminal_semantics(monkeypatch):
    assert hasattr(evaluation, '_cpu_payload_snapshot'), 'bulk payload snapshot is missing'
    packets = frames()
    report, raw = collect(packets)
    # Run the established CPU collector without the transport adapter. The
    # aggregation and raw JSON are the oracle; no outcome formula is copied.
    monkeypatch.setattr(evaluation, '_cpu_payload_snapshot', lambda packet: packet)
    reference_report, reference_raw = collect(deepcopy(packets))
    assert report == reference_report
    assert json.dumps(raw, allow_nan=False) == json.dumps(reference_raw, allow_nan=False)
    rows = report['trial_rows']
    assert [row['sample_count'] for row in rows] == [2, 3]
    assert rows[0]['peak_lift_m'] == pytest.approx(.035)
    assert rows[0]['native_success'] is True
    assert rows[1]['invalid_measurement_samples'] == 1
    assert raw[0]['sample']['slip_m'] is None
    assert raw[3]['sample']['estimated_torque_nm'] is None
    assert len(raw[3]['torque_diagnostics']['valid_mask']) == 12
    assert all(len(row['sample']['action']) == 24 for row in raw)
    assert all(row['sample']['dt_s'] == .01 for row in raw)
