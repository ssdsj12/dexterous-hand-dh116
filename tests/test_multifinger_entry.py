"""Exercise the entry's metadata writer without importing Torch or Isaac."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/run_dh116_multifinger_lift.py"


def metadata_writer():
    tree = ast.parse(ENTRY.read_text(encoding="utf-8-sig"))
    writer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "write")
    calls = []

    def original_write(path, data):
        calls.append((path, data))
        return data

    namespace = dict(Path=Path, hashlib=hashlib, ROOT=ROOT, state={},
                     options=SimpleNamespace(hand_force_range=(2.0, 3.0)),
                     original_write=original_write)
    exec(compile(ast.Module(body=[writer], type_ignores=[]), str(ENTRY), "exec"), namespace)
    return namespace["write"], calls


class TestMultifingerEntry(unittest.TestCase):
    def test_sampling_metadata_uses_actual_controller_timing(self):
        for hz in (480,960):
            with self.subTest(hz=hz):
                write,_=metadata_writer()
                write.__globals__['state']['control']=SimpleNamespace(
                    dt=1/hz,monitor=SimpleNamespace(window_steps=round(.3*hz)))
                result=write(Path('resolved_grip_lift.json'),{})
                self.assertEqual(result['raw_force_and_slip_sampling_hz'],hz)
                self.assertEqual(result['stable_window_raw_steps'],round(.3*hz))

    def test_resolved_contract_marks_force_range_as_success_gate(self):
        write, calls = metadata_writer()
        data = dict(force_limits_used_as_success_gate=False,
                    force_error_used_as_success_gate=False, force_goal=None)
        path = Path("resolved_grip_lift.json")
        result = write(path, data)

        self.assertIs(result["force_limits_used_as_success_gate"], True)
        self.assertEqual(result["force_goal"], "per_hand_total_normal_band")
        self.assertEqual(result["hand_force_range_n"], [2.0, 3.0])
        self.assertEqual(result["initial_hand_force_setpoint_n"], 2.5)
        self.assertIs(result["force_error_used_as_success_gate"], False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], path)
        self.assertIs(calls[0][1], data)

    def test_unrelated_metadata_keeps_existing_force_gate_flag(self):
        write, calls = metadata_writer()
        data = dict(force_limits_used_as_success_gate=False, force_goal=None)
        expected = data.copy()
        result = write(Path("configuration.json"), data)

        self.assertEqual(result, expected)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1], data)
