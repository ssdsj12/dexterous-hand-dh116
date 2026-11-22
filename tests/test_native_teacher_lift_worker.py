"""Fresh native Lift reward configuration must be explicit and immutable."""
import importlib
import json
from pathlib import Path

import pytest

from go2_pvcnn.control.dh116.tabletop_training_reward import load_config


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/dh116_training/approach_contact_prelift_v1.json"
LIFT = ROOT / "configs/dh116_training/tripod_supported_lift_diagnostic_v2.json"


def api():
    name = "go2_pvcnn.control.dh116.tabletop_native_teacher_lift_trial"
    assert importlib.util.find_spec(name), "native Lift trial contract is missing"
    return importlib.import_module(name)


def test_distinct_file_pins_opt_in_lift_configuration():
    reader = api()
    base = load_config(BASE)
    configured = load_config(LIFT)
    assert reader.read_lift_trial(base) is None
    trial = reader.read_lift_trial(configured)
    assert trial.prelift_stable_s == .25
    assert trial.config_sha256 == configured.config_sha256 != base.config_sha256


def test_changed_reward_bytes_cannot_activate_lift(tmp_path):
    reader = api()
    path = tmp_path / "candidate.json"
    path.write_bytes(LIFT.read_bytes())
    loaded = load_config(path)
    data = json.loads(path.read_bytes())
    data["lift_diagnostic"]["prelift_stable_s"] = .125
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA"):
        reader.read_lift_trial(loaded)


def test_valid_shas_do_not_admit_a_different_lift_variant(tmp_path):
    reader = api()
    data = json.loads(LIFT.read_bytes())
    data["lift_diagnostic"]["prelift_stable_s"] = 0.
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Lift trial"):
        reader.read_lift_trial(load_config(path))

def test_replayed_lift_rejects_substituted_height_potential_and_total():
    from copy import deepcopy

    from test_supported_lift_observation import reference, sample
    from test_teacher_lift_progress import row
    from go2_pvcnn.control.dh116.tabletop_teacher_lift_progress import SupportedLiftProgress
    from go2_pvcnn.control.dh116.tabletop_teacher_lift_reward import PROFILE, SCHEMA
    from go2_pvcnn.control.dh116.tabletop_training_reward import _excluded

    config = load_config(LIFT)
    trial = api().read_lift_trial(config)
    rod = sample()
    progress = SupportedLiftProgress(reference(rod), grasp="tripod", prelift_stable_s=.25)
    for tick in range(3, 126):
        before = progress.observe(row(rod, tick))
    assert before.phase.phase == "Lift"
    rod["node_positions"][0, :, 2] += .015
    after = progress.observe(row(rod, 126))
    term = dict(raw=after.observation.off_support_lift_m, unit="m",
                source="measured_supported_rod_fixed_reset", contribution=2.,
                valid=True, applicable=True, history_reason=None)
    # The cost is per second. One physical sample at 480 Hz incurs .01.
    def context(progress):
        return dict(config_sha256=trial.config_sha256, schema_sha256=SCHEMA,
                    lift_phase=progress.phase.phase, prelift_stable_s=progress.phase.prelift_stable_s,
                    lift_reset_rod_sha256=progress.observation.reset_rod_sha256)

    reward = dict(profile=PROFILE, profile_version=2, config_sha256=trial.config_sha256,
                  schema_sha256=SCHEMA, gamma=config.gamma, phi_pre=0., phi_post_raw=2.,
                  endpoint_phi=2., cost_per_second=[4.8], event=0., event_reasons=[],
                  reward=2. * config.gamma - .01, terminated=False,
                  pre_context=context(before), post_context=dict(context(after), lift_shaping_delta=2. * config.gamma),
                  pre_terms=dict(lift=dict(term, raw=0., contribution=0.), hold=dict(_excluded())),
                  terms=dict(lift=term, hold=dict(_excluded())))
    api().verify_lift_settlement(reward, before, after, config, trial)
    for field in ("raw", "contribution"):
        altered = deepcopy(reward)
        altered["terms"]["lift"][field] += .001
        with pytest.raises(ValueError, match="measured Lift"):
            api().verify_lift_settlement(altered, before, after, config, trial)
    altered = deepcopy(reward)
    altered["reward"] += .001
    with pytest.raises(ValueError, match="Lift settlement arithmetic"):
        api().verify_lift_settlement(altered, before, after, config, trial)
