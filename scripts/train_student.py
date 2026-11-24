"""Bounded offline BC/state training from a sealed stereo manifest."""
import argparse, json
from pathlib import Path
from dh116.rl.training import train_bc
from dh116.rl.training import fit_state
from dh116.rl.student_data import load_dataset as load_dataset

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--mode', choices=('bc', 'state'), default='state')
    p.add_argument('--schema-only', action='store_true')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    data = load_dataset(a.manifest, expected_sha256=a.manifest_sha256, allow_schema_only=a.schema_only)
    fn = fit_state if a.mode == 'state' else train_bc
    result = fn(data, cfg['network'], cfg['bc'], output=a.output, steps=a.steps, batch_size=a.batch_size, heldout_initial_sha256=cfg.get('heldout_initial_sha256', ()), schema_only=a.schema_only, device=a.device, stereo_qualification=cfg.get('stereo_qualification'), camera_config=cfg.get('camera'))
    print(json.dumps(result, default=str))
if __name__ == '__main__':
    main()
