"""Select the six calibrated DH116 pad references for the grasp runner."""
import argparse, hashlib, json
from pathlib import Path

CHANNELS = ('thumb', 'index', 'middle')

def select(surfaces, output):
    source = Path(surfaces).resolve(strict=True)
    data = json.loads(source.read_text())
    refs = [r for r in data['references'] if r.get('digit') in CHANNELS]
    if len(refs) != 6 or {(r['side'], r['digit']) for r in refs} != {
        (side, digit) for side in ('left', 'right') for digit in CHANNELS}:
        raise ValueError('expected exactly one calibrated reference per DH116 side/digit')
    refs = sorted(refs, key=lambda r: (('left', 'right').index(r['side']), CHANNELS.index(r['digit'])))
    if any(not all(map(lambda x: isinstance(x, (int, float)), r['point'])) or
           not all(map(lambda x: isinstance(x, (int, float)), r['normal'])) for r in refs):
        raise ValueError('non-numeric reference geometry')
    report = {
        'schema': 'dh116-grasp-references-v2-middle-hook',
        'asset_sha256': data['asset_sha256'],
        'layers': data['layers'],
        'source_report_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'grasp_qualified': False,
        'scope': 'Calibrated DH116 thumb/index/middle references; physical grasp NOT qualified.',
        'references': refs,
    }
    output = Path(output)
    if output.exists(): raise ValueError('refusing existing grasp reference file')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surfaces', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(select(args.surfaces, args.output), indent=2))
