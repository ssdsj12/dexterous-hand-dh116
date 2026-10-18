"""Audit copied/adapted files; preserve the original manifest as provenance."""
import json
from pathlib import Path
from tools.migration_copy import sha256

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'migration_manifest.json').read_text())
rows=[];errors=[]
extra=json.loads((root/'shared_tests_manifest.json').read_text())
for entry in manifest['entries']+extra:
    path=root/entry['destination']
    current=sha256(path)
    adapted=current!=entry['sha256']
    if adapted and entry['category'] not in ('code','scripts','tests','shared_dependency','shared_tests'):
        errors.append(entry['destination'])
    rows.append(dict(destination=entry['destination'],original_sha256=entry['sha256'],
        current_sha256=current,status='adapted' if adapted else 'copy_verified'))
report=dict(copy_inventory_entries=len(rows),files=rows,unexpected_changes=errors,
    live_pending=manifest['live_pending'],runtime_physics_verified=False,
    status='static_audit_passed' if not errors else 'FAILED')
(root/'migration_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],len(rows),'files,',sum(r['status']=='adapted' for r in rows),'adapted')
if errors:raise SystemExit(1)
