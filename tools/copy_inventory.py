"""Copy approved DH116 scope; live outputs are explicitly deferred."""
import json
from pathlib import Path
from tools.migration_copy import copy_file

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('/home/xk/coding/M1/.worktrees/t500-dual-panda-o6-mpc')
PROJECT=WORK/'Go2Pvcnn'
DATA=Path('/home/xk/coding/M1/assets/dh116_v2')
SKIP={'.git','__pycache__','.pytest_cache','.venv','.mypy_cache'}
entries=[];skipped=[]


def one(source,dest,category):
    entries.append(copy_file(source,ROOT,dest,category))
    if len(entries)%100==0:print('verified',len(entries),flush=True)


def tree(source,dest,category,exclude=()):
    for p in sorted(source.rglob('*')):
        rel=p.relative_to(source)
        if set(rel.parts)&SKIP or rel.parts[0] in exclude:continue
        if p.is_symlink():
            skipped.append(dict(source=str(p),reason='symlink_requires_review'));continue
        if p.is_file():one(p,Path(dest)/rel,category)


tree(PROJECT/'go2_pvcnn/control/dh116','src/go2_pvcnn/control/dh116','code')
for name in ('contact_bridge.py','wire_visual.py'):
    one(PROJECT/'go2_pvcnn/control/fixed_dual_panda_o6'/name,Path('src/go2_pvcnn/control/fixed_dual_panda_o6')/name,'shared_dependency')
for folder,pattern in [('scripts','*dh116*.py'),('tests','test_dh116*.py')]:
    for p in sorted((PROJECT/folder).glob(pattern)):one(p,Path(folder)/p.name,folder)
for folder in ('docs/superpowers/specs','docs/superpowers/plans','notes/log'):
    for p in sorted((WORK/folder).glob('*dh116*.md')):one(p,Path('docs/source')/folder/p.name,'documentation')
tree(DATA,'archive/dh116_v2','historical_evidence',exclude=('table_margin_raise1mm_trial1','table_margin_raise1mm_trial1.log'))
tree(Path('/home/xk/coding/M1/assets/fixed_dual_panda_dh116_v1'),'archive/fixed_dual_panda_dh116_v1','original_asset')
backend=Path('/home/xk/coding/DeformX/PyElastica-Mesh')
for name in ('co_sim','elastica','assets','tests'):
    tree(backend/name,Path('vendor/DeformX')/name,'backend')
for name in ('LICENSE','README.md','pyproject.toml','uv.lock','IMPLEMENTATION_LOG.md'):
    one(backend/name,Path('vendor/DeformX')/name,'backend_metadata')
for p in sorted(Path('/home/xk/桌面/dh116').rglob('*')):
    if p.is_file() and 'DH2116' not in p.name:
        one(p,Path('resources/manufacturer')/p.relative_to('/home/xk/桌面/dh116'),'manufacturer')
for p in Path('/home/xk/桌面').glob('DH116*'):
    if p.is_dir():tree(p,Path('resources/manufacturer')/p.name,'manufacturer_models')
manifest=dict(status='static_copy_complete_runtime_not_relocated',entries=entries,skipped=skipped,
    live_pending=[str(DATA/'table_margin_raise1mm_trial1'),str(DATA/'table_margin_raise1mm_trial1.log')],
    original_files_removed=False)
with (ROOT/'migration_manifest.json').open('x') as f:json.dump(manifest,f,indent=2,ensure_ascii=False)
print('STATIC COPY COMPLETE',len(entries),'files;',len(skipped),'links require review',flush=True)
