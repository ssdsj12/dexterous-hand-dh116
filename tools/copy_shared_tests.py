"""Keep tests for the shared contact/rod dependencies actually used by DH116."""
import json
from pathlib import Path
from tools.migration_copy import copy_file

root=Path(__file__).resolve().parents[1]
source=Path('/home/xk/coding/M1/.worktrees/t500-dual-panda-o6-mpc/Go2Pvcnn/tests')
entries=[]
for name in ('contact_bridge','wire_visual','free_rod','moving_contact','moving_contact_models'):
    file='test_fixed_dual_panda_o6_'+name+'.py'
    entries.append(copy_file(source/file,root,Path('tests')/file,'shared_tests'))
with (root/'shared_tests_manifest.json').open('x') as f:json.dump(entries,f,indent=2)
