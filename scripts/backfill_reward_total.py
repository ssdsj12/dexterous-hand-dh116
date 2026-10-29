"""Fill missing reward/total points from saved actual PPO rollout summaries."""
import json
import math
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from go2_pvcnn.control.dh116.tabletop_tensor_training import write_reward_scalars

run=Path(sys.argv[1])
events=EventAccumulator(str(run/'tensorboard'),size_guidance={'scalars':0}).Reload()
existing={x.step for x in events.Scalars('reward/total')} if 'reward/total' in events.Tags()['scalars'] else set()
count=0
with SummaryWriter(str(run/'tensorboard')) as writer:
    for line in (run/'iterations.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['iteration'] in existing:continue
        total=row['mean_step_reward']
        assert math.isfinite(total)
        assert math.isclose(total,sum(row['reward_terms'].values()),abs_tol=2e-6,rel_tol=2e-6)
        write_reward_scalars(writer,{},total,row['iteration'])
        count+=1
print(json.dumps(dict(tag='reward/total',added_points=count,previous_points=len(existing))))
