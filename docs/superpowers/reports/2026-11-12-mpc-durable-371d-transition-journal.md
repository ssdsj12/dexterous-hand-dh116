# 原 MPC 的 371D／24D 私有持久账本

该变更供以后新目录重试时选择使用，当前冻结 T5 采集继续使用它启动时的代码。

`collect_dh116_mpc_privileged.py --transition-journal "$private_new_file"` 安装现有 sequence context sink 的旁路观察器。文件需要新建，父目录权限仅限当前用户。记录模式保持现有 480 Hz／60 Hz、MPC 动作和场景。在原 `TensorEnv.step` 已返回、SDK 与 World 两种时钟均增加八步、原 sequence 绑定动作及原生首回合身份之后，旁路逐步保存原始观测、字段有效性、实际 issued 动作和相机侧 episode UID。Tensor 原始位模式及 dtype 保留。已经离开首回合的槽位仍属于当时真实批次，但只有具备原采集器身份的槽位附有可用行身份。账本记录中的 terminal 来自原采集器，只表示回合终止。

每条记录和关闭状态都 flush/fsync，创建文件后的目录也 fsync。在原 sequence 成功归档后处理开始之前，已完成的记录已经持久写入。异常后状态为 `quarantined_exception`，强制退出或日志未关闭时为 `quarantined_interrupted`。末尾部分行不参与恢复。关闭标记需要原始 MPC 归档完成且行数与逐步记录的实际身份一致。任何恢复结果都保持 `qualified_for_bc=false`。原 2×32 路径及省略本选项时的行为沿用旧实现。

## 复验和恢复

CPU 回归 45 项通过，新增八项覆盖严格时钟、原始动作、无身份、预算、fsync、晚期异常、半行恢复、错误 SHA、私有路径和关闭行数。用已有 2×32 原生 v02 同钟 371D／24D 归档、原生 sequence context 和原 480 Hz 物理读回做 CPU 回放。32 个批次、64 个环境行逐位恢复了原始观测、mask 和 issued action。回放文件 188606 B，权限 0600，SHA256 为 `e56038708fbadea4bba417a1b8fcca0118517ce3c40ecc5bb5ca88ae04f70c4a`。按环境和步数外推，T5 账本约 42 MB。未开启新的 Isaac 进程。

新的失败尝试只有在进程退出、原始文件稳定后才读账本。先记录 SHA，再用 `recover_journal` 检查完整记录。保留 `recovered.samples` 与 `recovered.identities` 作为失败取证。恢复出来的时钟和 371D／24D 只能通过原 `join_execution` 再同原生视觉和 480 Hz 证据配对，不得用恢复记录补写缺失的成功标志。原 success 归档及其完整原生终止、抓法、力学和输出资格仍需独立通过。

```bash
export PYTHONPATH="$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
python=/mnt/mydisk/xk/envs/dh116/bin/python
"$python" - "$journal" "$recorded_journal_sha256" <<'PY'
from pathlib import Path
import sys
from go2_pvcnn.control.dh116.tabletop_mpc_capture_journal import recover_journal

result = recover_journal(Path(sys.argv[1]), sys.argv[2])
print(result.status, len(result.samples), sum(map(len, result.identities)),
      result.discarded_partial_line, result.qualified_for_bc)
PY
"$python" -m pytest -q -p no:cacheprovider tests/test_mpc_capture_journal.py \
  tests/test_mpc_privileged_capture.py tests/test_dh116_privileged_action_audit.py
```

当前已启动的 `wp3_mpc_t5_first_episodes_v01` 没有新账本选项。若它在 legacy finalize 失败，先保留外层退出记录、`teacher_native_physics.jsonl`、`native_physics_substeps.pt`、`mpc_state_capture/`、`all_executed_supervision.pt`、`native_sequence_context_unqualified.pt` 和存在的结果文件，分别读取各自 SHA 和实际完成状态。原 `mpc_privileged_pre_action_unqualified.pt` 若已经完整落盘，就用现有的 SHA 和行身份核验路径。若文件不存在，内存中的完整 371D 无法从 3844D 视觉监督归档恢复。可以审计已有的逐步数据和具备独立来源的观测子集。应将完整 371D 专家归档标为不完整，并为重试使用独立冻结源码、独立结果目录和该私有账本选项。
