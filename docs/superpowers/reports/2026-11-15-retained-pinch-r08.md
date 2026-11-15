# R08 retained Pinch v3 independent source re-audit

基线 `5eac3ac40ffaaae69a4458b7838d4dca9e806f04`；分支 `codex/pinch-postexit-replay-20261114`。只新增独立 verifier、测试和本报告。现存两份 untracked scaffold 先备份再扩展；旧消费 wrapper、recipe、原始数据及失败 receipt 不改写。

## 实际结果与资格边界

两个只读 CPU 重审均 exit0，原 receipt、intent、outer、native.log、全部原 output 文件前后 SHA 不变。最终 `verified_saved_physical_sources=true`，scope=`retained_pinch_v3_saved_physical_source_reaudit_only`。原 scheduler 仍 failed（receipt 的 `blocker_kind=failed_native`），原失败原因仍是 numeric verify 模块缺少 sha 的 AttributeError。

`native_qualification`、`teacher_admission`、`same_rod_grasp_qualified`、`grasp_qualified`、`native_execution_performed` 均 false；`new_physics_steps=new_solver_steps=ppo_updates=0`。没有 Isaac launcher、GPU计算、新训练或原预算重置，也没有 Teacher/Student 成功池准入。

独立路径固定导入 `tabletop_mesh_reset_numeric_state`，哈希用 `scheduler.file_hash`；不通过动态 `entry.numeric` / `entry.prior.sha`。真实父审查继续执行；identity/scope/source 用显式 context 绑定并恢复。重审前后核对新依赖源码；报告用 exclusive create、0600权限写到项目外私有新路径，拒绝已有输出、symlink、父symlink、项目内和相对路径。

| 原始绑定 | SHA256 / 实值 |
|---|---|
| receipt | `6bc124165c6cd907e84ac013b07c9f9df45b2ee1cd4f9499253b42d924adc4ef` |
| frozen command spec | `341a3ea1fb256474e1a1d1a65a2ad70238bc0e9da82f011d04285dbbb18ecf64` |
| frozen code/config | `f0b5bc8cd80210db27052f4f96f435c7ce48a7ff23d53fd533a53fd3ed63502f` |
| physical.jsonl | `2a99a771962c977cd3f0f652456d2bf9c2a852d49c361758ac0789ff20cf0bf9` |
| 原实际耗时 | `161.76650943607092` s，<=180 s |
| 原退出/锁 | outer=0，inferior=0，signal=null，native_lock_observed=true |
| 实际SDK | 234 = 2 setup + 232 measured，SDK 3..234 |

`lifecycle_qualified`、`reset_mesh_qualified`、`substep_contact_frame_qualified`、`same_clock_health_getter_qualified` 均 true。v2完整packet=232/232，raw pair samples=116676，capture errors=0；真实父审查同时重验v1 complete capture与v2 raw一致。health232行source_complete=true，capture/nonzero error rows均0。同一env0、reset1、rod0、episode `dual-pinch-a3dc90d1b02a45aeb03ff60d6cd96b72:one`，rod ticks12..2784。数值Reset两个实际快照，23项invariant均byte-equal，新增SDK/solver step均0。

这些是保存源诊断资格；`same_rod_contact_qualified`、`contact_success_qualified`、`hold_motion_qualified`、`live_r0_qualified`及旧 `mesh_reset_qualified` 均仍 false。没有hand-table contact proof、Hold/滑移或原生抓取成功资格。

## TDD和实际CPU验收

实现前RED：outer/stable helper缺失导致16项预期失败；source inventory/private writer缺失导致10项预期失败。全部实际验证通过canonical `scripts/python.sh`，没有将缺pxr当成物理失败。

原测试身份分组实际66 passed = 58快组 + 2真实旧wrapper错误 + 5真实父审查变体 + 1完整总入口。两条旧错误在真实fast->numeric rebinding中均复现：父成功访问缺少的numeric；父失败但artifact存在访问缺少的sha。两alias始终缺失；新真实测试没有整体mock待测inherited audit。success、failure+artifact、missing、corrupt JSON、合法JSON但numeric摘要实际变动五类各1 passed；失败类均不能封存合格源。完整入口测试追踪真实helper调用并核对所有原文件前后SHA。

root review指出私有621MB重审不能默认进入可移植/600s发布检查。因此第一次CLI完成后加显式门控：当前默认 **50 passed /16 skipped (2.43s)**；16项仅在 `DH116_RETAINED_PINCH_PRIVATE_TESTS=1` 且原receipt/必要工件/source/input可用时运行，缺文件明确skip。原58快组内有8项依赖私有证据，不能称其默认可移植。缺receipt探针把测试变量指向不存在路径（未改原文件），实际 **50 passed /16 skipped (2.11s)**；当前显式opt-in快组 **58 passed (10.63s)**。长审查测试主体未改；历史66项不能冒充当前默认66项全跑。

私有开发证据目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-r08-dev-20261115-o0yvw4e5`。

| 日志 | 实际结果 |
|---|---|
| portable-default.log | 50 passed /16 skipped |
| portable-missing-private-v2.log | 50 passed /16 skipped |
| portable-opt-in-fast.log | 58 passed /8 deselected |
| regression-final.log | 73 passed /140.87s |
| legacy-two-real-errors.log | 2 passed /51 deselected /850.36s |
| complete-saved-sources.log | 1 passed /65 deselected /514.14s |
| real-parent-{success,failure,missing,corrupt,numeric_changed}.log | 每类1 passed；498.70/360.23/498.80/498.45/497.85s |

原2错误和完整入口日志来自实际工具stdout后续持久化，`.meta.json`记录来源/session，未为日志再跑长检查；其他final日志从进程启动即重定向。全量旧测试日志SHA见 `verification-summary.json`，SHA=`4d62ab8ab85ff44cb2ff82e1767b17fc756d903386a8ac047cd387401945406b`；当前门控/源码/重审汇总见 `verification-summary-portable-v2.json`。

自己的首次顺序五类CPU pytest在首例完成前SIGINT，以并行5份独立副本：61 deselected/314.68s，不纳入通过计数，记录 `interrupted-cpu-group.txt`。首次缺文件probe因launcher清理PYTHONPATH导致私有插件导入失败（尚未收集测试），改canonical私有脚本入口后通过；旧失败log保留。既有73项回归初跑通过，再为完整留存stdout补跑，最终73 passed；不重复计数。没有中断/重启任何native。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_retained_pinch_v3.py --tb=short
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q tests/test_mesh_reset_numeric_state.py tests/test_mesh_reset_numeric_worker.py tests/test_mesh_reset_health_worker.py tests/test_flexible_health_postexit.py tests/test_dh116_dual_pinch_postexit.py --tb=short
```

## 两个独立重审身份

第一次为门控前实际基线，exit0 /452.175s。因源清单包含测试，门控后按root明确授权重新冻结test SHA并执行一次新只读CLI；两次均非native，不扩物理预算，没有因观察超时重跑。

| 身份 | report路径 / SHA256 | replay code/config SHA256 |
|---|---|---|
| 原test identity | `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-v3-independent-reaudit-20261115-98z31_fu/reaudit.json` / `0424cede83e70c877ad94f9426f79e2f75a16e7c486eb478d1d73df39865f2ec` | `7574d4e85cba4b4ac785c56d6064b7a9f394bbd7f397d7d78d4ba17f352cde79` |
| portable test identity（最终） | `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-v3-independent-reaudit-portable-20261115-a639k8su/reaudit.json` / `fcf8f92c0b2838e72a6fe6d026cea61c54759b30350b607c0da6b26db6db6a15` | `223eb19a09cc00a08816126b4821637cfca2daed8da0f089c22af676b7bcfed6` |

最终CLI exit0 /462.579s，report mode0600。两份报告除replay source inventory/fingerprint外全部相同。两目录的start.json/execution.json/replay.log与original-before.json/original-after.json保留真实argv、env、PID、exit及前后SHA。CPU重审耗时与原native161.7665s预算分别记录。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B scripts/verify_retained_dh116_pinch_v3.py --receipt /home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-pinch-fast-evidence-native/f0b5bc8cd80210db27052f4f96f435c7ce48a7ff23d53fd533a53fd3ed63502f/attempt-2ynpy5si/receipt.json --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-v3-independent-reaudit-portable-20261115-a639k8su/reaudit.json
```

上方为已执行命令；exclusive输出不能复用原路径。最终script SHA=`5ff95f80a4281ac492203af40f42dea6f994dc82307f75de70f508bc93cce156`，test SHA=`d30df1aae718e7d1c8675555566023ccc94f1d97a5c4c69737543d8686bad66a`。

## 保护和提交范围

3023个原tracked文件逐SHA均未改变，baseline inventory SHA=`b51b6bc53cbc82ac458e56f847181e3be314e35d0141029c2f31869b9dfb359e`。`Repository.verify_protection()`完成528保护文件和2历史Checkpoint核验；另核对 `/tmp/dh116-wp3-teacher128-v02/teacher_diagnostic/teacher_update_000000.pt`、`teacher_update_000001.pt`前后SHA不变（`ec5831ac4de039e5e77e2ee4a3420497a0c103a2a5966a1a73662d4997966511` / `c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999`）。原recipe复算仍与intent/approval/receipt一致。

仅提交scripts/verify_retained_dh116_pinch_v3.py、tests/test_retained_pinch_v3.py及本报告，不提交私有JSON/日志/原始621MB数据/模型，不改公共Registry/README/CHANGELOG。Ubuntu本地提交后交Windows发布；本包没有自行push/reset/force。Teacher66+rod+context及新native资格属于后续独立身份阶段。
