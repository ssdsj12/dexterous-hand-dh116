# 柔性抓取的支撑证据与回合资格

基线 `8511a78b6ee716a0380f5808b25293807e96ad27`。CPU 回归先复现四项失败。已确认的 authored force 与其他支撑字段缺测时，旧聚合将违规写成 NA。对外力、附着和额外支撑分别注入一个已知违规步后撤除，旧候选会在随后 2401 个无违规样本端点结束时达到五秒 Hold。

`FlexibleGraspEpisode.step()` 现按三值 OR 聚合当步支撑证据。任一明确违规即为 true，全部明确无违规才为 false，其余保持 NA。该结果继续用于本步 R0，单次违规只记录当步贡献。

外力、附着或桌面之外的外部支撑任一明确发生后，`prohibited_support_latched` 保留至下次有效 reset，期间禁止产生新的 Lift 进度或 Hold 成功候选。撤除后的当步支撑可明确恢复为 false，历史标志继续约束该回合。桌面接触仍逐步检查，正常 approach 的桌面接触允许后续自然提起。Reset 清理历史标志后，同一流程可在新 episode 重新建立五秒有效 Hold。

执行以下 CPU 回归，164 项通过，含 4 项新增回归。528 个保护文件和 2 个历史 Checkpoint 校验一致。

```bash
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_deformable_grasp_task.py tests/test_dh116_deformable_task_budget.py \
  tests/test_dh116_deformx_thumb_contact.py tests/test_dh116_deformx_bridge_lifecycle.py \
  tests/test_lift_hold_window.py tests/test_dh116_stereo_runtime_rod_identity.py \
  tests/test_stereo_action_contract_audit.py tests/test_stereo_privileged_state_contract_audit.py \
  tests/test_privileged_teacher_reward_audit.py
```

本次未启动 Isaac 或训练。CPU 回归的成功候选仍带 `native_qualification=false`。实际外力、支撑、附着和材料锚点来源需在后续绑定中各自验证，原任务、动作、状态合同及 30 mm / 5 s / 0.5 mm 判据保留。
