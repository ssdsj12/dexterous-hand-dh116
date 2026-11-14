# 柔性抓取的 Hold 时钟与 Slip 参考身份

基线 `a46f1d652e1d0f5064b625e7ab1e37c2b7d7f162`。CPU 回归复现了两个原问题。旧代码接受 2400 个允许的微短间隔，实际跨度 4.999760 秒仍输出五秒成功候选。材料锚点建立后换用另一个接触来源，旧代码会继续使用旧锚点。

## 五秒连续 Hold

首次合格端点冻结物理步和源时间戳。之后逐步核对相同 episode/reset/rod、连续物理步和固定杆步进，任何已接受的不合格或缺测样本清空本段 Hold。成功候选同时要求 2400 个连续完整间隔和实际时间跨度至少 5 秒。R0 Hold 进度使用已观测时长的非负增量，上限为 5 秒。

一般时间戳直接求差，五秒阈值采用 `>=`，无向下容差。只有当两端时间戳分别与各自规范 `physics_step / 480` 值完全相等时，使用对应有理时钟的步差除以频率，消除二进制浮点相减误差。例如从第 1446 步到第 3846 步，两个正常时间戳相减得到 4.999999999999999，精确固定时钟跨度为 5 秒。4.999760 秒反例不满足两端精确时钟等式，仍须等到下一条合格端点才可能通过。结果保留 Hold 起始物理步、起始实测时间戳和判定时长。

## 固定材料与切向参考

左右手分别冻结以下参考 UID 及其接触测量 `source`。

| 字段 | CPU 输入合同 |
| --- | --- |
| `material_reference_uid` | 两手共同的一根杆的固定静止弧长坐标参考。同一参考内记录各手实际材料站位和不确定度，首个可辨的双侧站位作为漂移基准。 |
| `tangential_reference_uid` | 每手独立的固定指腹锚点参考。`tangential_slip_m` 表示相对该锚点的切向位移大小，带对应不确定度。逐步差分不符合该字段语义。 |

给出数值或不确定度时必须给出非空参考 UID。完整缺测保持 NA，清除连续 Hold 时钟并保留冻结的锚点和参考。相同来源下改变 UID、同一 UID 下改变来源或两手使用不同材料参考时，拒绝样本并结束本 episode 的计时入口。有效 reset 才可清除参考、旧锚点和 Hold 历史。输出逐手列出已冻结 UID 与来源。现有原生适配器尚未提供这些合格的物理参考读回。

## CPU 核验

先运行六项新增回归，均因旧行为失败。补充了精确 480 Hz 非零起点回归，复现初版时间差的浮点边界。最终 172 项 CPU 回归通过，含 8 项新增测试。528 个保护文件和 2 个历史 Checkpoint SHA 核验通过。

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

本次未运行 Isaac 或训练。原生资格仍为 false，真实 480 Hz 源时钟和物理 Slip 参考的来源绑定仍待验证。30 mm / 5 s / 0.5 mm 和原有 24D / 371D 合同保持原标准。
