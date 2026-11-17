# 双手柔性物理证据落盘

Ubuntu 开发基线 `df40ce854e467b666f7e580a6d8faf3e6c74c637`。本工作包仅执行 CPU 回归。Isaac 步数、原生 PPO 更新和成功次数均未新增。

`DualHandEvidenceJournal.record` 直接接收双手构造器和 `DualHandPhysicalEpisode` 的记录。双手实际预动作状态及其原 371D 有效位写入 begin，原 24D 动作写入同一决策行，每个 480 Hz 测量行关联 SDK、杆求解器、环境、episode、reset、policy step 与 substep。SDK 初始化的 0→2 两条边界单独存储。每行写完后 flush，控制边界、物理终止和关闭时 fsync。独占创建文件，保留完整前缀，拒绝重写、时钟错位、终止后的无动作补行及超过完整 8 步控制窗预算的决策。

`inspect_evidence` 流式重算序列和 SHA256，区分完整控制窗、已记录但未结算的控制窗及原始终止原因。沿用 R0 的 owned JSON 编码，保留明确的 NaN 标签、数组 dtype 和掩码。在 CPU 读回中 66/371 有效，剩余 305 维保持未知。记录器不持有 Isaac、手或求解器。落盘与解析不能单独授予原生、奖励或 Teacher 资格。

## 回归

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_dual_hand_evidence_journal.py tests/test_dh116_dual_hand_episode.py tests/test_deformable_r0_replay.py tests/test_dh116_evidence_journal.py
```

49 项通过，19.16 秒。测试覆盖真实 Python 物理 owner 与显式 CPU SDK/solver doubles、两个控制窗、原保护触发的终止记录、恶化的时钟/动作/杆身份、损坏尾行及失败前缀。日志留在服务器私有 evidence 目录，SHA256 为 `f8e9a091aafa5911ae66fba94a50879daf64eb676f4ecd2d2e9e1930dc42a0b7`。

下一步将记录器连接实际双手原生控制窗，并用独立外层退出和真实接触账本验证该范围。保持冻结物理、24D/371D 合同和 30 mm / 5 s / 0.5 mm 标准。
