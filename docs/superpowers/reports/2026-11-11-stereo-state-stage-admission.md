# 143D 同步证据与分阶段准入

基线 `98ba1fb872636459c13212ae97425fbb8b7c56c5`。在现有多来源 manifest 和候选采集上补充 `dh116-state-synchronization-v1` 旁路证据。新的原生候选帧会绑定 frame、reset、单个 pre-action 物理 tick、时钟、标定和光轴参考 SHA，分别记录左右 Native Instance ID 是否实际存在。参考只用于标签及审计。训练输入继续使用左右 RGB 计算的深度。读取候选时重新从绑定的参考推导元数据，缺少实际证据或不一致时拒绝。历史候选仍可核验已有中心坐标标签。

Velocity 的线速度标签须在前一完整控制区间后取得同一 episode 的下一帧，核对 8 个 480 Hz 子步、1/60 s、连续 policy tick、更新的 camera frame、标定和处理身份。两帧必须共享通过参考深度核验的有效 centerline track，且前帧有同钟的中心状态读回。无类型的历史 track 和表面点都不开放 Velocity。Wire Geometry 需在 Center XYZ 准入后单独审核。Deformation 缺合格 Cosserat 标签，Contact、Slip 缺逐接触力学证据，Force Proxy 沿用真实来源资格核验。所有这些 Head 在本批保持关闭。8 个或更少实际初态的冷采集候选显式标为诊断用途，即使存在三个分区和有效标签，也拒绝正式优化。相同物理初态与同一来源族继续按连通组分配。

阶段报告分别提供有效样本、独立组、各分区训练/评价统计、有效覆盖与不确定度校准。在已有 8 行、2 个物理初态的真实历史归档上运行三个 CPU 模式。`schema_test` 退出 0，`diagnostic_fit` 使用 6 行、2 组真实 Center XYZ 标签完成 1 epoch，`formal_training` 退出 2，原因 `blocked_insufficient_independent_groups`。无新采集或 Isaac 进程。

| Head / 原始索引 | 诊断训练行 / 独立组 | MAE / RMSE / P95 (mm) | 训练内有效覆盖 | 不确定度 |
|---|---:|---|---:|---|
| Center X / 61 | 6 / 2 | 27.097 / 27.721 / 33.406 | 100% | 未校准 |
| Center Y / 62 | 6 / 2 | 23.313 / 23.315 / 23.733 | 100% | 未校准 |
| Center Z / 63 | 6 / 2 | 24.327 / 24.369 / 25.999 | 100% | 未校准 |
| Wire Geometry、Velocity、Deformation、Contact、Slip、Force Proxy 各项 | 0 / 0 | NA | NA | 未校准 |

100% 指 6 条进入拟合的行，原始 8 行的中心标签覆盖为 75%。误差仅为同一批数据的诊断拟合读回，validation/test 为 NA。该拟合没有授予泛化、部署、成功池或正式训练资格。

Windows CPU 136 项通过，0 失败，25.93 秒。三项先失败的回归覆盖冷 8 正式误准入、非连续时钟/track 和参考同帧绑定。聚合验证中发现的空旋转残差问题已修复并重跑。测试命令为：

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_state_multicohort.py tests/test_dh116_stereo_state_training_modes.py tests/test_dh116_stereo_state_supervision.py tests/test_dh116_stereo_cold_candidate_admission.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_runtime_task_state.py tests/test_dh116_stereo_runtime_actor_modes.py tests/test_dh116_stereo_training_cli.py tests/test_dh116_stereo_resolution_benchmark.py
python scripts/train_dh116_stereo_state.py --dataset-manifest <verified-manifest> --dataset-manifest-sha256 <manifest-sha> --mode diagnostic_fit --epochs 1 --batch-size 2 --device cpu --output <exclusive-output>
```

服务器独立目录 `/tmp/dh116-state-stage-final-xiFKgQ` 保留三个模式的报告和日志。监督源码 SHA `88d663689ae52f1161ea18a8225482f27e9d4497b831391057d6f18efcb1f05e`，诊断报告 SHA `9a633df67b0ff894d796bc95dc56a4aec5eecea6d6eb3a0421b270c946283705`。历史 Teacher Checkpoint SHA 复核为 `c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999`。新增数据仍依赖单独批准的同步采集，采集后先核验两侧 Native ID 与时钟，不直接启动正式训练。
