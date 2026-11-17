# Pinch 的柔性网格 Reset 与逐接触诊断配方

配方使用现有双 DH116、自由 DeformX 杆、原桌面碰撞和原腕部安全投影。动作序列与已用完预算的 Pinch 诊断一致。执行范围是 GPU 3、单 Isaac 实例、1 环境、2 个初始化步和 232 个采样步，限时 180 秒，0 次 PPO 更新。同一配方只调度一次。

- `tabletop_mesh_observation_evidence.py` 在既有 Reset 后同步同一批 11 个碰撞网格，读取 current、target 和 solver pose，保留 Reset 前后的实际读数。每个完成的物理步保留同一 solver owner 的 v1 完整 packet、v2 接触读数及 health getter。
- `prepare_dh116_mesh_reset_health_projected_pinch.py` 在该配方作用域接入原始 v1 packet 的保留逻辑，退出时恢复全局绑定。留存的 v1 与 v2 必须来自同一物理步，且原始字段相同。
- `verify_dh116_mesh_reset_health_projected_pinch.py` 在启动前校验真实参数任务回执和源文件。已执行的历史诊断只提供固定证据。
- `run_dh116_mesh_reset_health_work_packages.py` 使用独立 worker，并在进程退出后复核 mesh Reset、实际接触姿态及 health。终止前的记录按实保存。
- `deformx_mesh_reset_health_projected_pinch_v1.json` 绑定现有 SDK 物理链与新 881701 CUDA 扩展，要求 `mesh_reset_state.json`、`mesh_observation_scope.json` 和 `health_getter_scope.json`。本配方的原生结果将单独记录。

命令如下，执行记录使用独立私有路径。

```bash
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_mesh_reset_health_worker.py \
  tests/test_flexible_health_worker.py tests/test_projected_pinch_worker.py
bash scripts/python.sh scripts/run_dh116_mesh_reset_health_work_packages.py \
  --run --scope all --limit 1 --jobs 1 \
  --evidence-dir "$DH116_PRIVATE_EVIDENCE" \
  --native-approval "$DH116_FROZEN_BATCH_APPROVAL"
```

入口与前代回归在 Ubuntu CPU 环境中通过 **48 项**。日志 SHA256 `1ce2793b642f2ea7de5da56ad264dc22262fc640e2f2c4f3fb0d9b37351baff0`。真实 scheduler 测试使用替代进程核对回执，完整 CPU owner 测试使用 CPU 替身。原生诊断和 PPO 更新分别仍为 0。三种抓法的正式奖励及训练准入仍需独立证明。
