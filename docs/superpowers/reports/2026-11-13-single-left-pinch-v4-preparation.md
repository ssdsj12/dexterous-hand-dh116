# 左手固定底座 Pinch v4 原生复验准备

v3 在 GPU 3 完整记录 240 个 SDK 步。103976 个原始 pair 全部属于杆与桌面。独立回放发现，闭合后的连续几何间隙约 4.7/5.4 µm，求解器实际节点经锁定 SDF 的间隙仍为 147/210 µm。v3 退出 1 由 FP32 与 FP64 归约不一致触发。独立核算最大 1.492 µN，原 1 mN 平衡限制继续使用。

## 本次变化

- 冻结单手固定底座左侧候选，保留原台面、41 个自由杆节点、重力、材料、执行器和保护阈值。初始 thumb spread 0.270050183 rad，thumb close 0.463812896 rad，index close 0.670957906 rad。原 quiet 48、闭合 160、静置 30 物理步。闭合速度分别 0.055666172 与 0.070218984 rad/s。
- 固定底座 CPU 路径净空下界 1.956487 mm，非目标形状净空预测下界 22.968 µm。预测 step 208 两个指腹的最小节点间隙为 −116.438 与 −149.750 µm。此姿态受限于固定底座 1.9 mm 门，双手六轴 2.1 mm 门单独保留。
- 启动前只读证据和源字节。建立实际 Articulation 后核实 source17 和 locked5 的同钟初始几何及 1.9 mm 下界。读写均沿用原构建器与动作内核。
- 独立审计校验原始 FP32 报告并重算 FP64 平衡。归还原本遗漏的 PHYSICS_READY 订阅，保留原 clear_instance、STOP、app.close 及全停 GDB 取证。新增回执区分 API 返回、Python 引用、外层进程状态。
- 独立解析原始逐指 pair、首次负载、同一步同一 solver substep 的双指正法向载荷。此范围只验证固定底座接触与反作用，不授予 Lift、Hold、Slip、双手或 Teacher 资格。

## 边界

GPU 3 单实例，最多一次，180 秒，1 环境，2 初始化 + 238 测量 SDK 步，0 PPO。沿用已有精确 SHA 与共享锁门。禁止同指纹重复运行。配方 SHA256 `740e52700102e5d90a6367b14a272aca9a3a0fbcde5ee5901169573f68d91fa5`，调度器代码配置 SHA256 `d7f87b161b3283a31ffe8a7befbbce0050f7d6bcc4b1473020e2e117067847cc`，worker 字节前检 SHA256 `089cf938a7d50e93eb7bc49c4155851a6f83320f8f0680481fdf669169ce3f21`。

## CPU 回归

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_pinch_v4_worker.py tests/test_dh116_pinch_v4_audit.py tests/test_dh116_pinch_force_precision.py tests/test_simulation_owner_cleanup.py tests/test_dh116_pinch_v3.py tests/test_dh116_pinch_v2.py
```

121 项通过，93.04 秒。原生配方尚未执行。保留 v3 的全部产物和失败状态。下一步在冻结指纹上验证实际接触、退出及安全边界，再决定双手 Pinch 控制窗。
