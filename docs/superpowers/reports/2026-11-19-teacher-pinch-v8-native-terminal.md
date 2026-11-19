# Pinch v8 原生终止与右拇指净空审计

按已批准的冻结指纹 `e4f597aecbf764ebf97fd149b7fcc175f832aca79973fec4b1ab86e24219f13a` 在 GPU 3 执行一次。使用 `Repository.start_native` 和签名握手，已核验调度回执的独立实例锁。本轮只执行这一次。

| 证据 | 结果 |
| --- | --- |
| Reset | 2 个 SDK 初始化步。Reset 数值读回 2/2，字节一致，无追加物理或求解步骤 |
| 采集 | 1 个控制决策，SDK 第 3–7 步共 5 个连续测量子步，60 个求解器子步 |
| 终止 | SDK 第 7 步，策略步 0，子步 4，`post_step_source_clearance:right` |
| 轨迹 | 1 条 terminal transition，`done=true`，`bootstrap_mask=false`，`gae_carry=false`，`truncated=false` |
| PPO | 0 次更新，无新 Checkpoint |
| 关闭 | `App.close()` 返回，外层和 GDB inferior 均为 1，耗时 124.515 秒。GDB 未记录崩溃信号。资源释放证据未通过，未写成功关闭标记 |

账本 36 条，`sequence` 为 0 至 35。控制循环终止后尝试补满 rollout 时发出 `fresh_verified_reset_factories_required`。物理终止和 terminal transition 已在该错误之前落盘。

由原 source 凸包复算，右手限制件为 `finger13_link`。Reset 时右手净空为 2.116346 mm。第 6 步为 2.102836 mm，第 7 步为 2.096474 mm，低于原 2.100 mm 保护线 3.526 µm。原始逐步记录的右手安全投影没有截断。使用 SDK 读回的 `base_link` 刚体变换和 `finger13_link` 实际姿态分解，第 7 步相对于 Reset 的基座竖向平移贡献为 +1.31 µm，基座旋转贡献为 −0.32 µm，连杆相对于基座的变化贡献为 −20.86 µm。这是几何分解。主动关节跟踪、被动联动和接触作用的动力学贡献仍需隔离验证。

```bash
python -B -m scripts.audit_dh116_pinch_ppo_clearance \
  --journal "$V8_JOURNAL" \
  --journal-sha256 f86e670f7912d59d30a7a68f5e57f3c1c8ab759e3ee8c9db0a0151299eafb88b \
  --output "$NEW_CLEARANCE_AUDIT"
```

原生回执 SHA `98d48d1addf7115ac69a3f7ab85235493f29918683043f083185121fad0a3309`，净空复核 SHA `637d4b3e7b58cf3d0f6eb4fc505ae5887cb56d034c938f2f648d04647b153135`，连杆位姿分解 SHA `404b15f2aa093e3a8f14bc1506155bbd233cfa99d4ffac50eda5c7e4b095536a`。99 项 CPU 回归通过。原始日志、手部状态和资源生命周期文件留在服务器证据目录。Registry 将 v8 标为 blocked。下一项优先复核右拇指闭合时的连杆动态净空，之后再为新配方核验同一保护线下的完整 rollout。
