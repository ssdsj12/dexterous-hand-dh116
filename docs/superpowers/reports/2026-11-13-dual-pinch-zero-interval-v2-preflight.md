# 双手 Pinch 外部杆场景元数据复验

v1 在原 `spawn_scene` 读取 `wire_center` 时失败，未运行 SDK 初始化或求解器步。独立外层退出 1，无信号，app.close 返回。保留 v1 配方和回执 SHA256 `5825d7b12b8661ea540b30647bedb330691793d3595c2fa1e4bfa53b81a0ed02`。

本次仅为原场景入口提供从已锁定 `wire_points` 求得的元数据。原 `external_wire=True` 分支继续创建双手和台面，柔性杆构造和 Reset、2.1 mm 腕部保护、原 24D 动作、力和 R0 设定均沿用已发布路径。独立配方继续要求 2 初始化、8 测量、1 环境、GPU 3、180 秒、0 PPO。实际来源通过同一后置审计，静态手与台面接触仍为 NA。

完整原 spawn 的 CPU 源码测试 7 项通过。worker 和原后置审计相关 28 项 CPU 通过。真实 Isaac 原生状态等待新配方执行。

- 配方 SHA256 `80284f089f88ee289c56f4ef792689aab5729150c87469e017d7aee7a391581b`
- 调度器代码配置 SHA256 `e658c044aca8513ab2bef1c9dc198a06bab1cbc5a487de4816e23cff25ee715e`
- worker 前检 SHA256 `2f98cde255b559f484bb6ab0b8e13ad6d272be556d274bd437292666c477f028`

本范围只验证双手同杆的 2+8 运行链路。抓取、材料 Slip 和 Teacher 准入保持待验证。
