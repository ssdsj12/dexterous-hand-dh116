# Pinch 无相机原生初始化复验

Ubuntu 与 GitHub 在启动前核对为 `9ab2cc9965a232ab7fcd7a0712a010b05886f6d3`。只执行已冻结的 Pinch 配方 `5f626a18ef57051ddcadb80ab6910180786bb5c67a5d9a20a75df819899dbbdf`。CUDA 可见卡为 GPU 2，Isaac 设备回执为 `cuda:0`，单 Isaac、1 个环境、2 个 SDK 初始化步。控制、柔性求解器、Teacher 查询、模型加载、RGB 和 PPO 均为零。

## 实际初始化读回

同一场景 SDK 从步 0 到步 2，时间为 0.004166666884 秒。两侧初始化读回均通过，读回前无额外 seed 覆写。

| 手 | 关节最大误差 rad | 掌根位置误差 m | 掌根旋转误差 rad |
|---|---:|---:|---:|
| 左 | 3.72235e-8 | 1.08061e-8 | 2.34664e-7 |
| 右 | 4.13113e-8 | 1.92846e-8 | 2.63207e-7 |

PhysX 返回 34 个碰撞体的 stored-cooked 几何，每个一组原始凸包。**严格拓扑通过为 0/34**，每项均为 `COOKED_HULL_TOPOLOGY_INVALID_0`。原始获取结果和失败缘由已保存。当前证据覆盖 getter 返回的 stored-cooked 几何，尚未证明实际 `PxShape` 的附着与支撑几何。

## 关闭和 Teacher 准入

Kit stat cache 释放接口返回。释放前后 SDK 都在第 2 步。STOP 后 SimulationManager 物理视图为空，`app.close()` 正常返回。外层独立回执记录 **48.23 秒自然退出，退出码 1**。本次没有 SIGSEGV、强制结束或重试。非零退出由严格碰撞体证据不完整触发。

本次落实的是当前代码的双手初始化读回。Teacher 的数值 Reset 证明还需要同身份的杆节点位置、速度、四元数及真实碰撞绑定的受保护原始证据。因此 `resolved_reset_manifest`、原生准入和 PPO 保持待完成。双 RGB 首帧属于独立诊断，本轮未启动。

## CPU 验证

本工作包运行初始化、受保护几何读回、Kit cache 生命周期与 Registry 回归，**195 项通过**。
