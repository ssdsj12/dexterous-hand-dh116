# Power 掌面反作用力写入 Isaac 原生关节

在 Ubuntu GPU 2 上运行单 Isaac 实例。复用原始 DH116 左右手、安装关节、关节安全和桌面净空。用现有原生 Reset 初始化两侧掌部，分别核对在线 base_link 与已封存 SDF 的三角几何 SHA、网格相对连杆变换、SDK 位姿和本地质心。两份原始 DeformX 掌面 SDF 刚性变换到当步实测左右掌位；每侧独立自由杆校准各执行 12 个求解器子步。

SDK 3、4 记录无负载对照。SDK 4 的 DeformX 反作用力经连杆原点到质心的力矩换算，写入各自 Articulation 的原生 WrenchComposer，复算实际缓冲区内旋转回世界系的力矩和力。SDK 5 记录由此产生的掌部响应。三个测量步均通过原始 DH116 关节、Mimic 和全连杆桌面净空保护。

| 指标 | 左掌 | 右掌 |
| --- | ---: | ---: |
| DeformX 原子接触对 | 7 | 7 |
| 反作用力向内法向投影（N） | 4.25384 | 4.01728 |
| 掌部线速度变化（mm/s，沿输入反作用力方向投影） | 4.44487 | 4.39664 |
| 加载后最小桌面净空（mm） | 2.10423 | 2.10931 |
| 聚合力守恒残差（N） | 3.37e-7 | 3.58e-7 |

两次无力控制的速度投影变化均低于 1e-9 m/s。App 自然退出 0，44.76 s，无强制结束。更早的冻结版本由于在 AppLauncher 前提前加载 USD 被原启动护栏拒绝，没有启动 Isaac。修复仅移动读取 USD 的导入位置；CPU 独立进程回归覆盖此边界。

入口 `python -B -m scripts.verify_dh116_power_live_palm_feedback --rays <native/palm_rays.json> --left-bake <left bake> --right-bake <right bake> --output <new native directory> --plan <new freeze plan>`。原生启动读取完全相同的输入和冻结文件并附 `--approved-frozen-sha256`。readiness 附原始账本、自然退出回执和独立失败回执的 SHA。

这次验证覆盖左右原生掌部收到 DeformX 实测负载后的响应。两侧采用彼此独立的夹具自由杆，下一步将两个掌面并入同一根任务杆的手指与桌面接触回路，再验证五指闭合包覆、稳定接触及离桌。零 Teacher 查询、零 PPO 更新，所有任务资格标志保持未通过。

最终 CPU 联合回归 189 项通过。
