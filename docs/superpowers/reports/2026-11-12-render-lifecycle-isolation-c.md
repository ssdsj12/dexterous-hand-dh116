# C 渲染生命周期阶梯与同步采集边界

现有 drop/hang 生成器产物和 AppWindow/Carb 退出证据归属 V5。本工作包增加独立入口 `scripts/probe_dh116_render_lifecycle.py`，沿用已固定的 IsaacLab 4.5 headless rendering experience 和 DH116 单实例锁。所有情况独立进程、独立目录，应用关闭使用 `fast_shutdown=false`，外层记录实际退出码。CPU 预检不会启动 Isaac。

| 用例 | 新增内容与边界 |
|---|---|
| V0 | 空应用启动和关闭 |
| V1 | 简单 USD、一个针孔相机和一个渲染产品，无 annotator |
| V2 | V1 加一个 RGB annotator |
| V3 | 同一产品读取 RGB、径向距离、Optical-Z 和数值实例 ID。ReferenceTime 和 IsaacReadTimes 绑定实际帧。已知平面验证中心和边缘 |
| V4 | 既有 DH116 刚性代理任务中 env0 的左右相机，直接从实际两个渲染产品读回，无上游 DatasetGenerator。两次完整采集之间只执行一次 24D 零动作。独立无碰撞标记位移用于原生图像配对测试 |
| V5 | 现有原始 drop/hang 证据独立保存 |

V3 和 V4 都在采集边界只调用一次渲染，随后读取所有相机通道。初始化最多八次 render-only 更新，逐次核对状态并排除在正式边界之外。V4 使用现有 IsaacLab SimulationContext.render，从实际任务时钟、SDK 步号、机器人状态、对象位姿速度和 episode/reset 读回建立边界。V4 的 `rod_step=null` 由刚性任务身份决定，柔性 Cosserat 采集资格需另行实测。跨相机帧号不同、渲染陈旧、读回中帧变动、物理状态或杆时钟变动均使结果失败。ID 复制原生数值和映射。深度按原始 `distance_to_camera` 与 `distance_to_image_plane` 分别存档，另存有效性和实际 USD 针孔内参摘要。2 mm 是本探针的检查容差，需由 V3 原生已知平面数据验证。

清理路径以稳定 USD 路径解绑自己创建的 annotator，再销毁自己创建的 render product。V2/V3 在产品仍有效时停下 Replicator 并等待完成，之后关闭自建 stage；V4 的任务 stage 和应用仍由既有启动器管理。此路径无需外部强制结束。外层工具独立记录产物摘要审计、同步和深度、App.close 返回、实际进程退出以及重启状态。单轮报告不会授予重启资格。

## CPU 结果

Ubuntu CPU 回归 20 项通过。新增用例覆盖单次渲染、双视图复制、陈旧和跨相机渲染时钟、读回期间变化、物理与 rod 时钟变化、中心与边缘径向/光轴区分，以及产物与 App.close 都存在时退出 139 仍失败。V0–V4 固定预检均通过。此提交没有启动 Isaac。预检、日志和限定启动配方位于 `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/f6-f7-lifecycle-preflight-GcTvRw`。

| 证据 | SHA256 |
|---|---|
| CPU 测试日志 | `29173821523d719720224e43822305d0e5d8b26502e41c91c28934d20efb1f3f` |
| V4 CPU 预算预检 | `b454062372393cb3ed25d3a02f5b401043d2eb261387be73078ddfb364eed2b7` |
| 限定启动配方 | `c9d22260be884fb12f66246546b745598a32d0acecba2d07eda7c4ce9c9a45ad` |

原生窗口中，以 `bash <预检目录>/run_case.sh V0 <新的绝对证据目录> --execute-native` 单独运行。查明外层自然退出后，再决定下一个用例。V4 的两轮应使用不同 PID、独立输出、两轮产物与边界均通过及严格时间顺序，之后才可登记任务相机重启资格。旧 V5 仍沿用已归档的失败状态。
