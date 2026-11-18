# Teacher 运行环境与 RGB 读取边界

## 服务器核验

开发环境解释器为 `/mnt/mydisk/xk/envs/dh116/bin/python`，Python 3.10.12、PyTorch 2.7.0+cu128、NumPy 1.26.4、Isaac Sim 4.5.0、RSL-RL 3.1.3。服务器有四张 RTX 4090，SM89。PyTorch 的 CUDA 构建版本为 12.8。

安装的 Replicator 扩展依赖 `omni.warp`。本机 `omni.warp-1.5.0/config/extension.toml` 精确依赖 `omni.warp.core` 1.5.0。上次首次 Pinch 原生运行的映射列出该扩展的 `warp.so`。同一 Python 环境的 pip 元数据列出 Warp 1.12.1，实际 Kit 读取依赖由运行时回执记录。DH116 控制源码目录无已跟踪的 `.cu` 或 `.ptx`。IsaacLab 与 Isaac Sim 已安装的 `SimulationContext` 源无 `shutdown()`，现有清理 API 继续用于该诊断。

## 已实施

- 固定双 RGB 记录器在读取首帧之前保存 `warp_rgb_abi.json`，包含 Warp 的实际模块路径与 SHA、constructor 签名、当前 Replicator 指针函数所引用的 Warp 身份、扩展配置和匹配策略。源和 ABI 校验由预检与实际读取复用。
- 录制器复用已有 `ManagedPreviewApp`，在 App关闭时解除该 launcher 持有的 PLAY/STOP 订阅。复用弱引用记录两只手的 Articulation、ArticulationData、根 Tensor View 和固定相机 reader。App 关闭返回后保存一份单独的映射和存活状态。外层进程退出码仍是自然退出依据。
- 兼容性证据支持保留已核验 Warp 的两条读取路径。当前无证据表明共享 `ogn_generated` 缓存导致报错。保留既有物理初始化步数与旧训练代码。

## 无相机 Teacher 准入

Pinch、Tripod、Power 分别在 `approach_wiring_smoke_v1` 与 `flexible_train_r1_full` 运行 CPU 预检。已建立六份独立配置，绑定各自的现有奖励配置 SHA 和 PPO gamma。六份均通过后端来源和 reward 绑定。六份均未获得原生准入。

| 抓法 | 冒烟版剩余门槛 | 完整版剩余门槛数 |
|---|---|---:|
| Pinch | 当前 Reset 证明、独立原生预算、旧 R08 builder 来源 SHA 不同 | 12 |
| Tripod | 当前 Reset 证明、独立原生预算、三指距离 reader | 12 |
| Power | 当前 Reset 证明、独立原生预算、握式距离 reader | 12 |

R08 封存的 builder 对应提交 `596d7a9dd932b1ae07290bb38302c0b9334cbc03`。与当前文件相比，差异在构建时新增了 `damping_override` 参数及传递。仍需按当前构建链取得真实 Reset 回执。无相机 Teacher 可独立处理该门槛。完整物理成功另需真实接触与既有成功判据。

## 验证

CPU 联合回归 **618 通过，0 失败**。本工作包使用已安装 SDK 路径验证 Warp 的原始非拥有 CPU 指针读取。新增原生运行、RGB、动作及 PPO 均为 **0**。上次 `app.close()` 返回后的 `-11` 仍待原生复核。配置、环境信息和预检原始结果留在服务器证据目录。
