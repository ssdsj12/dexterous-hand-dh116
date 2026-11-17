# 右手 Pinch 原有效指腹路径：独立原生执行准备

基线 GitHub/Ubuntu `70def33020c2a0a7c2e26c1e5ddf5ae6d0b11705`。仅新增隔离执行入口，不修改旧左/右/双手配方、原保护或历史模型。此报告为 CPU 与执行准备，**尚无本配方原生结果、PPO 更新或 Teacher 资格**。

## 已知物理问题及本次范围

此前同一 episode 的真实双手账本有 2111 个 hand-pair，但全部位于原有效指腹椭圆之外。已有 solver 碰撞不等于有效抓取。本次只验证独立右手的原指腹区域接近；左手保持原初态、零请求，不把两次单手运行拼成双手成功。

复用已发布右手 CPU 目标与路径：
- 目标 SHA256 `cc6fcfcbf001d597fcc46d0a27768974f8e0f4b1714a872074ac8a6a65d8ddbe`。
- 路径 SHA256 `03b70ba97423665eac710ddd0fe3e868282885428efe1e0775bf304ccf9aa280`。
- 原源网格、cooked hull、原 4/3.5 mm 指腹半轴与区域判据保持不变。
- 独立右手初态相对旧右手：平移约 [+0.302384,+2.000000,+0.750668] mm；World rotvec [-0.03,+0.00203393,+0.00846376] rad；主动关节差 [-0.00359877,-0.00041753,+0.00145040,0,0,0] rad。仅此独立配置，不覆盖旧环境。
- 接近过程中原手指速率不变，右腕下移 0.5 mm，横向 100 µm 分段绕行；原 0.01 m/s 腕部限速保持。
- 1025 个 CPU 路径采样及最大分段速度区间界：桌面最低净空 2.106441 mm；非目标碰撞最小证明余量仅 **1.528402 µm**；要求指腹最大进入深度 0.290936 mm。
- 此极小余量不能证明实际跟踪或移动杆响应安全，原生逐步保护继续判断，失败即停，不扩大范围。

## 实际改动

1. `prepare_dh116_effective_right_pinch.py` 从锁定 CPU 结果编译新 Reset、29×24D 动作和逐物理步边界；左手、自由杆和原速度限制不变。
2. `tabletop_effective_right_pinch_geometry.py` 复用原实际 source/cooked hull、桌面、非目标、SDF、mimic/joint 跟踪保护。仅用严格右手非零请求/投影/提交一致性检查替换旧“腕部全零”实验假设，未关闭原安全投影。
3. 每个 480 Hz 边界保存原 clock/episode、所有手指 solver pair 的 12 子步身份、载荷、实际 link 变换与原有效指腹区域判定。原始接触账本保留；接触深度、material slip、有效支撑及 Teacher 资格仍为未通过。
4. 独立 prepare/verify/validator/scheduler 入口复用已通过的参数读取和资源释放所有者。post-exit 检查、真实父 receipt、GDB、maps 和外层自然退出保持。
5. 发布配方将 USD 等受保护资源列为带 SHA 的绝对只读输入，而非可提交文件。调度器发布白名单未修改。

## 失败先行与修复

日志位于 Ubuntu 私有证据目录 `right-effective-native-20261114`，不上传完整原始账本。

- 初始 plan/geometry 缺模块测试失败留存。
- 原 CPU owner 夹具在 SDK102 触发 source:finger13_link 净空证明失败 -1.1631 µm。逐项核对实际 q、杆和 FK：杆误差 7.15e-9 m；link 位置误差恰为 [-0.0875,0,+0.1625] mm，与该步腕部请求相反。
- 根因是旧 CPU 夹具只刷新手指 FK，未使用实际 wrist_x/y/z/roll/pitch/yaw。补齐原内禀 XYZ carriage FK，未修改生产控制或保护判据。
- 增加全部 232 边界双侧 base_link 与真实请求积分的 1e-7 m 对齐断言，避免仅以末帧/峰值通过。
- 调度器先拒绝 USD 进入 publishable source_files；修复为只读 inputs。
- 临时子进程库随后缺旧右手配置，独立外层审计正确拒绝。补齐真实配置及代码指纹覆盖，而非绕过拒绝。

## 验证

**60 项 CPU 测试通过，0 失败，0 跳过，178.97 s**。完整日志 SHA256 `43e96fd8f1929e26fdfb8f3ee3dc90cd05d94f106e83ef971681c86af724171a`。冻结执行代码/配置 SHA256 `3c8ef52ef77a14821b392ad12aeb7f816dc3211d9a74e5d75d97ee24b89311b1`；原生预算配方与所有本地/外部源另由 scheduler 组合指纹绑定。

命令：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_effective_right_pinch_plan.py tests/test_effective_right_pinch_geometry.py tests/test_effective_right_pinch_worker.py tests/test_effective_right_pinch_owner.py tests/test_dh116_pinch_segment_geometry.py tests/test_deformable_pad_region.py tests/test_surface_contract.py
```

CPU owner 使用显式合成 SDK/solver double，只验证真实 Python 执行、物理步计数、数据来源、终止与生命周期路由，不能计入原生验证次数。

## 新预算与准入

- GPU3、单 Isaac、1 环境、0 PPO；最多一次。
- 234 个 SDK 步：2 初始化 + 48 静止 + 160 接近 + 24 保持；232 测量、29 个 60 Hz 控制窗口、2784 个 rod 子步。
- 墙钟上限 180 s；不重试、不追加预算。原生超时不得强制退出后伪造成功。
- 使用真实已通过 parameter-task 父 receipt `f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901`，不伪造输出路径。
- 用户“批准，后续自动批准”的既定有界范围覆盖；新冻结指纹仍写入独立私有批准文件并经 scheduler 校验。
- 原生尚未执行。下一包执行后独立报告实际接触区域、跟踪与保护、自然退出及 GDB。任何结果均不能把 234 步当作 5 秒 Hold。
- Pinch/Tripod/Power Teacher、正式 BC/DAgger/Student 均未因本 CPU 包取得资格；30 mm/5 s/0.5 mm 不变，R2 关闭。PR #2 保持 Draft。
