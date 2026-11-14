# 右手原有效指腹原生结果与跟踪/杆位移分解

源码发布 `f00a0a41e46a6d9cc2abcfeadacf6579690bc10b`，执行时远端/Ubuntu HEAD `b967c0361e34514ecc1bb38e394a361885c49261`。使用已批准的新冻结配方，只运行一次；旧 C5、旧左右手/双手预算未重跑。

## 结果：生命周期通过，接触目标失败

- 1 环境，GPU3，234 SDK 步（2 初始化 + 232 测量）、29 控制窗口、2784 rod 子步；0 PPO 更新。
- 外层自然退出 **0**，原生进程 **154.725142 s**，无 SIGTERM/SIGKILL/SIGSEGV。
- 原生保护未触发，232 条物理记录完整；真实资源释放、SimulationContext 清理、app.close 返回和独立退出审计通过。GDB finalizer 栈/threads/registers/libraries/maps 均留存；没有本批崩溃栈，因为本批没有崩溃。
- 父调度器等候窗口在独立 CPU 事后审计完成前返回 pending；原生进程已在 180 s 内自然退出。随后仅接收已完成回执、恢复发布事务，没有第二次执行或追加物理步。
- **原有效指腹接受数仍为 0**。仅 scoped lifecycle/clock/guarded-contact diagnostic 通过，不能解释为有效抓取、稳定支撑、Hold 或 Teacher 成功。
- GPU3 执行后回到 7 MiB、0% 利用率。其他任务和历史模型未修改。

| 右手指腹 | 实际 solver pair | 原区域通过 | 全段椭圆值 min/max |
|---|---:|---:|---|
| 拇指 finger14_link | 1002 | 0 | 1.038418 / 1.868063 |
| 食指 finger23_link | 1127 | 0 | 1.020704 / 1.068984 |

全部 2129 条的首拒绝原因为 `outside_inner_ellipse`。原接受上限仍为 1，未扩展原 4/3.5 mm 椭圆。首条实际接触在 SDK141；并无“两指同时有效承载”窗口。

## 逐项定位，不把规划预测写成接触

新增 `scripts/audit_dh116_effective_right_tracking.py`，只读取本批冻结 receipt 与 ledger，验证 env/episode/reset/rod、232 连续物理步、policy_step、几何身份及 SHA。并在 SDK234 同一真实状态上比较以下几何重算。

| 状态组合 | 拇指椭圆值 | 食指椭圆值 |
|---|---:|---:|
| 原命令 FK + Reset 直杆 | 0.949921010 | 0.949744007 |
| 实测 root/active q 的完整 FK + Reset 直杆 | 1.005748344 | 1.017642543 |
| 实测 link 位姿 + Reset 直杆 | 1.005689488 | 1.017650011 |
| 实测 link 位姿 + 实际变形杆节点 | 1.038921752 | 1.033872660 |

同边界实测数值：

- 右掌根实际相对目标 XYZ：[0.000485, 0.000698, 0.002447] mm。
- 右主动关节实际减目标（原六关节顺序）：
  [0.001764271, -0.000866522, -0.002381598, 0.000048852, -0.000000501, -0.000000620] rad。
- 整根杆节点相对 Reset 的逐轴最大绝对位移：[0.385381, 0.066969, 0.001304] mm。这是节点位移，不能写成整杆净 Lift 或材料滑移。
- 实测 active q FK 与实际指腹 link 位置差仅 0.00004484/0.00006157 mm。
- 按实测 link 与杆节点重算的接触点，与该指腹最后 solver 子步实际点的误差仅 **0.00001371/0.00002396 mm**，节点身份相同。

这些对照说明：本末步中，命令几何的静态 0.95 余量没有覆盖真实关节跟踪差和杆横移；换入真实状态即可复现落在区域外的接触。没有证据支持通过修改坐标映射、扩展指腹区域或放宽保护来修复。本分析是同边界几何分解，**不是 PPO 学习因果实验，也不能声称已分离每项动态因果贡献**。

计划里的 unit normal weight 仅用于几何查询，始终不作为测得载荷。原生实际力和 raw pair 留在真实 ledger。Contact depth、固定材料锚 Slip、正向有效支撑及教师资格仍未通过。

## 代码、测试与证据

新增：
- `scripts/audit_dh116_effective_right_tracking.py`：不可变来源、完整时钟、区域计数、逐命名关节、米到毫米、四组状态几何对照、实际 solver 接触点核验。
- `tests/test_effective_right_tracking_replay.py`：先缺模块失败；覆盖实际冻结数据、关节顺序、单位、时间身份篡改、SHA 拒绝和同节点预测/实测点一致性。
- 本报告；Registry/readiness/README/CHANGELOG 经工作包发布更新。

CPU：本分析相关 **40 passed、0 failed、0 skipped，18.58 s**。原生入口准备 60 项；原生发布回归 16 项；这些是不同测试命令的结果，不相加冒充不重复测试总量。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/audit_dh116_effective_right_tracking.py --output /absolute/private/new-result.json
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_effective_right_tracking_replay.py tests/test_deformable_pad_region.py tests/test_effective_right_pinch_geometry.py tests/test_surface_contract.py
```

冻结标识：

- 执行 entry 代码/配置 SHA：`3c8ef52ef77a14821b392ad12aeb7f816dc3211d9a74e5d75d97ee24b89311b1`。
- scheduler code/config SHA：`ad37bcb60d1b8a68b6c43b38202144abc0ce04cd80615648d85c21a379cf0205`。
- command spec SHA：`b84801392b897b1fff89bdd74190b2232716707c8830667fef11a729a84958e1`。
- receipt SHA：`9525c46f257fb6d5d81f585f71b47c571dcff7a05019b583e3e93c7535ed0bd8`。
- ledger SHA：`072fe9cf5212abdada92c09a0cd2a04a2032a12b98f437d903565c22eceadc5d`。
- 事后重算结果 SHA：`821b15b72c403d1008501fd26a41087825db062b7260c39387f862c31f29d2a9`。

原始数据位于 Ubuntu 私有 `approved-effective-right-20261114-gkv849uy` 与 scheduler evidence；不上传原始模型/账本。所有追加数据为本次真实单原生实例，重算报告明确标为 CPU replay。

## 阻断与下一包

1. 右手需要在原区域、原碰撞与关节限值内，针对实测跟踪差及杆横移构造有余量的目标/接近控制候选；先离线检查全路径，不把当前 1.53 µm 静态非目标余量当成鲁棒安全。
2. 左手原区域目标、Tripod 与 Power 独立可达性仍各自推进，不能以右手生命周期代替其资格。
3. Pinch 有效对向接触、材料锚 Slip、完整状态与 R0 仍未齐备。Teacher PPO 未启动；MPC BC 不是所有 PPO 的永久前置，但真实任务来源合同不能绕过。
4. 原生批次已消耗。后续若有代码/配置实质变化并通过 CPU 回归，按用户 standing approval 冻结新的独立有界指纹；不重跑本配方、不自动扩成长训练。
5. WireSeg 保留既有独立诊断结果，本批未增加训练步数。30 mm / 5 s / 0.5 mm、528 保护文件、两份历史 Checkpoint 与旧控制器不变。PR #2 保持 Draft。
