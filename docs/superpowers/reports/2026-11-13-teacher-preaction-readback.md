# Teacher 371D 前动作来源适配与历史缺口审计

本包从 07db9eb461f8db067beb294c12c077d1ea1b824e 开始，只在 Ubuntu 独立 worktree 开发、CPU 测试和读取已有文件。新增物理步 **0**，optimizer updates **0**；没有启动 Isaac、GPU、Teacher 推理或训练，没有下载 WireSeg 数据，没有改原模型、checkpoint、物理 worker、371D 归一化或受保护源码。Pinch/Tripod/Power 配置的 native readiness 不变。

新增 tabletop_teacher_preaction_readback.py 是独立来源适配器，尚未接入现有环境主循环。capture_named_hand() 在两次相同共享时钟读取之间复制真实 joint_names、hand.data.joint_pos/joint_vel 和 get_dof_types()；只接受当前固定基座 DH116 的 11 个已命名关节，且实际 SDK DofType 必须全部为整数 0（旋转）。target/cache、猜测单位和缺失顺序不能替代实测状态。

调用方必须先执行本轮 SDK/Articulation 更新，并在任何 target write 前调用适配器；两次时钟检查只能发现身份变化，不能独立证明调用方已刷新 SDK 缓存。此 CPU 包没有验证新原生读取路径，也不将来源字符串当成原生执行证明。

## 精确索引与有效性门

既有 docs/contracts/stereo_privileged_state_contract.json SHA256：
9f145cf8e68b1dfe8e2c7464ec5fc0cb00e81a2707115c2fe148ab49d77bbb62。
linear_coupling.py SHA256：
168e30634c2e240fbaecd225766b52a1f2a3c87afafc8c2a7b5a23dfcb1afe93。
半开索引使用原来的 left/right 与 ACTIVE/MIMIC 顺序，不使用 SDK 数组位置猜名：

| 通道 | 左手 | 右手 | 原有归一化 |
| --- | --- | --- | --- |
| active joint position | [0,6) | [6,12) | 2*(q-low)/(high-low)-1 |
| active joint velocity | [12,18) | [18,24) | qd/0.3 |
| passive mimic residual | [78,83) | [83,88) | (q_follower-ratio*q_master-offset)/0.02 |

adapt_named_hands() 接受带完整 run/env/episode/reset/rod/policy/physics/timestamp 身份、显式 pre_action_before_target_write 阶段、实际关节顺序及 DOF 类型来源的 packet。它只填上述通道，不截断超范围实测值。单手最多 17/371，双手最多 34/371；这是 CPU fixture 验证的适配能力，**不是新原生有效字段数量**。

剩余字段保存为 NaN、valid=false、source=null。双手 fixture 的未知区间为 [24,78) 和 [88,371)，共 337 项。require_teacher_common() 在输入 unmasked Teacher 前拒绝未知、无来源、伪造 valid mask 或有限零占位值；本版本任何适配结果都不能通过完整 371D 门。补齐来源需要新的明确实现，不能靠调用者置位。现有 Teacher.pack() 未被改写；此门只约束新增适配器的导出路径，不能声称已覆盖所有旧入口。

## 已有真实 Pinch 文件的只读结果

审计的是保存下来的 **1 行** Pinch 账本，并非 238 步候选成功结果。该行顶层 SDK step=3、rod tick=12；pre_state 在该步前复制，前动作 SDK step=2 是依据 producer 语句顺序推导的审计值，不是账本原存的完整共享身份。杆节点、速度、节点合力与 mesh wrench 在 step=3 后读取，因此不能直接与 pre_state 拼为同钟 Teacher 输入。

原 producer 验证 set(hand.joint_names)==set(JOINTS)，但没有把实际关节名顺序写入 row 或 plan。11 元素 q/qd 数组本身不能恢复该顺序。也没有右手状态、实测双腕及 reset reference、完整 controller/history、前一完整 policy interval 或原刚体 object-center/quaternion/cylinder 几何语义。不能以柔性杆节点平均值或固定基座常数填这些字段。

DeformX 节点合力与 mesh wrench 可保留用于独立力反馈审计，但不能填逐接触 partner、法向载荷、接触点、桌面接触缺失或材料 anchor slip。旧刚体数据的 204/208 字段来源证明不适用于此柔性 Pinch 行。

离线结果：**0/371 admitted，全部 371 索引未知，qualified_for_teacher_ppo=false**。CLI 保留每一行前/后动作时钟、72 个合同 block 的完整索引/归一化/缺失原因，以及原文件和当前实现 SHA；不改写原始文件，不接纳手工添加的虚构字段。重复输出路径会失败。

私有证据：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-preaction-source-20261113/inventory-final-source.json

- 最终源码 inventory SHA256：c834e71548a43f841ea6ef1d74699c7818efa6d09bea7590071c717a6360b370
- 首次 inventory.json 保留，SHA256：936d1e1ad6ce9ae684671ccc62e202de327995aad678eb8239d1d533c0e432a2；仅清理源码末尾空行后另写最终源码审计，没有新增物理步
- 原 physics.jsonl SHA256：0c8062301e82b3127b0c49aadc5fdfd0096310f27d00a1f9c9a2f552216a20d1
- 原 plan.json SHA256：8f803f05c04d946e87cf28416ce478bb8f7b740293a76e244c83cf9bf4e98efd

原生输入目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-c5-pinch-20261112-v9l5ajv3/native-runs/flex.f4-free-pinch-left-candidate-native/attempt-crp1pahm/native/。
两输入在读取字节后立即核验完整 SHA，输出只绑定这些已读字节。plan 中 producer SHA 仅标作历史声明，不等同于本次原生执行。

## 验证

先新增测试并确认缺模块失败；随后针对 DOF 类型与独占证据输出新增失败测试，再实现对应检查。新增 **35** 项 CPU 测试覆盖乱序名称、左右手精确索引、mimic 顺序、命令冒充实测、聚合力冒充来源、NaN、错误单位、未知/非角 DOF、跨 run/reset/episode/rod/physics 身份、前后动作错位、读取中时钟变化、数据别名、缺失/伪造 mask、原文件 SHA 和禁止覆盖。

最终相关回归 **67 passed / 4.80s**，退出码 0。曾因测试文件名写成不存在的 test_dh116_joint_target_readback.py 而收集失败，定位为命令路径错误后改用实际文件 tests/test_joint_target_readback.py；没有绕过测试。最终命令：

~~~sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dh116_teacher_preaction_readback.py \
  tests/test_dh116_deformable_force_evidence.py \
  tests/test_joint_target_readback.py \
  tests/test_dh116_deformable_privileged_teacher.py \
  tests/test_dh116_deformable_privileged_ppo.py
~~~

只读审计命令（新输出必须不存在；下方原目录与新的证据文件为路径占位）：

~~~sh
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python scripts/audit_dh116_teacher_preaction_sources.py \
  --ledger <原目录>/physics.jsonl \
  --ledger-sha256 0c8062301e82b3127b0c49aadc5fdfd0096310f27d00a1f9c9a2f552216a20d1 \
  --plan <原目录>/plan.json \
  --plan-sha256 8f803f05c04d946e87cf28416ce478bb8f7b740293a76e244c83cf9bf4e98efd \
  --output <新的私有证据文件>.json
~~~

下一阶段需在已批准的新采集中保存实际 joint_names/DOF 类型与共享前动作时钟，取得双手状态，明确 flexible rod 与原刚体 common 字段语义，再逐项接入 controller/reset/interval 和逐接触来源。本包不申请或执行额外训练；完整 371D/Reset/R0 与原生预算门仍分别阻断 PPO。
