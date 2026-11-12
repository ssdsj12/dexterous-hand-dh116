## 2026-11-12 — 10 个原生凸包 GPU SDF 烘焙与显式注册

10 个实际凸包使用 1 mm 体素烘焙并在同一 RopeBatch 注册；未推进物理或验证接触力。 Ubuntu CPU 回归 17 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-cooked-pad-sdf-registration.md)

## 2026-11-12 — F6 渲染自然退出崩溃 C++ 回溯

冻结 drop 入口的 GDB 首次 SIGSEGV 栈指向 AppWindow Python 绑定与 Carb，独立于 F3 PhysX Tensor。 Ubuntu CPU 回归 3 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-render-appwindow-backtrace.md)

## 2026-11-12 — 第二个 WireSeg 深度分片和实测速率

Ubuntu 完整 blob 现为 2/148。第二个 22.97 MB 官方分片在 Windows 用时 224.78 秒，正式预训练继续待完整数据。 Ubuntu CPU 回归 7 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f5-wireseg-proxy-transport.md)

## 2026-11-12 — 实际 PhysX CPU/GPU 凸包逐指配对

两次独立原生退出 0，全部 10 个实际指腹有向多边形一致、顶点最大差值 0 m。 Ubuntu CPU 回归 15 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-cooked-hull-device-parity.md)

## 2026-11-12 — F6 drop 原生渲染与关闭诊断

Drop 已产出 RGB、径向深度、Native ID 和 composed USD，外层退出仍为 139，hang 待实际验证。 Ubuntu CPU 回归 14 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-render-smoke-f6.md)

## 2026-11-12 — 原生 PhysX 读取 10 个 DH116 指腹凸包

实际 PhysX CPU 读回全部 10 个指腹的 cooked convex hull，外层自然退出 0。GPU 烘焙一致性仍待验证。 原生入口的生命周期调用与 F3 新入口分离，修复集成导入回归。 Ubuntu CPU 回归 13 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-cooked-hull-native.md)

## 2026-11-12 — WireSeg 分块续传与首个完整深度分片

Windows 续传 6 块，Ubuntu 完整校验 22,541,803 字节官方分片并读取 360 个深度数组，未增加训练更新。 Ubuntu CPU 回归 40 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f5-wireseg-resumable-relay.md)

## 2026-11-12 — F3 DeformX 与 PhysX 完整桥接自然关闭

A–E 均自然退出 0，两轮独立完整桥接各 240 步且与旧物理账本逐字节一致。原关闭路径的单一触发项尚未隔离。 Ubuntu CPU 回归 43 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f3-teardown-isolation.md)

## 2026-11-12 — 官方 drop/hang DomeLight 派生修复

四个派生 USD 只更新 DomeLight 纹理。76 个 USD 复审无未解析依赖，原始 109 个文件哈希一致。 Ubuntu CPU 回归 11 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-assets-f6.md)

## 2026-11-12 — 实际指腹烘焙设置和 PhysX 凸包读回入口

独立核对 10 个实际 USD 指腹的 cooking schema、坐标和离线凸包估计，确认已安装 PhysX 查询入口。 Ubuntu CPU 回归 11 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-cooked-hull-audit.md)

## 2026-11-12 — 单 PhysX 刚体双向传力及关闭失败诊断

240 个原生步核对单刚体力和力矩交换。物理证据可复现，PhysX 插件在 Python 终结时报 SIGSEGV，F3 自然退出仍 blocked。 Ubuntu CPU 回归 40 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f3-rigid-bridge-native.md)

## 2026-11-12 — 将默认开发目标切换为 DeformX 柔性线缆

默认调度覆盖柔性 F0–F10 和离线依赖。原 solid_cylinder 仍可独立回归。 Ubuntu CPU 回归 28 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-execution-target.md)

## 2026-11-12 — DH116 实际指腹碰撞几何预检

从实际 USD 解析双手 10 个闭合指腹碰撞网格，复现上游跳过 CollisionAPI 所在 Xform 的问题。PhysX cooked hull 和动态加载仍待验证。 Ubuntu CPU 回归 8 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-dh116-colliders-f4.md)

## 2026-11-12 — 官方 DeformX drop/hang 资产和轨迹审计

Ubuntu 审计 109 个原始文件、72 个线缆 USD、4 个场景与 6 份轨迹；原生渲染待独立验证。 Ubuntu CPU 回归 11 项通过。[报告](docs/superpowers/reports/2026-11-12-deformx-assets-f6.md)

## 2026-11-12 — Standalone CUDA 静态、动力学与接触验证

RTX 4090 独立求解器 36 项通过，保留 1 项上游高刚度限制 xfail。静态平台读回双侧力和力矩。 Ubuntu CPU 回归 12 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f2-standalone-gpu.md)

## 2026-11-12 — 隔离 CUDA 12.8 与 cp310/SM89 扩展

固定 cp310 和 CUDA 12.8.61 两次构建成功，实际产物均有 SM89 设备代码。 Ubuntu CPU 回归 18 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f1-cuda-build.md)

## 2026-11-12 — DeformX 主仓库与递归源码锁定

柔性主线锁定上游主仓库与 CPU/CUDA 两个精确子模块，原刚性环境作为独立回归。 Ubuntu CPU 回归 9 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f0-deformx-source-lock.md)

## 2026-11-12 — WireSeg-36K 数据接入与单目诊断

柔性主线 F5 的数据读取及共享视觉骨干诊断已验证。现有证据为 4 张样本、2 对深度与 1 次 CPU 分割更新。 Ubuntu CPU 回归 33 项通过。[报告](docs/superpowers/reports/2026-11-12-flexible-f5-wireseg-ingestion.md)

## 2026-11-12 — 握式局部可达性有限差分复查

原碰撞验证后两手最大指腹间距改善至左 1.899、右 0.950 mm。12 个候选均未通过完整握式筛选。 Ubuntu CPU 回归 219 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-wrap-local-fd.md)

## 2026-11-12 — 原生 T5 碰撞实例与 Wire SDK 来源核验

核验 76 个启用碰撞体的身份与 256 条接触记录。该 Python SDK 写入路径未观测到调用，完整回合力学资格仍待验证。 Ubuntu CPU 回归 99 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-t5-proxy-sdk-native-results.md)

## 2026-11-12 — 二指三指独立首终止原生配对

二指退出后继续观测三指。相同实测冷初态下，冻结 Teacher 的 sampled 和 mean 均出现右 finger12 合法目标下位置越限。 Ubuntu CPU 回归 46 项通过。[报告](docs/superpowers/reports/2026-11-12-finger12-per-slot-tripod-native.md)

## 2026-11-12 — 握式多腕姿原几何补充搜索

原有边界下每手 16 个不同起点，最接近的两手最大指腹间距仍高于原筛选线，握式保持待有效初态。 Ubuntu CPU 回归 219 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-wrap-diverse-wrist-search.md)

## 2026-11-12 — 握式 Teacher 整手包握几何预检

握式按双手各五指包握预检，8 个历史候选和有界 CPU 搜索未得到可用初态。 Ubuntu CPU 回归 234 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-wrap-geometry-preflight.md)

## 2026-11-12 — T5 原生 PPO Teacher 的对象侧接触验证

实际 R0 update-1 在 T5 的短窗完成 128 个物理步，双侧力闭合，端支架均持续承载。 Ubuntu CPU 回归 19 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-t5-wire-tensor-native-results.md)

## 2026-11-12 — 冻结 Teacher 同初态原生 sampled／mean 复验

修复原生合同编码与回执。四次运行均退出 0，采样和 mean 均复现合法目标下的同一右 finger12 越限。 Ubuntu CPU 回归 56 项通过。[报告](docs/superpowers/reports/2026-11-12-finger12-frozen-teacher-native-v04.md)

## 2026-11-12 — 原 GPU 任务 Wire 对象侧接触与支撑读回

原 GPU 路径完成 128 个连续物理步，双侧接触力互相抵消。短窗的两个端支架持续承载，完整回合支撑资格仍未通过。 Ubuntu CPU 回归 80 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-wire-tensor-native-results.md)

## 2026-11-12 — 原 T5 合同的特权 PPO Teacher 原生单更新

2×32 原生完成 1 次 R0 PPO 更新，Checkpoint 与 Adam 读回一致，完整成功回合 0。 Ubuntu CPU 回归 65 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-t5-native-r0-update1.md)

## 2026-11-12 — 记录 GPU 任务早期接触读回失败并恢复默认探针

早期启用报告读回导致 GPU reset 索引与缓冲区设备不一致。保留失败证据并恢复原探针默认。 Ubuntu CPU 回归 74 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-task-contact-early-readback-failure.md)

## 2026-11-12 — 原 T5 MPC 八个首回合原生采集

原生二指 4/4、三指 0/4，10,559 条状态／动作配对，全部首回合已分区。正式 BC 准入为 0。 Ubuntu CPU 回归 54 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-t5-first-episode-native-evidence.md)

## 2026-11-12 — MPC 成功轨迹进入 Teacher 的原生证据预检

只读入口实测读取三类原生日志，分开记录任务成功、正式证据与专家装配资格。 Ubuntu CPU 回归 53 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-teacher-provenance-preflight.md)

## 2026-11-12 — MPC 内部控制状态与 reset 初始化来源

新增隔离的同钟内部状态来源旁路。真实 T5 输入预检通过，实际字段资格待新的原生采集。 Ubuntu CPU 回归 54 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-internal-state-source-preflight.md)

## 2026-11-12 — MPC 371D 原生同步账本持久化

真实旧 64 行回放逐位恢复 371D、24D 和掩码。新版可选账本供未来执行，现有运行沿用冻结源码。 Ubuntu CPU 回归 58 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-durable-371d-transition-journal.md)

## 2026-11-12 — 将 Teacher R0 绑定到原 T5 原生物理合同

完整 T5 的实际张量合同与历史成功 T5 相同。新增只读派生与原生回调严格身份检查。 Ubuntu CPU 回归 85 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-t5-native-contract-binding.md)

## 2026-11-12 — MPC 完整上一动作窗口的腕部字段校验

62 条真实非 reset 行的 4 个腕部窗口字段逐值吻合，reset 行保留未测窗口。 Ubuntu CPU 回归 50 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-native-observation-value-proof.md)

## 2026-11-12 — 冻结 Teacher mean 配对的 sampled 完成证据

mean 启动前校验 sampled 回执、原生账本和动作身份。旧 Teacher Checkpoint 与历史 T5 场景仍不可互换。 Ubuntu CPU 回归 54 项通过。[报告](docs/superpowers/reports/2026-11-12-finger12-frozen-teacher-pair-runner.md)

## 2026-11-12 — MPC 371D 同钟数值与来源校验

旧短程原生的 64 行均实证 204/371 个字段，新增独立旁路，保留 167 个未资格字段。 Ubuntu CPU 回归 31 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-native-observation-value-proof.md)

## 2026-11-12 — Teacher 原生回调产物持久化

训练、采集和冻结 Teacher 评估在应用关闭前写入并同步回执。外层退出仍以独立监督记录为准。 Ubuntu CPU 回归 82 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-callback-durable-completion.md)

## 2026-11-12 — Teacher 原生成功专家数据装配器

装配器逐行核验完整 371D、实际 issued 24D 和原生证据。当前未产生合格 BC 专家包。 Ubuntu CPU 回归 13 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-expert-assembly.md)

## 2026-11-12 — 现有刚性 Teacher 任务逐接触读回探针

晚于首次 reset 的探针实测记录 256 个物理步，结构化报告仍为空。新版启用时点前移，待单实例原生复验。 Ubuntu CPU 回归 71 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-task-contact-probe.md)

## 2026-11-12 — Teacher Actor-only BC、R0 与冻结验收入口

新增 SHA 绑定的 Actor-only BC、R0 恢复与冻结验收预检，短程 MPC 归档通过拒绝测试。 Ubuntu CPU 回归 75 项通过。[报告](docs/superpowers/reports/2026-11-12-teacher-bc-r0-frozen-chain.md)

## 2026-11-12 — 冻结 Teacher sampled／mean 同初态配对入口

真实 update-0 与同一双抓法初态通过 CPU 预检。旧 MPC reset family 和 native readback 不匹配，三路资格仍待验证。 Ubuntu CPU 回归 53 项通过。[报告](docs/superpowers/reports/2026-11-12-finger12-frozen-teacher-pair-runner.md)

## 2026-11-12 — 原 T5 MPC 完整首回合采集预检

真实 T5 参数及 10 个来源 SHA 预检通过，归档按完整成功、失败与未结束回合分流。完整原生预算待独立批准。 Ubuntu CPU 回归 37 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-t5-first-episode-capture-preflight.md)

## 2026-11-12 — PhysX GPU 接触报告读回与左右拇指标定

原生左右隔离标定分别读出 37、43 个可识别接触 pair headers。任务逐指、外部承托及附着资格仍需单独验证。 Ubuntu CPU 回归 67 项通过。[报告](docs/superpowers/reports/2026-11-12-physx-gpu-contact-report-readback.md)

## 2026-11-12 — Cold8 状态标签与独立深度持出诊断

两种抓法各有 12 条 Center XYZ 和 12 条线速度监督；独立持出证据仍不足以授予动态泛化与双目精度资格。 Ubuntu CPU 回归 55 项通过。[报告](docs/superpowers/reports/2026-11-12-stereo-cold8-state-depth-diagnostic.md)

## 2026-11-12 — 右手 finger12 同初态三路配对预检

旧 Teacher 与新 Cold8 的实测初态交集为 0。预检记录了 MPC 近邻状态、测量缺口与三路配对预算。 Ubuntu CPU 回归 41 项通过。[报告](docs/superpowers/reports/2026-11-12-finger12-native-pairing-preflight.md)

## 2026-11-12 — 原 MPC 371D／24D 短程原生同步诊断

修复同钟记录器的 dtype 错误。原 MPC 复验取得 64 条状态／动作记录和 512 条环境级物理样本，完整成功回合为 0。 Ubuntu CPU 回归 30 项通过。[报告](docs/superpowers/reports/2026-11-12-mpc-sync-v02-native-evidence.md)

## 2026-11-12 — 8 初态双目同步采集

二指、三指各完成 4 个独立物理初态，合计 32 条同钟双目与状态动作候选样本。正式训练资格待独立验证。 Ubuntu CPU 回归 63 项通过。[报告](docs/superpowers/reports/2026-11-12-stereo-cold8-native-capture.md)

## 2026-11-12 — PhysX v3 原生自然退出与全新实例重启

原生两轮均完整记录 240 个 480 Hz 物理步，并自然退出码 0。该结果仅授予关闭与重启资格。 Ubuntu CPU 回归 65 项通过。[报告](docs/superpowers/reports/2026-11-12-physx-v3-native-shutdown-pass.md)

## 2026-11-12 — 复用双目深度辅助头的有界监督优化

71 项 Ubuntu CPU 回归通过。原生资格依赖独立证据。[报告](docs/superpowers/reports/2026-11-12-stereo-depth-auxiliary-optimizer-diagnostic.md)

## 2026-11-12 — 双目分割按已验证的 Head 能力构建候选

68 项 Ubuntu CPU 回归通过。原生资格依赖独立证据。[报告](docs/superpowers/reports/2026-11-12-stereo-segmentation-mask-policy.md)

## 2026-11-12 — 现有 solid_cylinder 同钟 SDK 线速度监督

72 项 Ubuntu CPU 回归通过。原生资格依赖独立证据。[报告](docs/superpowers/reports/2026-11-12-rigid-sdk-state-supervision.md)

## 2026-11-12 — Teacher 右手 finger12 下限事件窗口

29 项 Ubuntu CPU 回归通过。原生资格依赖独立证据。[报告](docs/superpowers/reports/2026-11-12-right-finger12-event-windows.md)

## 2026-11-12 — 现有刚性任务的密集和稀疏双目指标

56 项 Ubuntu CPU 回归通过。原生资格依赖独立证据。[报告](docs/superpowers/reports/2026-11-12-stereo-reconstruction-metric-evidence.md)

## 2026-11-12 — 现有刚性任务完整闭环（复用原环境，不重新建模）

默认调度目标选择现有 `solid_cylinder` 闭环，柔性后端延期。27 项 Ubuntu CPU 回归通过。[报告](docs/superpowers/reports/2026-11-12-existing-solid-cylinder-execution-scope.md)

## 2026-11-12 — PhysX 标定句柄释放与退出崩溃诊断

v2 首轮 240 步完整，close 返回后子进程 SIGSEGV。隔离入口补齐句柄释放和崩溃诊断，65 项 CPU 回归通过。[软件报告](docs/superpowers/reports/2026-11-12-physx-sampling-handle-lifetime.md) · [原生证据](docs/superpowers/reports/2026-11-12-physx-v2-native-exit-crash.md)

## 2026-11-12 — PhysX 标定关闭模式与子进程读回

首轮 240 步完整，关闭返回标记缺失。隔离入口修复并通过 61 项 CPU 回归，原生资格待复验。[报告](docs/superpowers/reports/2026-11-12-physx-non-fast-shutdown-recovery.md)

## 2026-11-12 — Ubuntu 提交与 Windows 发布中转

Ubuntu 调度器只保存待发布提交，Windows 推送并读回后登记清单。22 项 CPU 回归通过。[报告](docs/superpowers/reports/2026-11-12-windows-relay-publication-policy.md)

## 2026-11-11 — 双目 RGB 光轴深度辅助头

Ubuntu 65 项 CPU 回归通过。历史 VGA 复放通过左右 RGB 梯度检查，右图缺少独立标签。[报告](docs/superpowers/reports/2026-11-11-stereo-depth-auxiliary-diagnostic.md)

## 2026-11-11 — 独立排期刚性代理和柔性 Cosserat 验收

刚性代理与柔性 Cosserat 各有原生验收依赖。Ubuntu 19 项 CPU 回归通过。[报告](docs/superpowers/reports/2026-11-11-rigid-deformable-scheduling-gates.md)

## 2026-11-11 — DeformX 固定源码实际载荷预检

CPU `passed`，通过 7 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp5.source-payload.md)。

## 2026-11-11 — 状态监督残差及不确定度完整性

CPU `passed`，通过 18 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp2.metric-integrity.md)。

## 2026-11-11 — 准备 MPC 371D 与实际 24D 同步采集

CPU `passed`，通过 50 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp3.mpc-sync-prepare.md)。

## 2026-11-11 — GitHub 传输失败时保留 Ubuntu 工作包

Ubuntu 调度器在 GitHub 连接异常时保留已测试提交。17 项 CPU 回归通过。[报告](docs/superpowers/reports/2026-11-11-work-package-publication-defer.md)

## 2026-11-11 — 验证 PhysX 逐接触来源与条件方向

CPU `passed`，通过 50 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp4.source-audit.md)。

## 2026-11-11 — 双目候选置信度分组校准审计

CPU `passed`，通过 26 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp1.confidence-offline.md)。

## 2026-11-11 — Student 独立评估证据

Student 冷评估核对实际发出 24D、传感与推理时钟及逐样本任务身份。Ubuntu 92 项 CPU 回归通过，原生验收待资格与审批。[报告](docs/superpowers/reports/2026-11-11-stereo-student-independent-proof.md)

## 2026-11-11 — Ubuntu 状态合同审计

兼容旧 v1 配置的两个已审计字节表示，继续记录实际原始 SHA。集成 CPU 82 项通过，原合同与 Checkpoint 未修改。

## 2026-11-11 — 复核 Student 成功池与自有 rollout

CPU `passed`，通过 22 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp6.cpu-admission.md)。

## 2026-11-11 — 复核 DeformX CUDA 源码资格检查器

CPU `passed`，通过 2 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp5.cpu-source-check.md)。

## 2026-11-11 — 复核 PhysX 接触证据与退出合同

CPU `passed`，通过 41 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp4.cpu-contract.md)。

## 2026-11-11 — 复核 Teacher 动作合同与 PPO 回滚

CPU `passed`，通过 94 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp3.cpu-contract.md)。

## 2026-11-11 — 验证三种 Actor 的独立双目部署

CPU `passed`，通过 47 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp6.deployment-isolation.md)。

## 2026-11-11 — 区分状态监督的离线与原生资格模式

CPU `passed`，通过 49 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp2.mode-gates.md)。

## 2026-11-11 — 保留双目原色采样与候选分池

CPU `passed`，通过 36 项。详细结果见[报告](docs/superpowers/reports/dh116-managed-wp1.sample-retention.md)。

## 2026-11-11 — Ubuntu 工作包调度

Ubuntu 工作区负责开发、CPU 验证与提交。本机只负责 GitHub 中转。调度器和中断恢复验证 14 项通过，原生任务与依赖安装按独立批准门槛执行。[执行计划](docs/superpowers/plans/DH116_AUTONOMOUS_EXECUTION_MASTER_PLAN.md)

## 2026-11-11 — PhysX 复验申请

复查原有关闭验证器的冻结快照、GPU、单实例和输出路径，72 项关闭及训练准入 CPU 回归通过。记录原生申请预算、失败停止条件和逐 contact 资格边界。无新原生实验。[审批前复核](docs/superpowers/reports/2026-11-11-stereo-physx-shutdown-retest.md)

## 2026-11-11 — 双目匹配固定像素与采样归因

增加逐视图三类概率、可见与实际 mask 覆盖统计、联合参考和固定双向代价回归。41 项 CPU 通过，Front/Roll20 各四帧复放完成。候选改善部分覆盖，Roll20 准确点数仍低于原色，缺真实右 ID 的资格保持阻断。[复放报告](docs/superpowers/reports/2026-11-11-stereo-wire-point-traces.md)

## 2026-11-11 — Teacher128 同状态动作复盘

新增同身份 Teacher sample/mean 比较、逐关节首越线及跟踪误差、MPC 371D/24D 配对拒绝证据。94 项 CPU 通过，五次历史位置终止的实际力学原因仍待原生复验。旧 MPC 无可配对 371D，噪声与 BC 策略保持原样。[复盘报告](docs/superpowers/reports/2026-11-11-teacher128-paired-action-replay.md)

## 2026-11-11 — 线缆点来源与时序身份

候选记录新增逐点几何、有效性、来源和分段轨迹旁路。96 项 CPU 通过，两个历史 cohort 各四帧复放一致。原色轮廓 91/89 与软引导 67/65 个正确 Roll20 点均为表面来源，三维中心线时序资格仍待证。[验证报告](docs/superpowers/reports/2026-11-11-stereo-wire-geometry-sources.md)

## 2026-11-11 — 143D 同步证据与分阶段准入

补充同帧参考旁路身份、冷 8 诊断限制、连续中心线 Velocity 门槛和逐阶段训练、误差与有效覆盖报告。136 项 CPU 回归通过。历史 8 行、2 个独立初态上只拟合 6 行 Center XYZ，未建立泛化或不确定度校准。正式模式按预期退出 2。[验证报告](docs/superpowers/reports/2026-11-11-stereo-state-stage-admission.md)

## 2026-11-11 — 正式训练准入复核

42 项 CPU 回归核对已有成功池、DAgger 准入、Student 自有 PPO rollout 及候选隔离。Teacher128 诊断证据留在诊断范围。正式 BC、DAgger 和 Student PPO 继续阻断。[准入复核](docs/superpowers/reports/2026-11-11-stereo-formal-training-admission.md)

## 2026-11-11 — DeformX CUDA 来源复查

新增只读 CUDA 源码清单检查并通过 2 项 CPU 测试。服务器复查缺少完整目标仓库及 CUDA Toolkit 12.8，编译、GPU 求解和双向耦合均待解除阻断。[来源复查](docs/superpowers/reports/2026-11-11-stereo-deformx-source-recheck.md)

## 2026-11-11 — PhysX 自然关闭预检

新增顺序 240 步关停与重启验证、JSONL fsync、回调释放与 SIGTERM 证据。41 项 CPU 测试和服务器冻结副本预检通过。原生复验待批准，完整双手接触资格保持阻断。[执行计划](docs/superpowers/reports/2026-11-11-stereo-physx-shutdown-retest.md)

## 2026-11-11 — 双目线缆对应复放

增加保留完整 RGB 窗口的软引导匹配、逐帧对应和拒绝证据、原始 12 处整数视差固定回归。37 项 CPU 测试通过，Front 和 Roll20 各四帧复放均正常结束。预测 hard mask 的覆盖断层已复现，软引导恢复部分有效点，固定尾部仍失败，真实左右对应待合格右原生 ID。[验证报告](docs/superpowers/reports/2026-11-11-stereo-wire-correspondence.md)

## 2026-11-11 — Teacher128 关节动作审计

有界 Teacher 诊断增加 24D 分布和实际动作、全部关节的位置与速度、控制目标及越限证据。87 项 CPU 回归通过。历史 2,030 个物理子步和 256 次策略转移复核出五次右 `finger12` 实测位置越限，原生成功为 0。新增采集的原生复验和 MPC 的 371D 同步重采待批准。[验证报告](docs/superpowers/reports/2026-11-11-teacher128-joint-limit-audit.md)

## 2026-11-11 — 双目状态监督多初态准入

新增多来源候选账本读取，校验 RGB 深度来源、同帧 371D/24D 身份、参考标签和标定。分组跨文件保持物理初态隔离，Center XYZ 与线缆几何按顺序检查标签并分别启用。99 项 CPU 回归通过。历史归档仅有 2 个物理初态和 6 行中心标签，正式入口按预期阻断，无新模型或原生运行。[验证报告](docs/superpowers/reports/2026-11-11-stereo-state-multicohort.md)

## 2026-11-11 — 按工作包持续发布

将用户对独立工作包的持续提交和推送授权写入 `AGENTS.md`。README 和 readiness 同步记录目标开发分支、远端核验、失败证据保留和 Draft PR 规则。此工作包仅更新发布流程文档与状态记录，下一项原生工作仍是待独立批准的 8 初态同步采集。

CPU 回归 84 项通过，15.07 秒，退出码 0。本次无 Isaac 原生验证、采集或训练。

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_cold_candidate_admission.py tests/test_dh116_stereo_candidate_cold_trials.py tests/test_dh116_stereo_contact_calibration.py tests/test_dh116_stereo_wire_tail_audit.py tests/test_dh116_stereo_state_training_modes.py tests/test_dh116_stereo_wire_segmentation.py tests/test_dh116_teacher_wp3.py
```

## 2026-11-11 — 按明确指令发布WP增量与采集预检

本次按用户“更新到github”指令，将基于 `ae307a5` 的代码、独立配置、测试、处理后指标和预检报告发布到 `codex/dh116-stereo-privileged-dagger`，保留 Draft PR #2，不合并。新的8初态同步采集仍未获批准，本次发布不启动Isaac或训练。

发布前复验7份新增核心测试：**84 passed，11.56s，exit0**。1183份源码/脚本/配置/测试与已完成双平台1147项完整回归的冻结版本逐字节一致，528份旧保护文件不变。此次复验不重复计作新原生结果。原始图像、物理日志、模型和私有归档不加入Git。

本次包含真实RGB分割及point-source路径、三模式中心状态监督、Teacher KL事务与诊断、冷Reset同步证据准入和失败保存、PhysX来源/关闭诊断及768点尾部审计。Stereo精度、自然退出修复原生复验、逐contact来源与成功Teacher仍未通过；R2、Force Proxy、正式BC/DAgger保持关闭。最终测试和限制见[整合报告](docs/superpowers/reports/2026-11-11-stereo-wp-integration.md)、[readiness](docs/contracts/stereo_phase0_7_readiness.json)和[待批准采集计划](docs/superpowers/reports/2026-11-11-stereo-cold8-preflight-plan.md)。

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_cold_candidate_admission.py tests/test_dh116_stereo_candidate_cold_trials.py tests/test_dh116_stereo_contact_calibration.py tests/test_dh116_stereo_wire_tail_audit.py tests/test_dh116_stereo_state_training_modes.py tests/test_dh116_stereo_wire_segmentation.py tests/test_dh116_teacher_wp3.py
```

以下“未上传/不push”说明保留为预检阶段的历史快照；本条明确上传授权只改变发布范围，不改变实验授权或晋级状态。

## 2026-11-11 — WP1–WP6增量与8初态采集预检查（未上传）

以远端 `ae307a5` 为基线，保留已有架构。新增native-ID监督的双RGB分割、中心线/实测表面来源编码、右侧参考归档与逐点尾部审计；三种状态监督模式实际完成中心61–63的CPU诊断拟合。修复候选采集冷Reset读取、跨链初态身份、Checkpoint双检及失败样本保留。默认相机、371D/24D与原物理成功标准未变。

Roll20同cohort复放为二指91/96、三指89/96个正确点，固定分母2009/2054，完整率4.53%/4.33%。这是采样覆盖改进，尾部仍14–22mm；分割实际接入仍失败，历史片段ID未记录，不能授部署或跟踪资格。真实6行中心标签的训练内MAE为5.395/1.522/4.258mm，没有独立validation/test。

Teacher已完成本轮2env×128steps×1update原生R0诊断：256transition、5个joint_limit终止、成功0，受保护最终KL0.019917488。PhysX左拇指几何标定两次各240步完整，但退出需要采样后定向SIGTERM；修复代码经过CPU验证，尚待原生自然退出复验，pair来源未资格化。CUDA solver仍不可用，R2/Force Proxy、正式BC/DAgger/Student PPO保持关闭。

最终CPU：Windows **1147 passed / 1 skipped**（174.53s）；服务器现有Python3.10/Torch2.7 **1147 passed / 2 skipped**（152.2s），均exit0、无Isaac导入，统计不相加。528保护文件SHA不变。用户本轮明确暂不采集新的8初态、不push/merge，PR保持Draft。

[总报告](docs/superpowers/reports/2026-11-11-stereo-wp-integration.md) · [逐点预检](docs/superpowers/reports/2026-11-11-stereo-wp1-tail-preflight.md) · [待批准8初态计划](docs/superpowers/reports/2026-11-11-stereo-cold8-preflight-plan.md) · [readiness](docs/contracts/stereo_phase0_7_readiness.json) · [本轮文件清单](docs/stereo_wp_20261111_file_inventory.md)。以下保留历史阶段记录。

## 2026-11-11 — 双目点云、状态监督与 Teacher PPO 验证

本次更新以 085f6b8 为基线。Roll20 保存帧采用轮廓匹配和连通表面采样后，每帧二指 31/32 点、三指 30/32 点在 5 mm 内。正确点完整率分别为 1.54% 和 1.46%，三指光轴深度 P95 为 11.92 mm。该配置仍用于独立实验。

完成 143D 逐字段审计及三种 Actor 模式的 CPU 验证。新的 2 环境、4 步原生候选采集核验出 6 条相对中心 XYZ 标签。两个独立初态不足以完成三路划分，状态估计器训练仍待合格数据。两次独立的 2 环境、32 步、1 更新 Teacher 原生诊断均以 0 次抓取成功结束。在同一个 0.02 KL 目标下，回滚及学习率回退将最终 KL 从 5.84524 降至 0.01997，Checkpoint 回读一致。R2 和正式 BC/DAgger/Student PPO 的准入仍待验证。

本阶段完整 CPU 回归在 Windows 和服务器各通过 1063 项，分别跳过 1 项和 2 项。上传前执行以下 CPU 回归，93 项通过，用时 6.34 秒。

    python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_wire_depth.py tests/test_dh116_stereo_state_supervision.py tests/test_dh116_privileged_teacher.py tests/test_dh116_privileged_cli.py tests/test_dh116_stereo_runtime_teacher_identity.py

发布范围包括源码、配置、测试、字段审计和报告。原始测量、模型及日志保存在私有证据目录。[总报告](docs/superpowers/reports/2026-11-11-stereo-reliability.md) · [readiness](docs/contracts/stereo_phase0_7_readiness.json)

## 2026-11-11 — Stereo 运行链路修复发布

发布基线为 `9bfee426363527190011dbca164c104a00cbc999`。修复 RGB 二维拓扑与独立可见片段、TaskStateAdapter 到 Actor、左右掌局部点编码、真实 Teacher 身份和动作标签；保留旧路径，新增独立 latent/task-state/hybrid 配置。高分辨率 PPO 改为唯一帧池与有界 minibatch，粗到细区分默认 audit 与显式 accelerated。

最终 CPU 回归：Windows **1030 passed / 1 skipped**（153.36s），服务器 **1030 passed / 2 skipped**（180.92s），均 exit0。完成 2/8/16/32 环境的 CPU/GPU 帧池测量、三模式 2env×4step 原生配对、Teacher 2env×1update 及 deterministic mean 采集诊断。原始二指0点失败被保留；逐分量拒绝修复后的同帧复放恢复4点，三指9点。详细原生复验见报告。

这些是运行链路证据：Teacher/Student 没有取得成功资格，三模式未训练，深度完整率仍待验证；tiny Teacher batch KL偏高，同一episode跨多update的checkpoint切换边界仍需补齐。rigid_proxy明确标识，deformable_rod仅严格状态接口，真实后端未接通。528保护文件与30mm/5s/0.5mm标准不变，本阶段训练范围为短时诊断，发布目标为 Draft PR #2 的开发分支。

上传前在 CPU 环境执行 pytest -q -p no:cacheprovider @tests，@tests 为 PowerShell 展开的全部 tests/test_dh116_stereo_runtime_*.py 和 tests/test_dh116_stereo_student_ppo.py。176 项通过，用时 23.78 秒。

[完整报告](docs/superpowers/reports/2026-11-11-stereo-runtime-repair.md) · [机器摘要](docs/contracts/stereo_runtime_repair_validation_summary.json) · [readiness](docs/contracts/stereo_phase0_7_readiness.json) · [文件清单](docs/stereo_phase0_7_file_inventory.md)。以下为历史记录。

## 2026-11-11 — 按用户授权上传J2-A/J2-B

本次发布现有开发分支的J2-A（`a5a15fb`）与J2-B（`3cc5c12`）代码、独立配置、测试、报告和readiness；沿用[Draft PR #2](https://github.com/ssdsj12/dexterous-hand-dh116/pull/2)，保持Draft且不合并11-10。最新完整CPU回归885/886与48解析案例为上一阶段已执行证据；本次上传前另执行63项cost/integration检查，63 passed（20.36s，exit0）。528保护文件和1131份服务器已测源码SHA复核一致，未运行新Isaac/GPU实验或训练。

原10run深度精度、物理抓取及部署资格仍未通过；下一工作包J2-C尚未执行。原始图像、运行日志、模型、缓存和私有归档不提交。推送后核对远端分支与Draft PR的HEAD SHA一致；下文保留各阶段当时未推送的历史记录。

## 2026-11-11 — J2-B：四种独立匹配候选

新增显式Census/Hamming、ZNCC、中心梯度L1与.25/.5/1粗到细候选；独立配置/SHA，完整RAW支持、同指标亚像素与全range审计，不覆盖旧默认SAD或模型。新增134项回归纳入最终相关组：Windows885 passed/1 skipped（149.97s）、server886 passed/1 skipped（140.38s），均exit0、无Isaac导入。48个固定解析case通过列出门槛；四候选标准coverage全100%、最大视差P95 .0250001px，Census存在极少坏尾，不能只看P95。

这只是软件/解析证据。原10run native accuracy、物理与部署资格不变，GPU/完整吞吐未测；粗到细全range审计不宣称加速。无新Isaac、训练、push或merge。下一工作包J2-C，J2-F同原始数据比较与J3仍待执行。[完整报告/实际命令/失败记录](docs/superpowers/reports/2026-11-11-stereo-j2b-matching-candidates.md) · [readiness](docs/contracts/stereo_phase0_7_readiness.json)。以下保留阶段历史。

## 2026-11-11 — J2-A合同、可靠亚像素和安全回归（本地）

- 保留默认SAD数值和原12字段配置；新增mask别名、能力元数据与逐项拒绝原因。SGBM confidence保留兼容数值但不再表达质量概率，内部不可用项为NA。
- 新增opt-in `reliable_sad`，采用五成本模型、独立fractional photometric验证和单频重复纹理证据；失败五成本配置独立保留。修复NaN邻居LR绕过、无效RGB影响边界proxy及非正视差换算，来源SHA进入processing合同，旧包不静默迁移。
- `pytest -q -p no:cacheprovider`的五份J2-A测试组合156 passed（29.86s）；最终相关全组Windows751 passed/1 skipped（115.60s）、服务器752 passed/1 skipped（78.03s），均exit0、未导入Isaac。完整选择与命令见J2-A独立报告。
- 保存Pixel A离线crop误差7.218→2.681mm及解析遮挡27个错误有效点；不冒充全wire精度或原生抓取成功。同步更新README、readiness、166项五目录清单和机器验证摘要；保留RED/GREEN原件于本地artifacts，不上传raw、日志、归档或模型。
- 原528保护文件、控制/物理/执行器/Checkpoint和30mm/5s/0.5mm标准不变。未运行新native/GPU/训练，不push、不merge，PR保持Draft；J2-B与原生精度验证待后续工作包。

## 2026-11-11 — 按明确指令发布J0/J1至现有Draft PR

- 发布诊断实现`33d73d3`、完整报告/readiness与J2实施计划；保留原分支、Checkpoint和未达标状态，不合并、不训练、不新增native capture。
- 本次执行`pytest -q -p no:cacheprovider tests/test_dh116_stereo_error_diagnostics.py tests/test_dh116_stereo_error_geometry.py`：31 passed，9.75s，exit0。完整性脚本核验528保护文件、1111份服务器已测源码/测试及52图SHA；`git diff --check`通过。
- 595/596完整回归与10run重分析是2026-11-10历史验证，本次不重复计作新物理证据。公开范围仅代码/测试/文档/处理后指标；排除raw图像、日志、模型、缓存、密钥和归档。

## 2026-11-10 — J0/J1源合同、分区误差与原始数据根因分析

- 新增评估专用error diagnostics/geometry/CLI、原始数据交叉绑定和52张真实可视化；同像素固定reference分母、P50/P90/P95、NA、拒绝重叠/互斥及saved/replay四象限分开，右实例GT缺失不伪造。
- 定位Block9亚像素偏移、低纹理窗口混合、SGBM自洽坏尾及虚高confidence；实际extractor在12首帧env观测全部0/128 token。保持baseline matcher与既有控制逻辑不变；离线counterfactual不是算法晋级。
- 新增31项诊断/几何测试，保存RED/GREEN；最终Windows595 passed/1 skipped、server596 passed/1 skipped，均exit0。已有10run重分析完成57.223s，新增native0，158输入/来源SHA均未变，保护528文件不变。
- 更新J0/J1报告、153项当前文件清单、机器readiness及J2具体计划。accuracy/物理/部署qualification仍false，Teacher/Student未训练；按本轮任务范围不push、不merge、不启动J2算法改造或新Isaac实验。

## 2026-11-10 — Stereo原生接线、构图与分辨率测量收尾

- 仅修正新增native配置：原launcher预检所需的真实三指reset、camera路由和无旧demo的启动参数；原guard、稳定Hold reward和旧控制路径保持。增加front四分辨率及独立roll20/window5与9实验，不改变默认旧rig。
- 修复实际SDK instance-ID/光轴深度验证，逐env实际Wire mesh清单与未知有限深度fail-closed；旧v1材料不升级。保存producer退出、源码/配置/参考artifact SHA，区分测量可验证与accuracy资格。
- 最终Windows相关CPU564 passed/1 skipped（75.44s）、现有服务器Python3.10/Torch2.7 CPU565 passed/1 skipped（67.68s），均exit0。10次实际相机配置采集exit0及重读通过，但accuracy全部未达标；≤5mm正确3D点完整率与有效覆盖分列，最终resolution未选择。完整命令、逐文件清单、指标、失败和剩余门槛见集成/原生报告。
- 保留原528份生产文件raw SHA与历史Checkpoint；无模型训练、无原生Teacher/Student成功声明、不降低30mm/5s/0.5mm。force/support工厂、独立geometry资格、Teacher/Student每抓法100初态及完整训练吞吐待验证；PR保持Draft，不合并。以下同日记录为阶段历史快照。

## 2026-11-10 — 隔离shaping与实际SDK监督/手内坐标接线

- 实现R1四类与R2五类纯测量因子，默认关，signed support deficit只计一次；几何派生目标份额不强制左右等载。native工厂/独立geometry资格未完成，来源不合格拒绝启用。
- runtime逐环境逐维核对SDK q/qd，仅0:24打开state监督；实际FK将RGB-derived worldpoints转左右掌坐标，invalid保留NaN，不作为GT输入。
- 冻结全组Windows CPU528 passed/1 skipped、服务器现有Python3.10/Torch2.7 CPU529 passed/1 skipped，均exit0；原528生产源文件SHA不变。随后开始短相机实验，不训练或更换原模型，物理资格仍另验。

## 2026-11-10 — 多分辨率/亚像素与跨环境审计适配

- 新增320×240/640×360/640×480配置，默认主精度VGA；224×96保留原字节。资源batch/env可调，K随尺寸更新，流式视差与浮点LR一致性，不隐藏缩图。
- K读回单独容纳SDK float32舍入误差，T/物理容差不改；不允许把此改动当提高成功率或真实深度精度。
- 分辨率相关115项CPU组合通过。服务器首轮458通过/6失败：修复私有快照文档/子进程路径，并将AST审计仅适配3.10缺少的空type_params，保留原字节SHA与结构哈希；完整服务器重跑及原生实验待记录。

## 2026-11-10 — Stereo Phase 7：cold trials与raw proof gates

- 新增双抓法平衡冷试验、真实初态去重、逐480Hz正式tracker回放、raw来源/训练heldout闭包与独立模型选择；继续区分CPU、native capture及物理资格。
- 新bootstrap只接受原launcher正常退出且callback完成，非零/异常保留失败。18项真实runpy回归、关联164项组合通过，未改原launcher或单实例锁。
- 原生force/support/external/attachment来源、100独立初态成功和最终Student验收未取得；明确pending，不晋级、不合并。

## 2026-11-10 — Stereo Phase 6：Student PPO与独立raw RGB执行

- 新asymmetric PPO严格分离typed传感历史Actor与371D Critic，复用Tanh Gaussian/24D动作；独立保存policy/Adam/RNG/LR与actor-only导出。
- 接通真实采集/BC/DAgger/PPO/独立RGB CLI；模型绑定calibration、matcher及源SHA，原始GT不进入Actor。训练源登记缺闭包则禁止正式heldout晋级。
- Windows CPU463 passed/1 optional module skipped，exit0；服务器首次CPU6失败已保留日志并修复打包/AST兼容后重跑。尚无原生训练或Student成功，PR保持Draft。

## 2026-11-10 — Stereo Phase 5：真正在线DAgger

- Student实际step访问状态上查询Teacher；整动作owner选择、append-only池、独立Adam与PPO隔离。
- D0、失败访问状态和成功示范分开；无原生Teacher资格时不执行正式第一步。
- 8项DAgger协议测试包含于本轮450项CPU通过组合；原生DAgger与Student任务成功仍待验证，PR保持Draft。

## 2026-11-10 — Stereo Phase 4：pre-action采集与独立BC

- 新增Teacher采集协议、版本化supervision dataset、来源分组holdout、独立BC与训练模式/分块反传回归。
- 完整成功池要求本次原生raw proof；CPU/候选数据不能授成功资格，不修改旧控制与Checkpoint。
- 本次组合450 passed in61.80s；补充BC/运行组合19 passed/1 optional module skipped in18.02s，均exit0。原生采集、独立Student抓取仍pending；详见Phase4报告，PR保持Draft。

## 2026-11-10 — Stereo Phase 3：隔离Stereo Transformer Actor

- 新增真实共享GN-ResNet18/双目cross-attention/时序Transformer、多head估计、typed SensorHistory、SO(3)及masked losses；不改变旧3844D输入。
- 原始证据不足的force监督failclosed；peer发现的metadata-only资格漏洞已补，正式gate重开raw artifacts。
- Phase3单测27通过；含collection、DAgger、BC CLI及runtime后续集成组合51 passed in11.98s，exit0，未启动native/未授任务成功。按用户补充分辨率支持图像分批，原生精度及资源测试待完整链路后进行。

## 2026-11-10 — Stereo Phase 2：RGB-only stereo独立模块

- 新增标定、同步双RGB采集、视差/深度、坐标变换与验证模块；RGB-only原生reader不读取rendered depth，不推进物理。
- 增加RGB可见骨架/弧长点顺序提取，分叉/断开/缺深度拒绝，不补隐藏几何。新增probe和depth-reference CLI，默认不运行native。
- geometry+CLI+perception CPU组合43 passed in4.78s，exit0；实际标定、亚像素精度、原生性能和抓取均未验证。整数视差深度分辨率限制明确保留，PR继续Draft。

## 2026-11-10 — Stereo Phase 1：独立 Teacher 软件路径

- 新增独立371D privileged Teacher/PPO/reward/dataset/checkpoint、配置和collect/train/evaluate入口；新run独占目录和动作/状态/source SHA，不覆盖原模型。
- 原生桥只替换新进程最终训练callback，复用原Task/World/Env和成功判据；default不启动，raw capture不授force/support/native资格。
- CPU集成49 passed in6.10s，exit0；合成update/resume/terminal不是Isaac验证。成功专家池新增原始证据门控，R1/R2无资格不启用，R2未完成因子明确保留。
- 原生Teacher、双目准确度与Student闭环仍待验收；隔离后续开发继续，PR保持Draft。详见Phase1集成报告。

## 2026-11-10 — Stereo Phase 0：状态、动作与奖励审计

- 新增371行状态映射、24行动作合同、奖励公式/系数/dt/重复风险审计和机器JSON；关键21份源码/报告精确SHA、全部528份原源码/脚本/配置保持。
- 记录缺少嵌入有效性掩码、旧261D/26D Teacher合同、V4与旧hold/dwell叠加及正式/诊断滑移差异，明确NA与晋级边界；不修改原行为。
- 现有Windows CPU环境运行审计与已有surface成功/奖励回归；准确命令、计数、日志见Phase0报告。Isaac、GPU物理、训练及新抓取成功均未验证。
- 用户授权继续Phase1～7隔离实现；新增实施计划，先实现R0透传Teacher，不将CPU资格当原生成功。PR保持Draft，原11-10分支不覆盖。

## 2026-11-10 — CUDA 12.8 独立工具链计划（尚未安装或编译）

- 固定最终CUDA12.8 / cp310 / Torch2.7cu128 / SM89，不再将12.2/12.4作为最终编译候选。只读复核59个路径模式及实际nvcc，仍只发现12.2.91、12.4.99；拟用用户目录可写但未进行创建/链接测试。
- 新增[执行计划](docs/superpowers/plans/2026-11-10-cuda128-isolated-toolkit.md)与[官方归档锁文件](docs/source/deformx_cuda128_toolchain_plan.lock.json)。优先12.8.0官方redistrib，33个用户态归档/4,273,865,956压缩字节，逐归档SHA/版本固定，显式排除6个Driver/内核/服务组件；不直接运行全量安装或盲目flatten。
- 源码门禁优先：安全确认现有Windows GitHub身份为ssdsj12，未重试private后端404；仍需维护者授权精确提交及嵌套源码或完整归档。不在源码缺失时执行configure/try_compile/Solver build。
- 计划限制仅新目录、单次构建进程、明确12.8 headers/libs/SM89及Python解释器；保护Torch实际runtime选择，禁止全局loader/CUDA链接/现有包替换。runfile备选的系统写入及管理员影响须先审阅确认，不自动sudo。
- 本次仅路径/文档/metadata核查和计划验证，没有Toolkit二进制下载、安装、编译、GPU/PhysX/Isaac验证或训练操作；不宣称修复或抓取成功，PR继续Draft。
- 上传前实际执行`git diff --check`、6个拟发布文件凭据模式/本次新增相对链接检查、33项lock与已保存官方manifest逐项对照，均通过；不上传artifacts、原始日志、缓存或模型。这是文档与计划发布，不是原生适配完成。

## 2026-11-10 — DeformX 服务器源码与安装来源补查

- 只读检查实际 `/mnt/mydisk/xk/dh116`、项目副本、vendor/third_party、Git HEAD/gitlink/子模块；19 份旧源码都不是独立 CUDA 仓库，目标 commit 和嵌套依赖无法核验，未用旧后端替代。保留搜索范围及未覆盖项。
- 区分 Windows 与服务器认证：服务器 gh 未登录，Git HTTPS lookup 无凭据，当前无 SSH agent/已配置 identity，GitHub 22/TCP 超时；不把网络失败认定为仓库权限拒绝，不重复下载已返回 404 的后端，不输出认证材料。
- 核实际解释器、uv 安装器、Sim 本地 wheel / 未记录来源和 Lab editable 路径；明确 distribution 0.54.4、源码 VERSION 2.3.2 与未知 Git commit 的区别。继续以实际文件 SHA 审接口，不升级、降级或替换环境。
- 更新[服务器补查报告](docs/superpowers/reports/2026-11-10-deformx-server-source-access.md)、README 和前份报告的执行口径；完整CUDA源码到位后，在当前Python/Torch与独立12.8 SDK合同下审查编译兼容性。没有新接入代码、CUDA 构建、Isaac/GPU 物理验证或抓取成功；原训练/模型/vendor/控制器保持。

## 2026-11-10 — DeformX GPU 原环境兼容性审计（原生构建尚未开始）

- 固定指定主库 `8f32ad7`、CPU 子模块 `a5cfaf6`、不可访问的 CUDA gitlink `96b8514` 与官方 HF 资产 `9ef440fa`；629 个已取得源码文件重验 Git blob SHA，未以部分源码冒充 recursive clone。
- 通过实际服务器原启动器执行 426 个 Python 文件的 3.10.12 语法检查，0 失败；独立保存最小资产，3/3 数据文件 SHA 校验通过，USD 骨架/网格与 NPZ 数据可读。语法和资产读取不是 CUDA 编译、GPU 求解或抓取成功。
- 审计实际 Lab/PhysX 源码及上游 demos，记录 link/CoM、wxyz/xyzw、force/impulse 单位、反力提交顺序和一个 exchange 延迟等边界。尚未凭猜测实现或接通新 Adapter。
- 已确认源码阻塞：CUDA preview 源码当前不可访问；另记录版本差异——实际 Python 为 3.10.12 而非 3.10.20，现成 nvcc 为 12.2/12.4 而非初始清单的 12.8，尚未判定编译兼容性。未升级依赖、未编译/启动 Isaac、未修改原 vendor、控制器、Teacher/MPC/Student/PPO、历史模型或成功标准。
- 该轮补查保留服务器实际环境；用户随后已明确最终采用独立CUDA12.8，不以12.2/12.4替代，也不升级原环境。当前执行合同以最上方12.8计划为准。详细命令、结果、未完成验收及继续条件见[独立报告](docs/superpowers/reports/2026-11-10-deformx-gpu-compatibility.md)；PR 继续 Draft。

## 2026-11-10 — Stage 7 D：承托转移八阶段请求状态机

- 新增未接SDK的APPROACH/CLOSE/STABILIZE/LOAD_TRANSFER/LIFT/AIRBORNE_HOLD/RECOVERY/FAILED状态机。0.2s实测双侧对向稳定及有源object-following后，才建议0.5mm/s、累计请求曝光上限2mm的转移。
- 2mm是请求账而非实际位移；Lift默认1mm/s，至3mm/s需独立证据。缺测/丢接触请求原measured hold、pressure恢复/卸载，zero24不冒充安全保持；原observer成功仅保留来源，首terminal/no-op/reset时钟隔离，不自授任务成功。
- 本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。
- M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。 [本次组合报告](docs/superpowers/reports/2026-11-10-stage7-upload-cpu-components.md)

## 2026-11-10 — Stage 7 C：局部力驱动响应与批量FK数学预览

- 新增完整17D局部implicit capped force-PD数学响应及批量mounted FK，左右手按各自SDK行、关节轴和单位计算。NONE/零cap驱动与缺参数分开，未知量不补0，原始M不对称化、不加单位惯量或完美mimic。
- 未来17D targets/load是显式假设，驱动、接触、补偿符号及真实响应未校准；error_tube=None。批量/标量数学一致不代表原生几何、物理精度、实时deadline或动作准入。
- 本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。
- M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。 [本次组合报告](docs/superpowers/reports/2026-11-10-stage7-upload-cpu-components.md)

## 2026-11-10 — Stage 7 B：H8原命令桥接与约束诊断

- 新增未接控制的64子步命令假设与固定13项约束报告，复用原共享kernel和真实左右SDK命名映射。每手12个直接目标不冒充17D实际响应；反馈冻结/外推必须显式声明。
- 重放原有typed命令诊断、保留当前SDK限位/速度/lead来源与混合单位；物体、接触、碰撞和完整原控制等缺口仍NA/UNCERTIFIED，feasible=False、action=None。同步更新旧生命周期测试夹具，不改变生产控制器。
- 本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。
- M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。 [本次组合报告](docs/superpowers/reports/2026-11-10-stage7-upload-cpu-components.md)

## 2026-11-10 — Stage 7 A：三路被动接触观测侧车（默认关闭）

- 复用现有Wire→桌面/两端支架视图，新增默认关闭的只读采集接线；保留实际路径、每getter时钟/来源和独立normal/friction原值及NA。不创建World/view，不改变原控制或support字段。
- 正常量先own再读摩擦；所有行的count/start、容量、重叠仍严格检查，仅按真实Wire行选择活跃非reset载荷。保留首terminal、退休代及失败partial，不以空报告、未知摩擦或numeric ID缺失填造反力/资格。
- 本次独立Linux CPU主验证：1471 passed, 3 skipped, 246 subtests passed in 307.60s (0:05:07)；exit0，Native作业0，公开测试文件60个。本机完整组合失败与独立PRIVATE parity审查另记；各组是同一主验证组合的子集，不另相加。
- M0、默认Teacher、3844D Actor、371D Critic、24D Tanh Gaussian及原物理/材料/执行器/关节保护和30mm/5s/0.5mm成功标准保持。PR继续Draft，不合并，不开放新专家池或B3/PPO迁移。 [本次组合报告](docs/superpowers/reports/2026-11-10-stage7-upload-cpu-components.md)

## 2026-11-10 — Stage 7 双手接触状态与原预算委托

- 新增独立、未接线的双手接触状态wrapper，区分真实接触与几何候选、未知接触与LOST，显式支持pinch/tripod必需指。原1mm/1s/0.5s恢复预算与reserve/commit顺序委托既有ledger，接触闪断不补预算。
- 7状态包含正式CONTACT_CANDIDATE；未知slip/opposition独立NA，selected reset预检并隔离兄弟，首terminal不被no-op覆盖。原20ms仅pressure/recovery稳定，不是双侧0.2s或任务成功。
- 独立immutable source15组合CPU实跑1360 passed、3 CUDA-unavailable skips、238 subtests；本单元27项是其中子集，不另相加。
- 本单元未Native接入；既有B使用原恢复路径，D1的ABI元数据NA也不能算FK/J通过。不宣称完整MPC、新成功、专家晋级或B3/PPO许可；原默认Teacher、3844D/371D/24D与30mm/5s/0.5mm保持，PR保持Draft。详见[独立报告](docs/superpowers/reports/2026-11-10-stage7-bimanual-contact-runtime-cpu.md)。

## 2026-11-10 — Stage 7 H8 完整17D运动学外推

- 新增独立、未接线的双手完整17D运动学核心：8个60Hz控制周期展开为64个480Hz子步，显式速度预测与当前q/qd、来源、候选摘要分开。常速外推与候选无关，不能提供动作响应、物体/接触/驱动预测或控制准入。
- 完整17D含被动行，左右各用自己的mounted FK；区间速度逐步积分，缺失/无效数据不补0，不把目标当实测响应。
- 独立immutable source15组合CPU实跑1360 passed、3 CUDA-unavailable skips、238 subtests；本单元39项是其中子集，不另相加。
- 本单元未Native接入；既有B使用原恢复路径，D1的ABI元数据NA也不能算FK/J通过。不宣称完整MPC、新成功、专家晋级或B3/PPO许可；原默认Teacher、3844D/371D/24D与30mm/5s/0.5mm保持，PR保持Draft。详见[独立报告](docs/superpowers/reports/2026-11-10-stage7-h8-kinematic-cpu.md)。

## 2026-11-10 — Stage 7 descriptor 显式 USD TimeCode

- 修补当前安装USD绑定的 `GetPropertyStack` ABI：每次capture取得一个Default TimeCode，Attribute和Relationship共享它；不改属性Get、结构/关节判据、时钟、NA或默认关闭行为。
- 首次原生descriptor采集8帧/64条全部结构NA，封存审计通过；真实异常是缺少TimeCode参数。没有instance-proxy问题证据，不改变遍历或jointEnabled规则，原失败原件保留。
- 修复版独立source16 CPU1364 passed、3 skipped、238 subtests；原生复测已准备但尚未执行，FK/J未校准。固定腕双侧恢复为另一独立旧控制路径实验，不计作本修补或新wrapper验证。
- 原Teacher、物理、执行器、3844D/371D/24D Tanh与30mm/5s/0.5mm保持；PR继续Draft。见[原生失败与CPU修复报告](docs/superpowers/reports/2026-11-10-stage7-descriptor-usd-abi.md)。

## 2026-11-10 — Stage 7 mounted model CPU preview

- 新增默认关闭的现有stage/hand/scene只读descriptor，保存完整17D/23link/22joint布局及原始新getter、属性来源、时钟、失败partial；原model capture关闭时合同保持。
- 新增左右独立完整关节FK/J数学假设，不反射复制关节轴、不用affine mimic填被动状态、不执行控制；来源与坐标假设显式绑定。
- 独立source12 CPU769 passed、1 skip、238 subtests；新增Native0，实际FK/J和未来动力学/控制仍未验证。原控制/物理/模型/成功标准保持，PR Draft。见[报告](docs/superpowers/reports/2026-11-10-stage7-mounted-model-cpu.md)。

## 2026-11-10 — Stage 7 独立MPC目标算术

- 新增geometry/contact/support/slip/bimanual/joint/smooth/risk八项独立CPU算术和逐项贡献，显式实验权重、候选digest、时钟/单位/mask/NA合同；缺值不补0，已测常量不冒充未来预测。
- 独立state+objective实跑73 passed、41subtests、exit0；本次Native作业0。未生成或下发动作、未修改奖励/原控制器，不宣称H8、动力学、实时或任务资格。
- 默认M0/Teacher、3844D/371D/24D、物理和成功标准保持；180步原Teacher观测仍运行，正式MPC/恢复/配对/新专家Student未完成，PR仍Draft。见[验证报告](docs/superpowers/reports/2026-11-10-stage7-mpc-objective-arithmetic.md)。

## 2026-11-10 — Stage 7 filtered-contact数值与epoch v2

- 新增默认关闭的v2首回合逐env永久退役/显式generation收据；v1严格合同保留。仅完整归档成功后推进ledger，异常保留首terminal与同次partial，不把reset新回合混入首回合。
- F1复用原34 sensor views，保存实际配对normal/friction owned原始数组、来源/时钟/SDK行；两通道独立valid/NA，不补零、不将几何或可解析值当实际承托。
- 公开范围实际639 passed、1 skipped、238subtests；epoch1/values1/values16自然exit0及封存后审通过，8/8/128帧、数值2176/34816 valid且NA0，非零观测均0。全部截断，未验证真实异步terminal、不增加任何任务成功；低速诊断不授60Hz能力。
- 180步独立接触建立观测已显式启动，尚无结果。原控制/物理/24D Tanh/默认Teacher与成功标准保持，H8/恢复/载荷转移/100初态配对及新专家Student仍待完成，PR继续Draft。见[验证报告](docs/superpowers/reports/2026-11-10-stage7-filtered-contact-values.md)。

## 2026-11-10 — Stage 7 filtered-contact路径元数据

- 新增F0实际nested sensor/filter path resolver与fresh provider，原34 sensors/现有Wire view逐getter owned/clock/source/view括号，默认关闭，未知/漂移保留raw并拒绝，不补numeric partner ID或force资格。
- 新增独立有限 `--vision-mpc-filtered-contact-capture`，保留原state+aux/Teacher source-chain与原控制器；provider首错和secondary写错、首terminal、逆序清理分别保存。
- 实际1/16控制步原生短测exit0，闭合后审分别8/128帧、2176/34816配对，failures0；所有回合截断，不计成功。公开范围独立CPU582 passed、1 skipped、223subtests；原始数据及源码前后SHA不变。
- 已确认v1 previous-terminal严格边界会拒绝异步cohort后续帧；完整退役/epoch与数值接触仍未验证，不默清previous、不宣称新MPC/Student成功或实时能力。M0、原Teacher/PPO、3844D/371D/24D、物理/材料/限制与30mm/5s/0.5mm保持，PR继续Draft。见[原生验证报告](docs/superpowers/reports/2026-11-10-stage7-filtered-contact-metadata.md)。

## 2026-11-10 — Stage 7 只读完整关节观测与接触交付

- 新增全articulation只读provider、默认关闭的Model capture与两种独立contact-delivery observer，保存实际issued24/480Hz时钟、SDK/source/单位/元数据和owned raw；不进入Student、不改控制或物理。
- 修复等值但identity不同的实际SDK enum wrapper识别；保留原类型/协议与转换证据，未知、bool、fraction和不一致映射仍NA。原生source09 Model16后审通过128帧/1024状态，44字段各1024 valid，6个设计缺口保持NA；source08字段绑定失败与原raw不改。
- callback在等记账锁前复制SDK buffer，独立保存orphan/late ledger；Sequence→Aux→Model→Core逆序清理，prior/getter/write错误保留原异常/首terminal，各archive不重复写入，Aux重复finish重试恢复所属hook。
- 本发布范围独立CPU526 passed、1 skipped、147subtests；post-step与full-callback原生短测均exit0/归档检查通过，但full report仍空。所有短测回合均截断，不计任务失败率或成功；未授力/承托/未来动力学或60Hz资格。
- M0、原Teacher/PPO/3844D/371D/24D Tanh、原物理/材料/执行器和30mm/5s/0.5mm保持。H8/恢复/载荷转移/正式100初态配对与新专家/Student仍待后续，PR保持Draft。详见[验证报告](docs/superpowers/reports/2026-11-10-stage7-articulation-contact-delivery.md)。

## 2026-11-10 — Stage 7 辅助接触测量与原 Teacher 完整回归

- 新增实际 Cylinder/单位/COM/inertia 的 source-bound 读取、三个低层 Wire→静态表面 view、已安装 PhysX full report 来源与严格时钟/所有权记录；默认关闭，不加入 Actor 或改变原物理。
- 新增 `--vision-mpc-aux-capture`，与原 capture 共享 UUID/issued24/8 子步，首 terminal 独立保存；scene/raw/close 异常明确标记并完整清理。实际 source-chain 仅接纳原方法或精确的既有 Sequence 包装，不放松原 Shadow 合同。
- 原 Teacher 实际1800控制步/14400物理帧/84461首回合状态/8个terminal；二指4/4、三指0/4（两drop、两timeout），关闭归档完整性通过。不是100不同初态或新Teacher/Student成功。
- 独立 source06 CPU336 passed、1 skipped、93subtests；真实16步辅助冒烟exit0/128帧/1024状态，全8回合截断。记录并保留source05的初始化失败与修复，未在失败目录重试。
- 端支架有非零raw观测，Table矩阵为零而不代表已证无反力；全局report为空，partner ID、合格support/sign/frame/completeness继续NA。H8、正式配对、新专家池与Student提升仍未完成；PR继续Draft，原默认/物理/24D合同/成功标准不变。详见[验证报告](docs/superpowers/reports/2026-11-10-stage7-mpc-contact-source-audit.md)。

## 2026-11-10 — Stage 7 统一状态与原生测量链

- 新增不可变 Teacher MPC 状态及严格 provenance/NA/时钟合同，保留 Student 输入隔离。复用真实 backend target、原接触读取与 COM 点速度换算，不用命令缓存或整手载荷伪造逐指反馈。
- 新增默认关闭的原 Teacher 只读 480 Hz 采集：真实 issued24/8 子步关联、原 refresh receipt、首 terminal 关闭、cohort 与 actual mask 分开，invalid raw 非有限数据可恢复保存；异常时仍清理全部资源。
- 独立 CPU 实测 185 passed、1 skipped、58 subtests。实际 8 环境/180 控制步采集 exit 0，11520 状态与1440物理帧完整对齐；全为截断、无 terminal，不授新 MPC、专家池或 Student 成功资格。完整旧 Teacher 首回合采集仍在执行。
- 保留准备失败：child 输出目录误建、Git 资产清单旧路径。独立 runtime freeze 仅四个路径字符串重绑定，实际 USD SHA 和原源码/物理保持；启动器现在提前校验所有依赖。无证据的 support/反力/partner ID/硬件反馈仍 NA。
- 原默认、PPO、3844D Actor/371D Critic/24D Tanh、物理/材料/执行器和30mm/5s/0.5mm不变。H8、恢复、载荷转移、完整保护及正式配对/新专家仍未完成，PR保持Draft；详见[报告](docs/superpowers/reports/2026-11-10-stage7-mpc-state-measurement.md)。

## 2026-11-10 — Stage 7 原生分叉证据与目标门禁

- 完成 E0/B0/B1/B2 四组来源首回合及六时点比较，各 0/4；接触建立/保持、原标定跟踪与最终失败分别记录，微小数值差不直接归因。
- 完成 64 次真实冷重置短切换、64 项后处理和 256 个 Student 窗口；41,472 个实际 Student 子步、635,392 个 Teacher 前缀子步分开，缺失 0。全部窗口正常截断，未提供 Student 独立成功或新泛化初态资格。
- 以最终 SHA 绑定原生执行、独立进程/锁/原件交接与完整聚合附件；显式保留 event owner、UID、原始指针、诊断锚和原生成功锚区别。HOLD/60 的 B1/B2 四环境原生有效窗口均中断，不能以接触存在或继承高度替代合格保持。
- 当前无闭环目标赢家；B0 控制组、B1 事件诊断基线、B2 条件命令消融均保留，B3/PPO 与默认 Teacher 晋级继续关闭。独立 CPU 310 passed 与原生、Teacher/Student 成功分别报告；提交前仅清理一个测试行尾空格，AST 相同，定向 Linux 锁测试 1 passed/1 cache warning；首次 CPU01 快照失败及 hygiene_cpu01 复制元数据失败均保留。
- 原默认/PPO 路径、3844D/371D/24D Tanh、物理/材料/执行器及 30 mm/5 s/0.5 mm 保持；后续接触测量、MPC、共同暂停完整回合和二指/三指各 100 个不同初态正式配对仍待完成，PR 仍 Draft。实际命令和限制见[Stage 7 报告](docs/superpowers/reports/2026-11-09-stage7-teacher-student-divergence.md)。

## 2026-11-09 — Stage 6 证据分析、成功池合并与接口修复

- 新增只读物理分叉分析，分别保存微小数值差异、超过实际Teacher重复范围/标定阈值的具名关节跟踪偏离、绝对任务事件和逐环境峰值UID/发生时刻。报告、checkpoint、完整job目录、diagnostic UID、issued动作和时钟严格绑定；未记录的post接触范围不补造。
- 新增真实成功池合并工具，保留原观测/动作/历史和四组身份，只重映射本地episode ID；逐父验证原生成功、接触抓法、来源SHA、完整L16 parity，拒绝重复case或失败池，不伪装新Teacher原生报告。当前无真实Tier1成功输入，合并/重训门禁保持关闭。
- 修复Common比较侧CPU backend使用CUDA live索引崩溃，以及Tripod firstterminal后reset slot被旧固定腕目标检查/覆写；不改原active控制或四个controller字节。实际最新CPU 197 passed、2 CUDA skip，另Torch CUDA2 2 passed。纯Student E0/B0各0/4、B1/B2未完成，原生实验继续单实例独立执行，PR继续Draft。

## 2026-11-09 — Stage 6 双侧固定腕恢复实现

- 将Tripod恢复状态、获取原点、取消与压力掩码扩为N×2×3；逐物理episode/hand/digit累计1mm与1s，0.5s无实测进展停止，接触闪断不补预算。真实承载指腹走原压力保持，获准执行后按实际目标提交确认记账，几何预测不算真实接触。
- 显式Stage B支持left/both，保留原A/right；各手使用自己的源FK/Jacobian/pad区域和原非线性碰撞/关节限制。Stage B静态资格不解锁移动腕，C/D/E仍关闭，不产生三指Student专家资格。
- 发布`scripts/run_dh116_stage6_tracking.py`五项有限串行驱动，默认prepare不启动原生，显式execute检查完整SHA/原物理参数并独立freeze/LOCK_NB，busy即拒绝。本次5文件CPU实际79 passed（9.24s），原生A/B及三组完整跟踪结果仍待执行，PR保持Draft。

## 2026-11-09 — Stage 6 独立完整回合跟踪保护

- 新增默认关闭的共同双腕暂停状态机与独立original/unilateral/common实验入口，实际PhysX后端目标与cache分开，保留原控制器F32→F64转换和原guard；原始Actor与privileged hybrid分别记录，不授予纯Student成功资格。
- 用实际Teacher02同钟post-physics q/qd标定每关节P99.9，qualified报告SHA `bbbca065…93adc`。按SDK实际报告时钟单一affine映射校验，原1/480控制dt不变，不逐行拟合消除错位。无法表达真实姿态保持或未知卸载profile明确拒绝。
- 新完整回合入口记录独立UUID、首terminal、backend/cache、实际SDK/base及具名body位姿；legacy palm字段实际是base_link，未知palm_link不补造。当前5文件CPU实际98 passed（13.16s），新原生完整回合仍待串行验证，Teacher/MPC共同暂停晋级未通过，M0默认不变。

## 2026-11-09 — Stage 6 专家初态扰动与来源隔离

- 新增显式Tier1 Teacher manifest、真实reset/首相机包读回及成功二指/失败未完成分池。只改物体XY/yaw，原关节、手根、速度和double Task命令seed保持；不通过原evaluation的float32编码重写Teacher命令。原eval默认分支保留。
- 加固显式row/episode来源lineage冲突及同episode跨train/heldout拒绝，保留legacy轨迹断点分段。独立对真实02池与BC冻结版本比较：4638窗口、事件、概率、1000次同seed抽样及prepared index全部exact。
- 发布实际29job有限驱动的原字节实现和CPU测试。数据/训练/驱动范围125 passed（21.01s），Tier1与原Teacher/reset相关范围200 passed（22.74s），不是相加的原生次数。新扰动manifest已对真实c58/资产合同验证，尚未执行原生扰动或添加专家样本；原PPO及默认控制路径保持，PR仍Draft。

## 2026-11-09 — Stage 6 原生来源与短分支入口

- 接入五命令CLI、绑定真实模型/场景/数据SHA的原生来源和Teacher前缀短分支；分别记录真实issued动作、首次数值偏离、任务物理事件及首terminal。Student launch移除Teacher/MPC/旧评估/M2参数，相机warmup与真实reset读回严格核对。
- 只读cache绑定输入及13个源码SHA，仅复用saved-observation前向和shadow；所有物理reset、前缀及terminal重新执行。实际Teacher tape重放4/4、37584样本、1175policy步，纯Student计数0，来源模型四组和24分支继续串行验证。
- 同4698观测完整离线前向确认B0/B1/B2总体MSE下降，但B1/B2近似，不能据此晋级。当前entry/integration实际25 passed（13.29s），branch39 passed、冻结相关91 passed，非原生次数；完整命令、SHA及边界见[实施记录](docs/superpowers/reports/2026-11-09-stage6-sequence-control.md)。默认路径/成功标准保持，PR仍Draft。

## 2026-11-09 — Stage 6 实际序列BC模型

- 新增masked动作、条件累计目标及可选delta损失，独立`fit_actor_sequences`使用真实4×16连续窗口、fresh BC Adam、单次Tanh和原观测；旧`fit_actor`保持。主B0/B1/B2不重置源Actor，Critic、log_std和原PPO状态保留。
- 实际三组各3000更新、192,000有效曝光、0 padding，最终模型均保存并按预设最终更新报告，未按fit_only训练MSE选模。每250步检查固定事件的PointNet梯度及坐标干预，三组最终探针未失活。新增LeakyReLU仅允许显式新合同，主模型仍ReLU。
- 新原生Teacher采集02二指4/4、三指0/4，当前源码SHA绑定的成功池通过37,584子步及4638窗口严格目标parity。独立实际CPU命令101 passed（16.86s）；模型SHA、预算、命令和边界见[实施记录](docs/superpowers/reports/2026-11-09-stage6-sequence-control.md)。纯Student来源与分支评估在独立目录继续，没有新PPO或默认路径晋级，PR仍Draft。

## 2026-11-09 — Stage 6 条件展开与原生上下文验证

- 抽取原finger/translation/rotation/clearance运算为共享纯函数；shadow只依赖录制反馈，各自累计Teacher与Student命令，不冒充物理模型。修复原slice-inplace造成的完整命令反传错误；独立旧源码160组合×16子步、97,280项逐位一致。
- 新sidecar保存唯一身份、实际previous action、8子步pre-control反馈与真实目标，分开读PhysX backend与IsaacLab cache；terminal后inactive reset目标不参与原回合parity，缺测不补造。新增同钟post-physics q/qd及SDK clock资格接口。
- 新采集01二指Teacher/MPC 4/4、三指0/4；4698条成功视觉动作、37,584物理位native目标一致，完整累计shadow误差为0。新版源码SHA不同而被当前训练门禁拒绝，独立02重新采集；没有更改旧数据或伪造兼容SHA。
- 冻结CPU聚焦25文件533 passed（52.67s）；后续mode互斥检查11 passed。Git过滤往返验证4个来源绑定模块字节不变，避免跨平台换行破坏原生源码SHA。新BC与纯Student资格尚待实际训练及评估，原PPO不改、PR仍Draft。

## 2026-11-09 — Stage 6 序列数据构建

- 新增SequenceDataset、START/CONTACT/LIFT_ONSET/HOLD事件与来源均衡采样，16步真实历史窗口不跨reset/terminal/缺测；短完整episode padding显式mask，恢复标签须独立资格。
- 仅向原监督schema末尾添加sidecar验证/唯一身份联结，原validate_dataset行为不变。真实动作、previous_action、时间、8子步和SHA不一致不按行号拼接。
- 数据及旧schema回归76 passed（5.10s），命令与局限见[实施记录](docs/superpowers/reports/2026-11-09-stage6-sequence-control.md)。预冻结新200初态但原生次数仍0，Teacher上下文采集独立进行，尚未声明BC/Student成功。

## 2026-11-09 — Student原生执行链与逐手证据收口

- 同V4模型、seed及200个初态的诊断复测退出0，二指/三指各0/100；Lift、Hold及逐trial11项已保存统计与首轮相同，不声称全轨迹逐位相同。新复测二指超力/越限/掉落/超时为16/18/51/15，三指为40/7/53/0，旧run失败码不回填。
- 450个关闭块与2183个完整terminal快照独立核验，正式评估仅取200首episode的1353010条物理记录；额外1983次终止不计独立评估。实际clock、UID、Actor24与首终止跨流核对通过，真实backend/cache和17DOF单位覆盖完整，来源与旧缓存证据分开。
- 保存四组掌峰值的env/episode_uid/step/time，以及env45/104/136同环境双手完整动作轨迹。二指掌50mm时物体净Lift接近0，不将掌位抬高当抓取；各组掌位分位、每维动作、载荷及跟踪误差见[独立报告](docs/superpowers/reports/2026-11-09-stage5-student-execution-audit.md)。
- 补T5成功窗的载荷来源边界：ordinary是手物拒接触，桌/支架反力缺测；修正数处旧进度语句，历史模型/数据/资格原件不改。522 passed、1 skipped的源码检查保持；本次仅整理已完成原生数据与文档，不启动新PPO或三指迁移，PR仍Draft。

## 2026-11-08 — 第五阶段物理映射与协同实验实现

- 新增 `tabletop_bimanual_joint_tracking_audit.py`，在原生终止后、物理 reset 前保存三处动作边界、SDK 目标来源、q/qd、被动联动、仿真估算力矩、指腹载荷及同子步推导冲量。T1 独立重放复现两次右 finger12 越限，证据与未知动力学贡献分别保留。
- 新增左右独立物理动作适配器，读取各自源 USD 完整关节树、实际 base/palm 位姿和主动/被动耦合。源 FK、解析/有限差分及 53 组原生微扰已验证，原 SDK 24D 语义往返保持一致。Adapter 版本、资产和源码 SHA 进入显式合同。
- Z0～Z3同源同初态各200首回合已完成：全部0成功；Z1消除下沉但没有合格抓取，Z3三指关节越限66/100，候选不晋级。共同暂停双腕在1024环境、0.533秒短窗消除原两次越限；独立暂停引入的env661过载与完整失败轨迹均保留，不将短窗当成完整抓取。
- 增加V4独立模型绑定、源模型只读派生与运行时SDK/资产/源码精确核验。误标V2/V3但携带Adapter元数据的模型也明确拒绝，避免绕过显式验证。旧M0、Tanh和旧PPO路径保留。
- 新增Teacher专用480Hz首终止审计，分离Teacher请求、SDK目标、实测关节、Hold窗口漂移与重锚前fresh接触锚滑移。T6三指候选以实际T5二指4/4证明和输入SHA为门禁，只修改三指行，保留合法下降接近、接触/跟踪失效时停止共同上升；其修改动作不作为BC标签。
- 本次原Teacher路径与Adapter对照各完成8个首回合：每臂二指4/4、最高31.627704 mm、Hold5 s、Hold锚滑移最大0.079632 mm，三指各0/4。每臂84461条有效480Hz样本通过身份、时钟、动作对应与首终止冻结核验，SDK命令缓存目标读取覆盖100%（后端未记录）；T5通过，两臂10559条动作和主要物理状态一致，语义浮点往返与实际执行差明确区分。
- 增加带原始输入SHA的关节跟踪/接触载荷图及视觉迁移报告。第1024轮源模型的PointNet在两组实际固定输入上输出零特征、对坐标清零不敏感；只报告已测输入，不伪造第513轮或训练起点因果证据。
- 新增独立`bound_student_eval`入口与authored验证初态生成器，严格绑定V4、模型/清单SHA、相机、物理及预处理合同。修复冻结源码路径替换误改显式模型路径的问题，锁后复核实际child argv身份；新增篡改/路径冲突回归。生成时200个authored初态不计原生；后续实际读回、标准评估及同集诊断复测均已独立记录。
- 补充逐指恢复接入缺口与只读碰撞资格审查：当前T6只约束双腕，未完成剩余预算逐指恢复；旧272个cooked hull拓扑实际闭合，复合核验失败集中在近似合面一致性。原数值容差、cooking/资产及live资格均未改，不能用离线包络冒充已附着原生形状。
- T6完成原生8首回合：二指4/4、请求/执行及SDK主动关节目标保持；三指0/4、最高0.002054 mm、六pad共承托0 s。右接触提高伴左接触退化，候选不晋级，95184个有效物理样本与首terminal独立核验通过。
- T5/T6结束后完成独立二指BC1000次离线更新：显式恢复PointNet，MSE从0.039165852降至0.000195265，3844D/371D/24D、Tanh、Critic/PPO Adam/log_std/RNG保留。首轮200初态原生评估完成，两抓法均0/100，最高净Lift25.235939 /0.651073 mm。1353010条物理记录、200首terminal和0覆盖独立核验；失败码/逐手SDK链未存，不补造，后续同模型同初态被动诊断复采已完成，见11-09报告。无三指BC/DAgger/长PPO。
- 更正旧目标读回来源：IsaacLab `data.joint_pos_target`为命令缓存，旧native字段不是PhysX后端证明；历史raw、4/4物理成功和旧资格原件保留。新增真后端getter与cache独立字段、V1～V4来源兼容分析，缺读回不再假通过；T1独立原生补证见下一条，TeacherAudit V2尚未取得独立原生验证。
- 新T1独立原生补证完成：764条watch记录的双手full17真后端目标与cache一致，原M2两次越限实测轨迹重现，合法后端目标下仍发生实际跟踪越界。不是新Student/Teacher成功，不倒改旧缓存来源。
- 新增默认关闭的`--vision-evaluation-physics-audit`：60/480Hz显式身份、实际或NA时钟、失败码、掌/关节/载荷及backend/cache分列、独立terminal和分块NPZ。新诊断SHA不混入原五项证据资格；仅按同次raw m判来源有效性，真0/False不误标、缺失不跨步泄漏。统一522 passed、1 skipped，39.46秒；控制/奖励和默认关闭路径保持，后续原生复测结果见11-09记录。
- 修复配对协调器在结束/新episode上继续驱动的问题：动态首回合mask同时限制coordinator、tracking guard和排队目标撤销，活动首回合逐值不变。原冻结Z3不重写，已完成报告仅取原首回合。
- 完整CPU冻结快照6136 passed、113 failed、48 skipped、25 errors及91 subtests passed。对`3f7fd69`的195新增节点全通过、无新回归；37个旧失败状态恢复，旧113失败/25依赖或外部资产错误保留。修正CRLF导致的两处固定SHA误报，恢复漏同步测试，增加定点LF属性，物理值与保护保持。后续T6下降获取、新入口/清单及相关控制、模型和Teacher路径的当时聚焦回归359项通过，43.53秒。

## 2026-11-08 — 右掌偏置第四阶段

- 200 对实测物理初态的完整回合评估显示，M2 三指净 Lift P95 为 0.401537 mm，81/100 例关节越限。M0 三指对应 0.074721 mm、12/100 例越限。两种抓法均无成功。固定其余 23 维动作的右 Z 干预结果、同调度回放对照、初帧点云匹配子集与滑移计数详见[原生配对报告](docs/superpowers/reports/2026-11-08-vision1024-stage4-native-pair.md)。
- 增加 M2 历史动作回放、冷初态配对和固定左手的右 Z 干预。关节回放在物理 reset 前记录完整 articulation、SDK 主动目标与 Actor/执行动作。原轨迹两次越限均可复现。
- 修正第三阶段报告里的终止次数和模型文件留存说明。奖励项相关性按各自有效样本计算，并核对 policy 与物理行的身份。第 1024、1025、1026 轮模型的同输入推理统计见[阶段报告](docs/superpowers/reports/2026-11-08-vision1024-stage4-ppo-bias.md)。
- CPU 验证使用 `pytest -q tests/test_vision_eval*.py tests/test_evaluation_telemetry.py tests/test_dh116_tensor_env.py tests/test_joint_limit_replay.py tests/test_vision_action_diagnostics.py tests/test_vision_mirror_ablation.py tests/test_stage3_ppo_bias_analysis.py tests/test_vision_paired_bias_eval.py tests/test_vision_squashed*.py`。使用原 SHA 固定的 USD 资产和最终代码，281 项通过。

## 2026-11-08 — 视觉 Student 右掌不对称 P0 动作诊断

- 新增默认关闭的 `--vision-action-diagnostics`，逐环境记录24D动作、左右手分量、episode/physics step/时间戳、请求目标、投影前后目标、最终 SDK 目标、实际掌位/速度、净空、safe_z、接触载荷与有效接触；诊断不改变控制、奖励或模型输入。
- 独立最终 Checkpoint 单更新回放记录32768条逐环境样本；右掌原始Z动作均值−0.289、左掌−0.047，左右安全投影前后差值和投影比例均为0。当前证据指向左右初始化/策略非等变，不把投影或奖励修改认定为根因。
- 第513轮Checkpoint与旧峰值的逐环境轨迹不存在，本阶段不声称完成513/1024因果对比，也不修改奖励或拆分Actor。详见右掌不对称审计报告。
- 后续复核修正实际掌位基准：每个环境保存物理 reset 后实测掌位，控制器动态 lift offset 单独记录；修正后的独立单更新回放仍显示左右投影差值为0，未改变原训练。

## 2026-11-08 — 右掌偏置第二阶段双层诊断与镜像消融

- 新增默认关闭的60Hz policy/PPO诊断和480Hz physics诊断；terminal物理样本先保存，后续no-op不覆盖。记录latent、log_std、log_prob、value、advantage、returns、全部reward terms以及逐子步掌位/接触/投影数据。
- 完成M0/M1/M2离线镜像消融：M0 L2 0.754994、M1 0.806492、M2 offline-only 0.296581；M2方向一致率0.833333。镜像回环自身四帧/点云/关节/掌位/quaternion/previous action误差全0。
- 独立短原生回放输出32768条60Hz记录和262144条480Hz记录，未启动长PPO，未修改奖励或旧模型。详见[第二阶段诊断报告](docs/superpowers/reports/2026-11-08-vision1024-stage2-diagnostics.md)。

## 2026-11-08 — 右掌偏置第三阶段 PPO分析与M2原生反事实

- 对32768条policy记录完成按抓法/左右手的动作、log_std、advantage、returns、reward terms、接触载荷和相关性分析；二指/三指右掌Z均值为−0.263025/−0.322588，相关性弱且无terminal/failure样本，明确不作因果解释。
- 新增M2独立原生预处理路径和动作逆变换，M0默认路径保持不变；同Checkpoint短回放中M2二指/三指净Lift为0.003487/0.014168mm、成功0、出现2次joint_limit，未证明可纳入部署合同。
- 补强episode_uid、时间戳/physics_step单调性和terminal覆盖测试，诊断完整性专项10项通过。详见第三阶段PPO、M0/M2和完整性报告。

## 2026-11-07 — 右三指独立恢复、非线性区域与隔离原生 A

- 新增每物理episode/env/hand/digit独立恢复与执行记账，保留1 mm/1 s/.5 s限制，拒绝回滚reset、预算重置和未提交候选收费。
- 新增实际SDK全11关节/全链接校准FK、原USD真实指腹区域、34-hull与双手扫掠审核，分开DOUBLE逻辑积分、float32 SDK编码及真实接触。几何近似与无证书拒绝均明确记录，不更改物理碰撞或执行器。
- 修复接触中途建立后旧接近动作继续执行却不记账，以及旧压力动作在接触丢失后成为无预算接近；每480 Hz切换动作来源并保留原退压。新增独立fresh pressure参考，未将旧MPC失败计数重置来扩大预算。
- 新增完整raw ownership、actual SDK/cache核对、终止/reset/fatal证据、真实掌位移和固定锚点滑移；模拟力矩/命令/drive限制分开。A/B/C门槛不能由单帧、unknown、换锚或返回原位伪造；原支架/桌面载荷未知时不认证承托。
- 分离物理episode接触历史与A阶段内的实际恢复证据；前缀已恢复事件不能提前通过A。逐子步补充实测区域间隙/法向见证、best gap、round和接触/预算状态，供独立重算0.5 s进展判据；不改变动作或预算。
- 最新完整源码CPU回归：5917 passed/原96 failed/28 errors/49 skipped，91个subtests通过；5786个基线节点无状态变化，304新增全通过，共6090个节点，最终1334文件源码清单保持不变。此前A02快照的5883/270结果已被此回归取代；CPU不计原生抓取次数。
- 独立 `stage_a_01` 在进入A控制前因桌面父Xform未直接带CollisionAPI而退出，0个A试验；只读修复为启用子碰撞体的世界边界并集，原边界与物理限制不变。3项定向及2项独立USD测试通过，01失败数据/源码保留。
- `stage_a_02` 进入真实480 Hz物理子步后由操作者SIGINT结束，被独立审计标为censored/invalid evidence，而非A通过、失败或三指0/4：481逻辑控制、482物理区间、3856原始SDK子步、30840物理行，8个首episode均censored、0完整episode、0接近提交/预算消耗、0 Student/PPO。env4–7只有自然Tripod接触，阶段内0次REACQUIRING、固定阶段恢复、terminal或success。杆/腕未静止，角漂移0.0022–0.00328 rad高于0.001门槛，几何决策0；SIGINT flush顺序造成末8个诊断缺失，未补造reducer。B/C/D和Student不运行。
- cooked readback 03/04均为reset-only探针，不构成A或抓取证据：03实际mesh getter计数272项全为0；04改用owner-first后各存储hull getter count=1，但272个hull全部未通过独立closed-convex复合验证，complete=false/censored、live attachment=false，未recook或替换几何。04无控制/物理/Teacher/PPO/Student，仅读取一次RGB-D。owner-first模块/测试SHA前缀`ddf703d...`/`e55cb...`；最新定向CPU52 passed/13.34s，完整CPU仍需在最新源码后重跑。
- 新增显式独立stage A入口、单Isaac冻结启动和门禁；A02审计结果为censored/invalid，B/C/D仅有诊断门槛，未完成原生运行，未启动Student或长PPO。当前原生对象仍为原刚性圆柱代理，不能替代柔性线缆或实机验证。原生结果在[恢复验证报告](docs/superpowers/reports/2026-11-07-tripod-recovery-budget-validation.md)中独立更新。保持所有历史数据、二指路径、Tanh/v2、3844D/371D/24D、原物理和30mm/5秒/0.5mm，PR继续Draft、不合并。

## 2026-11-07 — 三指几何接触获取与约束求解

- 增加显式Teacher-only几何获取profile：源CAD指腹/实测位姿、有限圆柱侧面、正间隙与朝向门禁、逐pad位移/时间/无进展累计预算和原生fault拒绝；已有承载pad保留压力控制。Teacher腕部不变，不修改Student/PPO/物理参数。
- 完成第一轮8环境固定来源原生回放及独立物理/资产/初态/动作/RGB-D审计：二指4/4、最高31.627704 mm；三指0/4、最高0.006431 mm，四个首回合均掉落。双手六个所需pad未同时承载；不把低压力、运动学预测或接触片段当成功。
- 六个真实停止快照证实被拒的soft提案存在合规替代解，增加独立 `pressure_geometry_constrained` profile，仅修复不可行提案，逐值保留已可行解；记录完整求解输入、H4轨迹、原始代价/约束及float32实际编码首动作验证。原几何模块冻结，原预算/永久停止和压力限制不变。
- 增加默认关闭的有界零参考记录尾段，保存全部实际专家动作与同步传感数据，保留原first-terminal与成功池完整长度门禁；默认历史路径保持。第二轮完成1800控制步/10559样本、8首回合完整、0截断，独立原生审计exit0：二指4/4、最高31.627704 mm；三指0/4、最高0.019719 mm，2掉落/2原超时。1056尾段样本全部验证，尾段成功0；三指六pad加对向仅两个环境各1个480-Hz子步，实际载荷/深度峰增加，稳定抓取仍未解决。
- 完整审核524次求解调用、全H4/编码动作和45504准入子步，执行限位违规0。后段固定法向局部模型的H4冲突具有独立数值证据，不能扩大成物理不可达；CAD中心点间隙不等于接触判据。最终完整CPU5613 passed/原96 failed/28 errors/49 skipped/91 passing subtests；原5670节点不变，116新增全通过。另有177项最终相关回归、261项求解器相关回归及6真实快照检查通过。没有新Student训练或三指合格标签。
- README和[本次验证报告](docs/superpowers/reports/2026-11-07-tripod-reacquisition-validation.md)同步；所有原模型、运行数据和失败证据保留在Git外，原3844D/371D/24D、Tanh Gaussian/default v2及30 mm/5 s/0.5 mm标准不变。未启动长PPO，PR保持Draft。

## 2026-11-06 — 接触MPC与quaternion、启动采样的分阶段验证

- 新增显式绝对参考与压力参考Teacher MPC；前者限制累计参考偏移，后者从实际承载pad建立压力几何目标，并保留原关节、速率和实测lead边界。独立复审修复480Hz累加舍入导致的虚假不可达，以及未知法向保持约束覆盖lead边界的问题。默认局部MPC、物理限制和30mm/连续5秒/0.5mm成功标准不变，模型不声称预测真实力或摩擦动力学。
- 完成BC100纯Student二指/三指各100个固定冷初态、1800控制步与1,503,649条原生样本独立审计：0无效测量，两组均0成功、0双手有效对向接触，抬升仅为数值沉降。确认旧Actor对等价腕部quaternion符号敏感及启动状态欠拟合；新增显式actor预处理合同、BC v3和默认关闭的启动采样，3844D/371D/24D、Tanh PPO及历史raw路径保持，静态恢复和推理核对模式。
- 两项MPC完整独立审计通过：`absolute_reference_pilot_v2` 二指4/4成功、最高31.627704mm，三指0/4且仍为原 `excess_force`；`pressure_reference` 二指仍4/4，三指越过旧约419步过载时刻，但在684/938/783/864步全部 `object_dropped`，最高1.07533mm、成功0/4。接触建立/保持仍不足，三指失败标签不入成功池；这些8环境回放不计作100独立初态或Student成功。
- 实际raw1000、canonical1000、canonical1000_startup25仍为单二指来源、无三指标签/heldout的离线拟合。`f=0.25`启动混合采样将前4步MSE从0.00449366降至0.0000122437，全来源MSE同时增加3.97%；132项保护状态/采样核对通过。启动分布绑定训练子集、来源SHA、算法及真实RNG；默认零比例保持原 `torch.randint` 路径，失败示范不进入成功池。
- 新候选已用绝对参考MPC合格4698帧（3355帧动作修正）完成1000步canonical BC及25%启动混合采样，101项独立检查与73项保护状态检查通过；仍为单二指来源、无三指标签/heldout。纯Student已完成200/200回合，独立审计exit0，原生执行证书与条件遥测完整性核验通过；两组仍均0/100成功，30mm达成与合格Hold均0。二指Lift P50/P90/P95/max为0.00388034/0.0111964/0.102051/24.537986mm；按全部5个pad统计的双侧对向接触在100回合均曾出现，时间占比仅0.7462%，100回合均丢失、共5765次。三指对应Lift为0.0000558165/0.0000843213/0.0001004292/0.0008006615mm，无双侧对向接触、滑移NA；稳定抓取尚未解决。首批200原生动作与CPU最大差1.19e-7及符号不变性复验不另计原生试验。最终完整CPU5497 passed/原96 failed/28 errors/49 skipped/91 passing subtests，5529旧节点不变、141新增全通过。命令及边界见[本次验证报告](docs/superpowers/reports/2026-11-06-contact-aware-mpc-validation.md)。无长PPO或实机反馈成功声明，模型/原始运行数据不上传Git，PR #1仍为Draft、未合并。

## 2026-11-06 — Teacher参考、MPC动作监督与独立视觉BC

- 新增有限原生视觉Teacher采集模式，与PPO路径独立；记录同tick RGB-D、3844D历史、参考/实际issued24D、env/episode/物理时钟及来源。原始RGB-D默认关闭tap分块保存，失败/异常数据保留；首回合原生成功、无recovery和实际抓法审计后才能进入成功池，CPUfixture不能取得原生资格。
- 新训练专家wrapper重用多步局部pad MPC，优化12指关节、保持Teacher腕部12维，逐环境记录实际body/COM/Jacobian快照SHA、激活、修正和回退。明确无完整物体/摩擦/力预测、非任意Student状态DAgger；旧inference MPC限制和原成功门槛保留。
- 新独立Actor BC使用single-tanh bounded MSE和独立Adam/RNG，更新PointNet和action MLP，拒绝privileged输入和失败示范。按原轨迹分组heldout；无独立验证集必须显式fit-only。新版本BC不能直接当PPO恢复，显式派生新v3，逐值保留原Critic/log_std/PPO Adam/LR/计数/RNG及旧来源，后续save/resume保留监督历史。
- 独立复审修复短参考动作结束后缺尾样本误入池，以及已有preflight trace偏移；成功示范严格核对terminal与完整记录长度。最终完整CPU5356 passed，原96 failed/28 errors/49 skipped及5416节点状态保持，113项新增全部通过；本次Teacher基线二指4/4、三指0/4（excess_force）。有限原生和实际学习证据、完整命令/已知限制见[验证报告](docs/superpowers/reports/2026-11-06-visual-teacher-mpc-supervision-validation.md)。无长PPO，不改变物理/执行器/观测动作合同，不宣称抓取可靠或退化解决；PR继续Draft。
- 实际局部MPC二指0/4、峰12.489mm，未将失败MPC标签交给Student。实际BC审计发现1439 PointNet全部失活，旧v1只完成本体拟合，原件保留。新增独立BC v2、训练前失活拒绝、显式仅重置PointNet、私有CPU RNG及真实梯度/点云坐标敏感度门禁；不修改PPOv3架构/合同。
- 实际BC v2完成100步，12个Actor tensor改变、6个PointNet tensor均有100步非零梯度；Critic/log_std/PPO Adam/LR/计数/RNG/完整合同逐值保持，初态SHA独立复核一致。8环境180步教师关闭原生推理正常，0无效测量、0完整episode；仅证明数值视觉分支与推理路径，不能称可靠抓取或每抓法100次验证。

## 2026-11-06 — P0接触证据与中心线几何筛选

- 新增独立逐pad事件归约器和默认关闭的PhysX只读采集：保留原始点/法向、位置/法向分门控、env/episode/子步时序、建立/持续/丢失/censor、七原因、二指三指/左右手有效对向比例与coverage。坏力矩保留为未知和原始逐关节证据，不阻断独立接触记录；模拟力矩与无锚滑移不冒充实机测量。
- 新增源CAD中心线候选生成、独立五指atlas、完整17hull/线缆/self/interhand/table/支架检查与静态LP。seed必须与刚体中心线/COM/轴/长度一致；几何结果不能自签原生准入。02及原腕限位内扩展姿态的03各24候选、200步优化均0 accepted，几何前置失败而未进入LP；全部拒因保留，未闭环执行失败候选。
- 完成实际129/1439同初态配对审计，每模型每抓法100个原生冷初态、0 invalid、readback相同。原生成功全0；保存Lift分位数、接触时间/丢失、模拟力矩饱和、Hold和缺失滑移边界，不冒充历史peak。
- 新增102项测试全通过；完整CPU5243 passed/96 failed/28 errors/49 skipped/91 subtests，旧5314节点状态完全保持。独立规格复审已修复两项几何P1和两项诊断P2。8环境120控制步原生历史观察采集完成，0完整episode、无PPO更新/生产证书，不作新抓取成功证据。
- 按本次P0优先要求取消尚未启动的R0 Adam排队任务；长PPO、课程、单因素消融和泛化未启动。所有历史数据与模型保留，原合同/物理/执行器/成功标准保留。README及[验证报告](docs/superpowers/reports/2026-11-06-grasp-contact-geometry-validation.md)同步，PR继续Draft。

## 2026-11-06 — 原生初态身份审计与真实断点选模兼容修复

- 增加 opt-in 初态 sidecar：完整实际物理读回、camera 前后三个时钟、完整初始 policy SHA；保留/重置 Adam 的 reference 差异在 PPO 构造前拒绝。默认关闭，Actor/Critic/动作/物理合同不变。
- 修复实际1439派生断点的 phase_force_reward.excluded_sources tuple→JSON list 往返；只规范化该已知标签字段，未知 tuple、布尔/数字/物理类型及.pt typed合同SHA仍严格，exact resume 不放宽。
- 采集器在开始采样后使用 blocking 批量CPU快照；terminal、invalid、NA、12关节力矩和24D动作及 raw JSON 对照一致。纯Torch CUDA对照不计原生试验，也未测得新版原生加速倍率。
- 单实例串行 Adam pilot 计划固定同源128环境、各64次新更新，完整保留每轮，只用预冻结1455/1471/1487/1503四候选及独立validation。真实129/1439评估完整证书通过前不得开始；原生结果仍待执行，不宣称退化解决。
- 实际129模型原生冷初态每抓法100次已完成并独立重放854757条首terminal物理数据；两组成功0%，二指/三指最高净lift为0.005627/1.033019mm，slip为NA/coverage0。1439评估继续，不以历史峰值模型代替。
- 最新完整CPU回归5141 passed，原96 failed/28 errors/49 skipped逐节点不变，91个subtests通过；新增235个T2节点全通过，较d796917新增40项全部通过。

## 2026-11-06 — T2评估、选模、warm-start与原生冒烟

- 新增逐抓法100个独立冷初态、实际状态读回、视觉确定性推理和首terminal物理数据保存。native证据绑定checkpoint/manifest/相机/完整12D力矩估计与24D命令；缺测保持null，不能通过自报NA/报告flag取得认证。
- 显式小规模模型投影和reward-only派生保持源文件；周期/最终完整v3 checkpoint不可变保留，原生独立validation选模保留全部候选，successes0不产生best-success模型。
- 增加保留/重置Adam选项，实际加载后复核reset更新预算；保存reset、投影和奖励派生来源。新增R0～R5独立reward合同，正项保留严格物理资格，下降项覆盖真实terminal并正确隔断无效测量；物理reset瞬时基线与inactive tail分开处理。R0默认不变。
- 真实16环境Isaac完成1440/1441两次更新，另进程恢复完成1442；RGB-D有效率100%，最大logp重建误差3.34e-6，原生无更新恢复的完整模型/Adam/LR/RNG/计数逐项相等。successes0，不作抓取改善结论。
- 用户明确授权原进程退出后，先冻结1750轮完整原模型，再正常SIGINT、确认GPU3释放并结束旧监控。1750不是2048预算完成；129/1439原件仍在。配对100次/抓法评估正在单实例串行执行，奖励消融及选模原生结果仍待完成。
- 本轮完整CPU回归5101项及91个subtests通过，原96失败/28错误/49跳过逐节点保持；本阶段新增195节点全通过，原1207源码/资源文件SHA无变化。规格/质量复审PASS。命令和限制见[本次报告](docs/superpowers/reports/2026-11-06-native-stage2-progress.md)；原始模型、原始运行日志和新实验模型均不上传Git。PR #1继续面向11-5，保持Draft、未合并。

## 2026-11-06 — T1训练集成、恢复保护与明确的验证边界

- 新增独立v3显式选项及离线v2派生工具；保留默认v2、3844D actor / 371D critic / 24D动作。记录有界动作概率、熵/KL、饱和率和梯度诊断，不把仿真20D反馈追加到Actor。
- 恢复和派生共用完整policy/Adam/noise/RNG校验；静态相机/初态/物理/奖励配置进行字段新增、缺失和数值双向比较，在写入/启动Kit前拒绝不兼容请求。源checkpoint和旧输出受保护，禁止maximize反向优化PPO。
- 最终327项定向CPU回归通过；规格、质量复审PASS。完整CPU回归4906项及91个subtests通过，原96失败/28错误/49跳过逐项保持；新增278节点全通过，无旧节点缺失或状态变化。实际1439轮原件SHA不变，派生状态逐项保留，真实历史视觉输入确定性动作差0；详见本轮验证报告，不声称旧测试全部通过。
- 新增T2 authored冷重置与首terminal统计工具，43项CPU测试通过；尚无原生认证或100次真实验证。20字段硬件来源/单位/时间/有效性设计补全；真实传感接通状态仍未确认。
- 原训练与1207个源码/资源文件保持，未修改奖励、物理或0.2Nm上限；当前只发布实现与CPU证据，原生配对评估和后续实验等待现有训练自然完成2048后继续。

## 2026-11-06 — T1 有界动作概率与微批 PPO 核心

- 新增独立 Squashed Gaussian 策略与 PPO；rollout 同时保存 latent、实际有界动作和对应 log probability，不从饱和动作逆推 latent。稳定 Jacobian、当前策略有界熵及 Normal KL 有明确概率语义。
- 保留 3844D actor / 371D critic / 24D 物理动作与默认旧训练路径。按样本数累积微批梯度，保留正常配置 40 次逻辑 Adam 更新；有限性保护、单样本和尾批纳入测试。
- 新增 frozen logp、KL、饱和率、各维动作统计和梯度诊断，与旧 loss 返回值分开。107 项新旧 CPU 测试通过，独立规格/质量审查通过；全仓库收集的 11 个旧依赖错误单独记录。
- 入口版本化和显式 warm-start 集成尚未完成；没有修复版原生仿真或稳定抓取证据。原训练、历史 checkpoint、奖励、物理和执行器上限均不变。

## 2026-11-06 — 独立修复准备、T0和硬件来源审计

- 在独立codex修复分支/服务器副本开展工作；原11-5与正在运行的GPU3训练不变。
- 保存第1439轮完整断点、80个有限tensor及完整链日志；源码与资源保留SHA256/只读副本。未发现269/482/1328历史模型，明确记录缺失。
- 完成USD/URDF/ROS/SDK/服务器接口审计；未证实实机传感通道，Actor3844/Critic371保持，native20D不进入Student。
- 隔离CPU基线81项通过；T1按新版本tanh动作合同推进，旧rawGaussian+clip不被误称为天然无效的概率比。尚无修复版原生成功证据；原生评估等待当前实例自然完成。

## 2026-11-06 — `11-5` 当前训练进度与定期高度表

- 更新定期监控说明：每次成功检查都汇报二指/三指本轮及历史最高物体净抬升，保留左右掌本轮/历史最高实际位移和峰值 PPO 轮次；连接失败、停滞、原生首次成功及完成处理保持明确边界。
- 2026-11-06 03:21:14 UTC 只读快照已完成 1328/2048 次 PPO，44007424 transitions、34985 episodes、最慢环境 27 回合、原生 successes=0，GPU3 与采样正常。全视觉链物体最高二指 26.163260 mm@269、三指 31.813391 mm@482，不能以高度门槛代替真实成功。
- 本次验证为服务器/发布源码 SHA256 对齐（699 个运行源文件一致）、完整日志及字段核对、文档链接和 Git diff 检查；1429 项 CPU 回归属于上一轮实现验证。运行训练算法、模型合同和物理配置沿用已发布版本，继续目标 2048；模型、运行日志与本机 SSH/Codex 配置不进入 Git。

## 2026-11-05 — `11-5` 左右掌实际抬升诊断

- 从实测掌心 body-link 世界 Z 计算相对每环境本回合物理 reset 的位移；收集每个存活物理子步，保留终止前峰值。只重设发生物理 reset 的环境基准，不用目标腕位姿、命令位移或物体高度代替。
- 按二指/三指分别记录左右峰值、最后有效样本均值、全局峰值环境 ID 和有效数，缺失为 null；保留向下位移。TensorBoard 直接写入 `lift_mm/{抓法}/...` 的物体/左右掌毫米曲线。模型输入、奖励、RNG 与恢复合同不加入诊断字段。
- 保存第 129 轮完整断点并同配置恢复到新诊断 run，第 130/131 轮用时 39.77/44.61 s；累计 4784128 transitions、2669 episodes、原生 successes=0。两组各 512 环境的左右掌测量全部有效，图像和物体有效率 100%。
- 相关 CPU 回归 1429 项、83 个 subtests 通过，2 项 GPU 测试跳过。当前视觉策略继续训练至绝对 2048 次 PPO 更新；旧手掌历史未记录，不能补填。原生断点与测量证据、命令、限制见[本次验证报告](docs/superpowers/reports/2026-11-05-palm-lift-telemetry-validation.md)。

## 2026-11-05 — `11-5` 固定相机与 1024 环境续训

- 新增固定相机后端：一环境一相机、单 tiled product；初始预热后从一次 render 获取新帧，保留同产品时间、物理步和 FK 冻结检查。只在首次保存最多两个环境的真实传感证据，不使用 GT 补点。
- 显式 4096→2048/1024 缩容转换按抓法平衡投影计数和 critic reference，保留策略、Adam、LR、RNG、累计更新和采样步数；未知字段和类型变化仍拒绝，迁移记录随断点持久化。
- 加入官方支持的 USD 逐环境解析及显式碰撞隔离路径，避开实测快速复制阶段的等待与原生 heap 崩溃；加载原物理属性、几何、材料与驱动。
- 2048/1024×320×240 固定相机均遇到 Vulkan GPU 内存分配失败；1024×160×120 因三指有效点为零而未进入训练。最终采用 1024×224×96，全部初始几何审计通过，最差 P95 0.338320 mm。
- 从第 5 轮续跑，第 6–8 轮实测 39.29 / 42.38 / 40.22 s，render_passes=1、camera_batch_count=1、感知有效率 100%；1 秒显存采样峰值 21.266 GiB。第 8 轮全部 80 个 checkpoint tensor 有限，19 组 Adam step=320。原生 successes=0，训练继续朝绝对 2048 次更新推进。
- 相关 CPU 回归 1194 项及 83 个 subtests 通过、2 项 GPU 测试跳过。每轮数据量和图像尺寸同时降低，吞吐对比与短验证限制详见[验证报告](docs/superpowers/reports/2026-11-05-fixed-vision-throughput-validation.md)。

## 2026-11-05 — `11-5` 视觉 Student 与 4096 环境容量

- 接入前方 RGB-D、真实像素反投影、左右掌心坐标和四帧 PointNet actor。3844D actor 与 371D 特权 critic 分离；二指、三指各占一半物理环境，共享 PPO。旧特权 actor 和无同步图像的教师示范不自动迁入视觉训练。
- 相机池分批覆盖全部环境，核验真实姿态和同一 render product 的时间戳。原生 AA Off 修复观察到的轮廓 RGB/depth 混合；64 环境 RGB-D 临时块保持数值逐位一致。默认场景全可见，避免逐步反复修改大量 USD visibility；可选 batch 模式单独绑定断点合同。
- float32 rollout 存在 CPU pinned RAM；推理分块和加权梯度累积限制 CUDA 激活，保留 40 次逻辑 Adam step/PPO。精确 float64 CAD 高度计算分块，保留全部几何顶点。增加 reset/bind 定位信息并选用 Kit 32 工作线程。
- 最终相关命令通过 1132 项测试、83 个 subtests，CPU 模式跳过 2 项 GPU 测试。另有 CAD 高度 CPU/GPU 14 项与实际 RGB-D 后处理逐位等价验证。命令、环境和容量实验边界见[验证报告](docs/superpowers/reports/2026-11-05-vision4096-validation.md)。
- 本次 4096 原生结果：完成两次完整 32 步 PPO 更新，用时 423.29 / 424.65 s，共 262144 条 transition；两秒间隔的显存采样峰值为 19058 MiB（18.611 GiB），无 OOM，训练已继续第三轮。全部初始图像/物体有效，每环境 128 点，最差 P95 误差 0.390714 mm。原生 PPO successes 为 0，不把容量或高度门槛当作可靠抓取。
- 视觉策略从零运行至累计 2048 次 PPO 更新；`collection_status.json` 与完成更新计数分开，严格合同恢复且断点原子保存。旧 64 环境第 43 轮模型及所有运行原件保留。下一步继续训练并独立监测二指/三指净抬升和首次原生成功。

## 2026-11-04 — `11-3` 从零训练启动器快照

- 归档 4096 环境从零训练使用的 `scripts/fresh_multistyle.py`。它校验多抓法入口的 SHA256，并拒绝从命令行载入旧断点或 Student actor。快照依赖 README 所列的部署路径。
- 发布前启动器通过 Python 编译检查。`--help` 返回 0，`--resume unused.pt` 与 `--student-actor unused.pt` 均按预期退出 1。README 所列的 `python -m pytest -q -p no:cacheprovider` 定向 CPU 回归通过 62 项。7 个对照代码和配置文件与分支基线逐字节一致。
- 本次读取的原生训练状态为二指第 983 次 PPO 更新，累计 85,753 个 episode，最慢环境 17 个，真实 Student PPO 成功数为 0。继续监测首个成功及跨抓法切换。发布验证未启动额外的 Isaac 实例。

## 2026-11-03 — `11-3` 两种抓法续训

- 切换已验证的 reset 家族时重新计算 371D 合同摘要，继续严格比较其余合同字段。36 个相关测试通过。
- 由第 902 轮派生 64 环境续训断点，保留原始断点。二指第 903 轮和三指第 904 轮预检已完成，成功计数为 0。已启动二指、三指各 64 次更新的串行续训。

## 2026-11-03 — `11-3` 奖励边界修复

- 成功终止当步仍可领取达到的 Hold 里程碑。失败当步不发放，按环境重置里程碑状态。
- 滑移独立跟踪有效接触的固定锚。高速下仍可测量，接触丢失或测量失效时停止计分并重新建锚。
- 承托因子计算统一采用原生整手净力上界，并保留原 371D 观测和 baseline 计算。
- 归档 Teacher 原生成功事件逐值一致。将 5 s 里程碑与成功终止对齐后，两次终止子步的发放由 0 分变为 5 分。
- 同组边界测试从 5 失败、13 通过变为 18 通过。完整定向套件 296 个测试和 24 个子测试通过。
- 用同一 Student 策略完成二指和三指各两个首回合并审计 baseline/reference 与八个单因素。原生成功 0/4，最大净抬升为 29.49 mm，最长原生 Hold 为 0 s。修复后的承托因子相对整手原生净力的最大高估为 0 N。原始断点 SHA256 一致。此次执行确定性评估，PPO 学习效果待验证。

## 2026-11-03 — `11-2` 奖励修复

- 高度替换组只为达到新的合格净高度计分，移除旧高度计分，保留历史 baseline。Profile 分开参考组、单因素和累计课程。
- 以完成的 60 Hz 物理窗口计算滑移和断触，分离有效测量与 Hold 成功条件。Early Hold 奖励每回合封顶 10 分。
- 承托组使用合格指腹法向力、可归属摩擦力和实际物重。平滑组使用实测关节和腕部运动及安全投影偏差。
- 统一 371D 入口的 Profile 选项，派生断点逐项保留 PPO Adam、BC Adam、计数与 RNG。真实成功仍要求 30 mm、5 s、0.5 mm。
- 293 个定向测试和 24 个子测试通过。归档 Isaac Teacher 子步重放的成功跟踪输出逐值一致，3 个真实越界滑移窗口全部扣分。第 902 轮 Student 的六组 2 环境原生回放各运行 180 控制步，抓法内观测、动作和杆轨迹逐值一致，观察期内 0 完成回合、0 成功。PPO 学习效果待验证。

## 2026-11-02 — `11-2` contract与确定性验证阶段

- 新增严格24D动作合同和371D反馈合同，保留351D输入前缀、原尺度、时间对齐、无效数据中止和contract SHA。
- 新增351D→371D独立checkpoint派生，保留actor/critic、PPO/Adam、RNG、计数；原checkpoint不覆盖。
- 原生A/B各1800控制步：24D nominal/executed actions逐值一致，B的371D反馈20维全有限，351D前缀最大误差4.44e-16；A/B均0/2成功，未宣称Student已学会稳定抓取。
- 新增单因子奖励消融配置和实际Student失败状态的24D DAgger标签池构建器，禁止Teacher servo进入Student执行。
- 独立审查后修复baseline合同兼容、奖励项注册、保留BC的单因子替换、SHA最终化、真实物重与摩擦支撑、接触中断和滑移记录。纠正标签绑定真实Student状态/episode/tick/env及SHA；未通过原生纠正验证的候选拒绝用于BC。
- 新增只读原生纠正候选采集器和独立辅助BC消费者，共享保留BC Adam，不进入PPO storage。新增独立阶段checkpoint派生器，逐项验证原模型、PPO/BC Adam、RNG和计数均未改写。
- 371D Teacher原生回放2/2成功，1175控制步；与351D示范动作逐值一致，观测前缀最大差2.22e-16。显式派生修正示范合同摘要，原样本及原件保留。
- 本轮直接相关回归190项与24个参数化子测试通过；仅一次更新的完整状态续训371→372已完成，新checkpoint全部tensor有限、PPO/BC Adam均更新，累计28147episode、0success。第372轮确定性Student及纠正候选采集串行进行；奖励原生消融与安全纠正验证尚未完成，未开始4096环境长训练。详见[本次验证报告](docs/superpowers/reports/2026-11-02-371d-contract-validation.md)。

## 2026-11-01 — teacher/student修复、可复现4096 episode续训与11-1发布

- 补齐原生student只读诊断，绑定actor/合同SHA和权重前后不变证明。新服二指teacher回放2/2成功；475轮student去噪16环境最大净升34.225–43.195mm，但最长保持0.250s、0成功，明确区分教师、BC和PPO证据。
- 显式stable_hold_v2采用独立actor Adam、保持/腕通道加权及持久辅助optimizer；holding_v2 Gaussian初始std=0.005，原采样/概率语义保持。新密集lift限定合格原生指腹，增加实际连续保持与3mm起真实固定锚滑移软代价；真实30mm/5s/0.5mm成功和运动许可不变。
- 训练恢复显式记录noise/辅助优化器/奖励变更，原475轮actor/critic、PPO状态及累计回合保留；4096环境每环境2048episode跨style按最低128回合切换，不误设为2048次update。严格student准入改为显式且可继承的选择，默认继续自由探索；启用时验证证据身份、实际训练族和累计诊断预算。
- 增加可选401维因果接触历史与显式输入扩展；本轮351维原生结果不当作401维验证。TensorBoard改为runs/current仅看当前训练，专用Supervisor控制不影响Isaac。
- 修复后原生4096环境实跑已完成第476、477轮PPO更新，原53533个episode/min8保留，独立辅助Adam已有6组状态，模型及优化器tensor全部有限；相关回归1373项和85个subtests通过。当前student successes仍0，不声称已解决稳定抓取的学习结果。
- 原始运行数据、模型和本地备份保留，不上传GitHub；已知全库Open3D/旧DeformX资产及旧fixture失败如实列于[详细报告](docs/superpowers/reports/2026-11-01-student-transfer-repair.md)。发布前相关回归和续训实跑结果另见同报告。

## 2026-10-31 — 二指与三指teacher、student示范训练

- 按用户要求补二指以形成多抓法：定位原二指抬升初期食指失联，新增显式教师的抬升前指令保持。AM每手仅拇指/食指承力，独立复核真实净升31.536 mm、30 mm以上保持5.017 s、相对漂移0.00367 mm，导出二指成功原件；保留原物理/电机限制。
- 三指通过普通TensorEnv补采2个实际成功回合、2542组351/24观测动作，完成BC及64环境2次示范辅助PPO更新、4096条transition；不宣称student已学会lift。新增独立示范辅助损失和严格surface actor迁移，旧156/20入口保持。
- 加入真实保持窗口的抓法标记和第三指拒收检查、二指独立初态入口。二指补采2个实际成功回合后与三指合并4891组样本，按抓法均衡预训练、显式初始化族严格迁移，64环境跨初态4次PPO更新通过，AQ正各追加64次更新。357项集成通过；局部MPC的腕预测、锚点及局部fallback修复完成CPU验证，实际收益仍待评估。没有GitHub推送。证据及限制见[多抓法记录](docs/superpowers/reports/2026-10-31-multistyle-teacher-student-validation.md)。

## 2026-10-31 — PPO训练准备与原生入口验证

- surface模式分力均衡改为默认软奖励、策略拥有动作目标；保留物理联动/执行器限幅，成功不再要求分力精度，回合预算30 s。
- 增加原生控制周期均值与逐子步物理检查的成功跟踪，AK 10,240帧回放与独立5.017 s成功窗口一致；增加显式训练批量原生采样选项。
- 167项定向测试通过；16环境、2次PPO更新、512条transition实跑正常，checkpoint/optimizer有限，TensorBoard事件已保存。未启动长训练、未宣称策略学会lift；静态预抓姿初始化不等于教师策略预热。详见[PPO验证记录](docs/superpowers/reports/2026-10-31-ppo-readiness-validation.md)。
- 修复已完成训练后IsaacLab停止回调持续render导致的退出阻塞，关闭前清理仿真回调和singleton；相关19项通过，最终v2再次完成两次更新并自动退出0。

## 2026-10-31 — 原生中心抓姿、混合闭环及固定总6 N要求

- AK第二例取得独立合格的原生lift示范：最高净升31.534 mm、30 mm以上保持5.017 s、最大相对滑移0.00263 mm，保持时左右实际总法向力1.807/2.664 N；不是3 N精确力控达标。保留原物理/电机限幅，成功轨迹不含物体绑定/外加抬升力；另一个参数例失败，不宣称通用成功率。已导出原生示范数据，未训练PPO策略。
- 用户进一步明确分指力允许偏差、最终以lift成功为准：3 N/手、面间和分指误差保留为控制参考/诊断，不再作提升或成功硬门。补齐倾斜触点的完整整体上移参考，回中只纠正相对运动；力误差不再取消物理盒内向上运动。成功检查真实净升30 mm/5 s稳定有效指腹保持，窗口内滑移≤0.5 mm，瞬时或分指等力不是成功门槛。
- 新增显式教师的实际触点回中、可选原生传感器批量读回与CAD高度批量检查；保持原480 Hz/物理参数，修复PhysX normal/friction共享索引缓存。两秒实测68.09→48.50 s，原始物理/控制记录逐值一致、reset 0；尚非实时或抓取成功。补reset周期不计入新回合稳定时间，以及任意选中承力指的软件主目标冲突测试。
- 按最新要求切换显式教师目标为双手总6 N、每手3 N±0.3 N；每手两个对向面各1.5 N，同面承力指均分，逐指、每面及手总误差分别保存。用户明确采用控制周期真实均值（8个480 Hz子步），瞬时力另存，不额外EMA平滑验收；法向目标不含摩擦。
- 核对source/native FK与COM换点Jacobian正确，筛出4组指腹中心静态候选并补可复现轴枢轴搜索；可选向上支撑偏好只影响搜索代价，不固定手掌姿态。
- 新增显式原生教师全XYZ/法向与后续法向力—切向位置反馈诊断，修复目标冲突、失联寻触、回退优先级、参考到位前馈、reset轨迹和累计滑移锚点；原物体/执行器参数保留，PPO未增加阶段。
- 增加可选批量原生接触采样，保留与原输出等价的回归和不支持布局回落；不改默认正式采样。当前28文件786项和8个subtests通过，后续教师参考及法向优先33项、CPU/CUDA一致性通过。动态合格抓取仍待实测，原始证据和限制见[本轮记录](docs/superpowers/reports/2026-10-31-native-pickup-continuation.md)。

## 2026-10-31 — 有效指腹分面分力与端支架探索

- 新增24维自由手指/双腕探索模式，无固定总目标力；按有效对向抓取面先各50%，再同面承力指均分，保留原实际关节/速度/扭矩限制。
- 新增各指CAD局部指腹与原生接触有效性筛选、普通碰撞独立记录/惩罚、保守摩擦归属和正向奖励来源限制，静态六维重力/力矩/扭矩可行性筛选。
- 新增显式静态端支架与支撑面相对净高度奖励/成功口径；修复reset后旧manifold缓存和旧2–12 N奖励门残留、原生24维通道计数，以及有限负分离项被误判非法导致的reset。普通双手碰撞不能建立抓取锚点。旧模式默认不变。
- 本轮905项定向回归与8个subtests通过；完成短PPO接口更新、4组静态候选筛选。最终L两环境各2 s、960帧实测reset与adapter异常均0，原始力交叉核对一致；圆杆最高净升0.0101 mm，动态合格教师仍0。没有长训练或GitHub推送。实际命令、原始证据及限制见[本轮报告](docs/superpowers/reports/2026-10-31-surface-sharing-validation.md)。

## 2026-10-30 — 受力核对、控制补偿及实际提升探测

- 完成手/桌面法向、摩擦、重力与杆动量的逐步核对；残差约1e-6 N，区分错误静态碰撞过滤与实际波动。960 Hz、TGS和刚性mimic均未带来合格抓取，未设为正式默认。
- 新增按物理时间缩放的控制/验收窗口，显式`--native-pad-compliance`串联柔度候选及原生body/env映射；修复审查发现的斜向投影刚度下限错误。先红后绿，最终501项定向回归、2个subtests通过；未运行全仓库回归。
- 几何搜索加入法向可控性得到4组候选，但J实测18.206 s均未抓稳。独立K2提升探测分开运动许可和原始合格验收，原始读数不改写，3组双腕真实上移约0.32–0.49 mm后暂停，杆未抬起；实测法向竖直合力向下1.2–2.4 N，向上支撑不足。目标力和平摊仍未满足，恢复/重试控制仍有限制。
- 全部对照已结束，reset/合格教师/成功/PPO均0，源码及原始记录保留并下载本地。没有GitHub推送或长训练；下一步筛选向上支撑抓姿并完善接触纠偏。具体命令、统计、版本证据和限制见[本轮报告](docs/superpowers/reports/2026-10-30-force23-momentum-compliance-validation.md)。

## 2026-10-30 — 新Host恢复及每手总力2–3 N诊断

- 保全新服务器旧修改后对齐GitHub`10-30`基线，核验hand环境/资产并恢复专用Supervisor与TensorBoard、本地16006转发。重启脚本改用专用socket，移除历史硬编码PID操作。
- 按用户确认采用每手总法向力2–3 N；承力指平摊预算，未接触指独立寻触；原始144步内的手总力、逐指分力比、CV及防滑分别验收。滑移时撤销排队的上升目标，原限幅和物体物理参数保留。
- 实际A四环境8.625 s均未稳定、未进入提升、reset/成功/教师0；独立Jacobian几何复算正确，记录快速接触/联动波动。contact-last对照因4.5不支持在步进前拒绝，未绕过读回；另执行显式16→64速度迭代数值对照，结果及验证限制见[本轮报告](docs/superpowers/reports/2026-10-30-force23-new-host-validation.md)。没有PPO更新。
- C速度64与D固定目标隔离各完成四环境8 s，均无reset、无合格保持。D结束后部署手总力平衡、承力指控制掩码和元数据声明三项审查修复；最终483项定向回归、shell语法与编译通过。最终修复版的完整物理验收仍未完成。

## 10-30 分支 — 保存多指分担、防滑提升的当前开发状态

- 新增五指原生接触/Jacobian位置驱动控制、均衡载荷分配、原始稳定性窗口、相对杆三点固定参考及显式提升保持入口；保留原实际执行器上限，不固定用户目标力、不启动PPO。
- 新增4组geometry-only预抓姿、可复现搜索脚本、设计与实施计划。214项定向回归、搜索自测及编译检查通过。
- 实际4环境×6s短测未达到稳定条件、未进入提升，reset/成功/教师均为0。完善滑移时手腕队列目标撤销、独立同窗口5s重放和完整物理提升仍待完成。
- 禁止直接沿用旧height-only成功事件导出防滑案例，旧高度摘要单独保留。记录防护修改后未重开Isaac。源码、测试、配置和说明进入Git；运行数据、日志和备份保留在仓库外。详见[本轮记录](docs/superpowers/reports/2026-10-30-multifinger-nonslip-lift-validation.md)。

## 2026-10-30 — 右手式对捏持续接触与实际提升复核

- 显式诊断新增原生 manifold 深度、最近8步最小力接触判据及静止保持选项；源深度继续留存，默认source/mean及普通PPO行为保留。新增有效buffer slice、空接触、padding及坏数据回归，先红后绿，112项通过。
- 单变量实测将静止四指持续承力由约2ms增至3.0896s，但右拇指力波动仍大，不能称稳定抓取。400→200 s⁻¹对照未解决问题，不采用该参数候选。确认源/cooked hull差异，但未解释全部原生间隙差异。
- 实际提升I中双腕均上移46.9208mm，圆杆最高仅离桌0.006880mm后失联，reset=0、成功=0、教师=0；没有力目标、物体外力/附着或PPO更新。保存逐480Hz状态与接触、源/原生深度、独立审计、图和179文件运行源码冻结。详情见[本轮验证](docs/superpowers/reports/2026-10-30-pinch-force-stability-validation.md)。

## 2026-10-30 — 双手适配右手末节对捏参考

- 新增合法刚体拟合、无轴向滚转的杆轴坐标系、完整末节凸包参考和独立几何复核。预抓姿零采样穿透/自碰，最小桌面间隙0.459mm，双手分离。
- 新增显式逐指接触存在性诊断和记录回放入口，保留原实际执行器限幅；无固定目标力、无外力或物体附着。原PPO入口不变。
- 固定微闭合及逐指补闭合的实际试验均未稳定提升；教师0，不用于训练。42文件1043项回归通过，数据与运行源码分别保留。

## 2026-10-30 — 取消目标力门槛的实际提升诊断

- 按用户最新要求新增关节闭合、保持与物理驱动提升入口，不以目标力大小/误差、四指分配或轴下条件阻止lift；此前严格力跟踪入口保留为可选历史诊断。
- 新成功判定使用实测圆杆底部离桌30mm并接触保持0.5s，记录历史成功和末帧保持，逐480Hz原始状态重放检查。保留实际执行器限幅、异常reset和恢复后读回验证；恢复无效时停止共享场景。
- 新模式显式关闭休眠并核对属性，未证明休眠是此前失败根因。41文件940项回归通过。
- 实际40组搜索及同布局重跑均有1组实测离桌49.6405mm、保持4.4542s，末1s双手均力4.3723/1.4632N；单环境原点重跑失败。独立几何审计发现闭合峰值穿透5.119mm，故教师仍为0，不用于PPO；保存可复现预抓姿和逐子步证据。implicit torque只标为估计，配置上限由PhysX原生drive读回核对。

## 2026-10-30 — 多点搜索、持续力跟踪与捏紧保持（实现及实际诊断）

- 实现原生USD多点/全指节候选搜索、有限圆柱SDF、完整凸包/自碰/桌面检查和严格教师导出；双手仍是12主动+10从动=22手指关节。
- 增加逐物理步夹持/托举反馈及目标力探索，修复源身份、动作与力目标一致性、原始子步重放、无效触点、失联旧力矩、手腕上升被norm压缩、坏候选隔离及PI请求限幅问题。保持原0.2Nm电机总预算。
- 用户要求先捏紧保持，新增锁定目标力、保持手腕命令、仅手指持续补力的诊断模式；取消该模式的自动lift和指尖定高约束，保留原PPO自由探索。
- 实测审计发现软件联动纠偏可抵消右食指闭合；保持模式取消该主动目标退回，原生mimic约束、实际执行器限幅及异常隔离保留。使用原位置驱动补力，初始目标在2–3N内选择后保持，关闭额外接触力矩。修正2.1mm桌面软件间隙导致的自动抬手，以及固定补力增益对小力臂的不稳定；均值只用于控制，原始力不改写。最终定向回归40文件863项通过。
- 实际B/C/D均未达成稳定抓取/10mm提升，教师0；全指节几何组合192不代表抓取成功。未启动长PPO。最终保持诊断与验证见[报告](docs/superpowers/reports/2026-10-30-surface-teacher-validation.md)。

## 2026-10-30 — 自由姿态抓取控制（实现及实际试验，尚未抬起圆杆）

- 已实现显式 `--free-wrist-grasp`：手指、抓点和手掌朝向可变；双手本体仍是12主动+10从动=22手指自由度。26是策略命令维数，另含12个外部手腕运动指令和2个法向目标参数，不是26个手指主动自由度。
- 新增六轴物理手腕、完整积分状态观测、真实几何/逐指承力反馈及实际离桌奖励；保留有符号向上奖/向下罚、连续运动与异常隔离。旧20维及手动入口保留；新合同不能直接加载旧模型。
- 本次最终定向回归33文件、668 passed in 12.64s；原生Isaac验证实际旋转与选择复位，随后完成1次PPO更新。3组短PPO分别完成4/8/8次更新，另2组自由位姿接触搜索各360步。
- 最高实际离桌仅0.133572mm，10mm保持和正式成功均为0；存在失载及瞬时过载，不能称为抓取成功。手指力矩、物体物理参数未改；外部手腕参数未经硬件标定。
- DexGrasp-Anything已核对论文和代码，当前仅调研；可借鉴多点指腹几何、穿透代价及多抓姿筛选，不把无重力扰动测试当成本项目lift成功。未安装其环境、未启动长训练。
- 命令、逐项结果、来源和限制见[自由姿态验证记录](docs/superpowers/reports/2026-10-30-free-wrist-validation.md)。后续仍需解决稳定双手承力与实际提升。

## 2026-10-30 — 持续提升与异常隔离（实现及实际验证）

- 新增 `--continuous-motion`，标准手动及 policy PPO 脚本默认启用。有限联动误差记录并有界纠偏，不再单独触发双手暂停、mimic失败或复位。
- 保留逐指力分配、原始接触读数和现有执行器限幅；无效状态先隔离 reset，完整验证复位及首次复发，无法恢复才停止。手动恢复保留 Run/lift 意图，显式 Reset+Pause 保留。
- 本次最终定向回归：27文件、493 passed（11.66s），含169个新增控制、状态与PPO采集恢复用例；shell语法检查通过。关闭模式的历史语义仍有回归覆盖。
- 实际Isaac：单环境5N/手控制下150控制步未复位，双手完成约10mm上移；圆杆未离桌，不能视为抓取成功。两环境异常读数注入只复位故障环境；额外80物理步的真实CUDA采集验证启动校验、坏动作/状态的两次隔离与完整重采，失败记录完整，权重和优化器未更新；PPO更新0。
- 代码和既有驱动/环境未完成硬件能力标定；后续优先解决夹持和抓取保持。未启动长训练，GraspXL A的25维逐指观测/奖励扩展尚未实施。
- 详细命令、对照结果、回执位置和限制见[验证记录](docs/superpowers/reports/2026-10-30-continuous-motion-validation.md)。

## 2026-10-30 — GraspXL 借鉴方案 A（设计阶段）

用户选择保留自由探索、渐进补齐逐指反馈、有限幅度几何引导和抓取提升期间的持续协调。
设计见 [方案 A](docs/superpowers/specs/2026-10-30-dh116-graspxl-option-a-design.md)，待文档审阅后制定实施计划。

- 已核对：PPO 已有逐指负载；新增重点为逐指真实触点信息、统一受限控制及逐指质量评价，避免重复叠加已有奖励。
- 设计保留：固定掌朝向、每手 2–12 N 法向目标、四指轴下、两指力差 <=50%、双拇指稳定及无阶段自由探索；原向上奖/向下罚和独立物理保护保持。
- 当前实现状态：本次仅新增设计和进度记录；联合闭环仍未通过合格抓取与真实 10 mm 提升验收，不将历史测试计为本方案验证。
- 本次验证：源码/约束核对、git diff --check、占位内容扫描及文档链接检查通过；未运行新的 Isaac 或 PPO。下一步先审阅设计，再完成定向测试、单环境物理验收和最小 PPO 冒烟。
- 本地设计提交不代表已经上传 GitHub、部署新控制器或恢复长训练。

## 10-29 故障恢复快照（2026-10-29 UTC）

当前阶段：保存服务器故障前留在本地的源码与配置。以 `10-28` 的 `bd805c6570eb14fdfcbb3beb8d4afd890c2e9415` 为基线，叠加本地可恢复修改；不是服务器磁盘或虚拟环境的完整镜像。

- 已保存：连续抓取/抬升与奖励、掉落及关节异常 reset、手动 Isaac 控制与四指原始接触诊断、拇指稳定判据、逐指力差控制、实验性的接触位置/力联合闭环、相关测试与环境说明。
- 联合闭环目前仅在诊断入口显式创建，尚未接入默认手动窗口，也未通过稳定抓取及真实 10 mm 物体抬升验收。手掌朝向保持，原始测量与物理保护保留。
- 本次上传验证：73 个 Python 文件 AST 解析、8 个配置 JSON 解析、相对模块与测试辅助模块存在性检查、源文件哈希核对通过；本地没有 Torch/Isaac，服务器不可达，未重跑运行时测试。
- 故障前历史验证：联合控制模块最后 10 项定向测试通过；188 项较早的相关回归通过发生在最后几项审查修复之前，不代表此快照的完整回归结果。物理对照仍失败，不能视为成功抓取。
- 未取得：仅存服务器上的 PPO 权重、TensorBoard 数据、完整 Conda 环境和未同步文件。最后已知 PPO 按用户要求停在第 259 轮，本次没有重启训练。

恢复范围、未完成项和历史物理结果见 [恢复说明](docs/RECOVERY_10-29.md)，逐文件校验见 [恢复清单](docs/recovery-10-29-files.json)。

## 10-28 上传快照（2026-10-28）

当前阶段：策略直控与抓取控制改进已实现并完成部分离线验证；接触反馈探索闭环尚在开发，未完成真实物理抓取抬升验收。本分支是当前代码快照，不是全部测试通过的发布版。

- 改进：包含策略直控动作接口、物理驱动手腕、接触力/抓姿保持与恢复、物理诊断、教师/学生验证及训练观测相关工作。策略直控20维动作采用12关节位置、6手腕位置、2驱动刚度，不能直接续训旧力目标模型。
- 本次上传前实际检查：下列针对性命令结果为 **204 passed, 5 failed in 4.88s**。`git diff --check` 与 `bash -n scripts/python.sh scripts/train_dh116_tensor_ppo.sh` 均通过。未运行完整仓库回归。
- 5项失败均位于 `tests/test_dh116_contact_exploration.py`：4项因 ContactExplorer 尚未实现而导入失败，1项因策略任务尚未实现取消旧稳定等待的即时提升转换而失败；保留这些测试以记录待实现要求。
- 实际Isaac状态：此前新策略接口的固定指令演示手腕上移约2.6–2.9mm，物体未稳定抬起，回合失败复位；不是自主接触探索或抓取成功。本次上传操作未重新启动仿真，历史结果不作为本次验证。
- 下一步：完成接触反馈探索与实测抓稳后直接提升闭环；定位物理步进、接触传感器读取和执行循环耗时，再完成单实例实际验证。OakInk仅进行了源码调查，尚未集成。
- 不上传 runs/、archive/、运行日志、缓存及密钥；本地原件保留。依赖本地归档/运行产物的验证不保证仅凭GitHub克隆即可复现。

```bash
DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_policy_control.py tests/test_dh116_contact_exploration.py tests/test_dh116_tensor_training.py tests/test_dh116_tensor_env.py tests/test_dh116_tensor_world.py tests/test_dh116_coordinated_grasp.py tests/test_dh116_lift_grasp_hold.py tests/test_dh116_grasp_shaping.py
```

# 2026-10-28 — 策略直控抓取候选

新增独立 policy 控制模式，取消该模式的脚本补力、普通暂停和抓姿覆盖；策略控制手指位置、手腕位置与驱动刚度。保留驱动边界和硬保护，超力/严重联动冲突使用实测关节中性目标。新动作合同拒绝旧检查点兼容恢复；默认训练脚本选择 policy，并传入 xyz 探索。本次相关离线回归 **204 passed in 5.00s**，shell语法与diff空白检查通过。Isaac保持关闭，本次仅离线验证；实际稳定抓取和提升待验证。详见 docs/superpowers/plans/2026-10-28-policy-owned-grasp.md。

## 2026-10-27 — 修正批量力动作映射回归

- `BatchTask` 全手接触控制在配置上限 `max_force_n >= 2.0` 时使用 PPO 的 2–12 N 动作映射；低容量配置保留 .1–配置上限映射，兼容旧标量/批量环境。
- 新增 12 N（−1→2、+1→12）与 1 N（−1→.1、+1→1）端点回归。验证：`DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests/test_dh116_batched_task.py tests/test_dh116_tabletop_object_contact.py tests/test_dh116_tensor_training.py`（54通过，1预期溢出警告）。

## 2026-10-26 — 当前奖励配置从零重训

## 2026-10-26 — 修正控制后从零训练

- 用户要求从零开始：停止v154，保留第618轮60751872样本检查点及完整旧目录，另存`runs/control_restart_20261026_v155/previous_tensor_ppo.pt`并核对SHA256。
- v155新建训练目录，4096环境、24步更新、每环境1024回合；沿用协调抓握与物理驱动手腕，奖励及物理保护不变。启动不传resume，training_start实测策略未加载旧检查点、初始iteration/timesteps/episodes/successes/Adam状态数均为0。
- 已完成第2轮实际PPO更新，成功0；后台只读观察器启动，TensorBoard运行名`v155_fresh_coordinated4096`。此处验证启动与更新，不代表已学会抓取。

## 2026-10-26 — 抓取控制联调与物理手腕提升

- 修复固定根位姿更新时“手掌位置变化、物理速度为零”的控制路径：增加三轴物理驱动手腕，显式USD DriveAPI；只在复位时写根位姿，正常提升由驱动执行。启动预检与GPU批量路径共用实测手掌观测及跟踪保护。
- 协调控制改用实测手指位移补偿手掌；政策动作下限不再覆盖受力状态下的位移限制，拇指接近限位时释放对侧食指并暂停移动。手指力矩、奖励及硬物理保护不变；手腕支持参数为仿真假设，不是厂家标定值。
- 本次验证：空手16只手均完成约29.9mm行程，最大跟踪误差约0.103mm；最终8回合抓取探针全部完成、无硬保护故障，但3mm与30mm成功均为0，最大实际离桌0.531mm。不能称为稳定抓取成功。
- 相关CPU回归105通过、2跳过（CUDA测试未运行）；shell语法及diff检查通过。详见[联调结果](docs/superpowers/specs/2026-10-26-integrated-grasp-control-results.md)。
- v154在新目录恢复4096环境、每环境24步更新，从保留的第583轮策略/Adam/计数继续；合同仅增加协调控制与物理手腕。已验证第586轮实际更新，成功0。训练准入与抓取成功分开；继续由PPO学习稳定抓握。后台观察器只记录异常/趋势，不自动修改代码。

- 按用户要求停止v133，保留第409轮40206336样本检查点及日志；v134不传resume/initialize-from，全新策略、Adam和计数启动，4096环境、24步更新、每环境1024回合预算。
- 保留受力势差、3/10mm一次性离桌里程碑、close→lift向下−3及持续下压罚，物理参数/保护和156维观测不变。
- 实际training_start验证resume_source为空，初始iteration/timesteps/episodes/successes/optimizer_state_entries全为0；已完成第2轮196608样本，策略/Adam有限，新奖励及切换力事件TensorBoard标签可读，成功0。只读观察器已启动，旧记录未删除。

## 2026-10-26 — 受力方向势函数与实际离桌里程碑

- 用户批准新增两项学习信号，保留切换向下−3、持续下压罚、现有向上/高度/成功奖励与物理保护。新增`--grasp-shaping`仅允许当前close-direct-lift奖励配置。
- 将手对物体有符号竖直合力除以运行时mg，按0.05秒滤波；势函数Phi=2*tanh(filtered_ratio)。每个PPO控制步仅计算一次0.995*Phi(next)-Phi(previous)，与算法gamma相同。成功/失败终止下一势置0，rollout分段不清势，防止通过终止或来回变化增加折扣总收益；这是势差项，终止补偿不能解释为抓取成功奖励。
- 实际物体最低点分别离桌3mm/10mm，lift阶段且实测载荷ready、速度<0.03m/s、滑移≤0.02m持续0.1秒，一次+2/+3；中断稳定重新累计，已领取不重发，复位才清除。最终30mm/0.5秒+25不变，无新增hold阶段。
- 观测追加滤波力势、两项dwell和领取位，151→156。精确单向合同迁移，旧网络列与Adam矩保留，新5列及对应矩补零；实际第402轮检查点CPU加载验证通过。动作仍20维。
- 新增lift_gates遥测，统计lift物理子步中联动/跟踪/载荷/滑移/压力阻塞、实际命令推进与高度上限比例（阻塞可重叠）；不放宽控制条件。TensorBoard新增reward/force_progress、reward/lift_milestone、lift_gates各项；切换瞬时力曲线保留。
- 163项相关CPU/GPU回归通过，覆盖势差终止折扣抵消、60/480Hz滤波/里程碑一致性、选择复位、失败隔离、任务终止快照、151→156权重/Adam零扩展及各提升门槛。shell/diff检查通过。
- v132保留第403轮39616512样本检查点，策略/Adam有限；旧Isaac退出后v133从该检查点继续4096环境。training_start确认第403轮/13项Adam状态保留，迁移报告仅force_progress_lift_milestones_v1及151→156零扩展。实际第406轮39911424样本，首批force_progress产生非零值，lift_milestone仍为0；第405轮检查点策略/Adam有限，TensorBoard新增奖励与门禁/事件标签读回通过。第406轮仅46个lift物理采样中命令推进0，载荷阻塞73.91%、压力阻塞80.43%、联动阻塞13.04%（可重叠）；这是早期小样本诊断，不是完整物理根因结论。物体底部最大0.30000590m，成功0，尚未离桌。独立复核59项CPU通过，另TensorEnv自动复位终止势结算检查通过；观察器无故障。

## 2026-10-26 — close结束向下瞬时扣3分，并显示切换瞬间受力

- 用户确认叠加惩罚：有效close→lift切换物理子步，双手对物体的有符号竖直合力Fz<0时，`close_downward=-3`一次性计入；Fz≥0不扣，既不乘dt也不按力大小缩放。原close中向上奖励与close后持续下压罚分保留。`--close-downward-penalty`仅允许当前直接lift/连续有向力配置。
- 新合同仅允许精确单向增加事件奖励，网络151维、Adam及计数保留，不修改物理参数、阶段准入和成功标准。独立reward/close_downward用于查看事件扣分。
- 新增lift_entry_force遥测，只在有效close→lift同一物理子步、复位前采样。TensorBoard `lift_entry_force_N/竖直合力_向上为正`、`向上力_平均`、`向下力_平均`、方向峰值及左右手合力；`lift_entry_force/事件数`记录本rollout事件个数。对事件求均值，不混入其他时间的受力；无事件时force=null且不写力标量点。原force_N全程曲线保留。
- 144项相关CPU/GPU回归通过，含60/480Hz边界符号、重复/选择复位、非法/失败/inactive隔离、精确合同、环境中切换采样与无事件显示。事件惩罚独立复核16项CPU通过（4项CUDA未运行）；shell/diff检查通过。
- v131第369轮检查点（36274176样本、13项Adam状态）有限，旧Isaac中断不退出，验证检查点后终止；v132单实例从该检查点恢复4096环境。training_start确认第369轮/13项Adam状态保留；第373轮完成36667392样本，策略/Adam有限。该轮实际发生1次close→lift，切换合力−2.16610 N（左−0.75417、右−1.41193），close_downward批量均值×4096×24恰为−3，持续下压罚同时为负。TensorBoard瞬时力8项及事件数、两项罚分实际event读回通过；观察器无故障，成功0。新增遥测独立复核30项CPU通过（2项CUDA未运行）。旧训练记录保留。

## 2026-10-26 — 取消hold，close完成直接lift

- 用户明确要求删除额外hold等待。默认训练脚本启用 `--skip-hold`：双手各1–15.02 N、物体速度<0.02 m/s、滑移≤3 mm连续满足0.2秒后直接从close进入lift，不再等待0.3秒hold；原阶段编号/151维观察保留，未启用该选项的历史流程不变。
- close阶段持续向上奖励；首次close→lift后的下一子步开启向下罚分并锁存至reset。提升仍检查实际载荷、滑移、压力、跟踪/联动误差等，阶段进入lift不代表实际离桌。保留实际离桌30 mm并稳定0.5秒的成功标准。
- 新readiness合同明确hold时长0与close_to_lift，精确单向迁移保留策略/Adam/计数，拒绝反向或额外参数变化。116项相关CPU/GPU回归通过；最终补充普通丢失接触而非异常状态断言后，新5项CPU/GPU测试再次通过。shell与diff检查通过。
- v130第28轮检查点（2752512样本、13项Adam状态）有限；旧Isaac退出后启动v131，以4096环境、24步更新继续。实际已完成第31轮，首次新增close→lift直接转换1次，当前rollout的hold占用0；历史hold计数随检查点保留，不代表新流程仍经过hold。第29轮全在close，向上奖励均值2.00290e-5/控制步、向下罚分0，确认close已获得向上奖励。实际TensorBoard正负力和向下罚分启用比例可读、策略/Adam有限，独立复核4项CPU通过（1项CUDA未运行）。当前成功0，尚未达到离桌验收。只读观察器无报错。

## 2026-10-26 — close阶段向上奖励，完成close后向下惩罚

- 新模式 `--force-reward-start downward_after_close`：close开始即按双手对物体竖直合力 `+0.5*clamp(Fz,0,10)*dt` 持续给向上奖励；向下惩罚 `-0.5*max(-Fz,0)*dt` 从首次进入hold后的下一子步开启。掉回close不重新豁免，只有回合reset清除标记。没有1秒窗口或每回合5分上限。
- close仍要求双手各1–15.02 N、物体速度<0.02 m/s、滑移≤3 mm连续0.2秒；hold再稳定0.3秒进入lift。这是接触稳定条件，不是向上承重或离桌成功证明。物理保护及实际lift验收不变。
- 151维观测保持不变，计分位表示向下惩罚是否启用；新增TensorBoard `force_reward/downward_penalty_enabled_env_fraction`。精确单向v3→v4恢复保留策略/Adam，其他合同差异仍拒绝恢复。
- 验证：close_upward_reward、after_close_force_reward、continuous_vertical_reward、vertical_reward、tensor_training、tensor_task、tensor_env、vertical_force共100项CPU/GPU测试通过；独立复核48项CPU通过（5项GPU未运行），shell语法及diff空白检查通过。真实v129第24轮检查点策略/Adam有限，迁移仅含close_upward_reward_v4；旧Isaac已退出，v130单实例从该检查点继续。

## 2026-10-26 — after_close配置从零训练

- 按用户要求结束v128并保留第117轮检查点，另起v129全新4096环境；移除resume/initialize-from，不恢复策略、Adam或回合计数。
- 保持close完成后计分、连续向上奖励/向下惩罚、151维观测及中文受力曲线。training_start确认恢复来源为空、初始轮数/步数/回合/Adam状态均为0；实际首轮98304样本完成，策略与Adam有限、TensorBoard力标签存在。
- 当前成功0。只读监控继续运行，旧训练记录保留。

## 2026-10-26 — close完成后才计竖直力奖惩

- 按用户最新要求，以首次完成close进入hold为起点；close→hold边界子步不计力，下一个子步开始，之后本回合持续发放向上奖励/向下惩罚。掉回close不会重新豁免，只有本回合reset清除计分标记。没有采用此前讨论的0.5秒方案。
- `--force-reward-start after_close`仅与连续有向力奖励一起使用。力测量、中文TensorBoard曲线及物理保护全阶段保留；时间、失败、实际lift/保持奖励保持原逻辑。TensorBoard新增force_reward/enabled_env_fraction用于区分未开始计分与没有力。
- 追加计分状态观测（150→151），精确单向迁移保留旧网络列和Adam状态，新列及对应动量补零；策略能够观察计分是否已开始。新旧奖励合同被区分，非指定变更拒绝恢复。
- 118项相关CPU/GPU回归测试通过；独立复核54项CPU通过（4项GPU未运行）。真实v127检查点CPU迁移验证旧网络列未变、Adam状态保留。
- v127第110轮10813440样本检查点保留。v128单实例恢复当前策略/优化器/计数，启用after_close；实际已恢复至第113轮，151维观察、13项Adam状态均有限；第111–113轮全部环境仍在close，有实测受力但两项方向力奖惩均为0。TensorBoard计分启用比例和原中文力曲线均正常写入。hold后的正负计分由阶段边界CPU/GPU测试验证，当前恢复首批尚未进入hold，不能据此声称已经抓稳。

## 2026-10-26 — TensorBoard向上/向下力标注

- 新增每个有效物理采样的双手竖直合力正负分量归约，先分向上/向下再求均值，避免正负样本被平均抵消。记录向上/向下平均大小和瞬时峰值，均为N；原有带符号合力及分手数据保留。
- TensorBoard新增`force_N/向上力_平均`、`向下力_平均`、`向上力_瞬时峰值`、`向下力_瞬时峰值`、`竖直合力_向上为正`、`物体重量_参考`。同步英文vertical_force/upward_mean_n等标量和monitor字段。
- 79项相关测试通过，另实际TensorBoard临时event文件读回验证10个新标量（含中文标签）。仅新增遥测，不改变奖励、控制、观测或检查点合同。
- v126保留第25轮检查点（2457600样本，策略与Adam有限）；v127使用同一检查点恢复策略、Adam及计数，避免为记录指标重新学。旧Isaac在回调中吞掉中断，验证检查点后终止，未同时启动第二实例。已恢复至第27轮，training_start确认初始第25轮/2457600样本/4082回合/13项Adam状态；新中文曲线实际event读回及localhost:6006服务可见性通过。第27轮向上平均0.006236N、向下平均0.111738N，差值等于带符号合力−0.105502N。

## 2026-10-26 — 全回合持续向上奖励

- 用户确认取消向上力奖励的1秒窗口与每回合5分上限。新增`--vertical-force-reward-mode continuous`：每个有效非失败物理子步`contact=0.5*clamp(Fz,0,10)*dt`，整个回合持续发放；向下`-0.5*max(-Fz,0)*dt`保持不变。实际高度、保持、成功及有向夹稳一次性奖励不变。
- 保留历史window模式默认供旧配置复现，默认训练启动脚本显式启用continuous。150维观测保留旧时钟以维持形状，但它不再限制continuous奖励；新合同禁用contact_reward_window，监控不再把连续奖励标为已过期。新旧奖励合同不同，不静默加载旧检查点。
- 验证：96项相关CPU/GPU测试通过，包括两种采样频率、超过1秒/5分、晚接触、NaN/失败/inactive隔离、CUDA选择复位、新旧合同拒绝混用；shell语法与diff空白检查通过；独立复核95项CPU测试通过（4项GPU未运行）。
- v125中断并保留第299轮检查点（29392896样本，成功0）。按此前不续训要求，v126启动4096环境、24步/轮、每环境1024回合预算全新训练，不传resume/initialize-from；实际已完成第4轮393216样本，确认resume为空、初始计数0/Adam空，更新后策略与Adam有限。第4轮覆盖仿真1.2–1.6秒且没有回合复位，contact仍为正（均值2.96574e-6/控制步），downward_force为负（−0.0101464/控制步）；证明1秒后仍持续计分。该轮19.36秒，成功0。
- 持续力奖励仍可能鼓励低高度托举，不能代替真实离桌验收；保留原物理守卫，未宣称修复此前超速物理根因。

## 2026-10-26 — 向上奖励、向下惩罚，从零重训

- 使用物体过滤的法向力+摩擦力得到双手对物体的有符号竖直合力Fz。新开关`--vertical-force-reward`替换无方向contact：close开始前1秒内`0.5*clamp(Fz,0,10)*dt`，每回合最多5分；所有live子步（含终止）施加`-0.5*max(-Fz,0)*dt`，向下罚分不设1秒限制和饱和。
- 夹稳+3只在已准入lift、仍ready且实际Fz>0时一次发放；向下或零力不发，允许后续实际托起后领取。实际高度、保持和成功奖励保留。修复lift使用旧力目标、压力释放使用新目标的不一致，复用当前pressure_limited；未放宽硬保护。
- 108项相关CPU/GPU测试通过；独立复核首轮73项CPU、增量14项CPU通过。新奖励与时序合同不静默接受旧模型。默认启动脚本同步当前软指垫/联动400/实测夹持/新奖励配置。
- 用户要求旧训练先备份后删除：`/home/xk/coding/load/dh116_before_fresh_20261026_095257/training_history.tar`，3,419,822,080字节，107145个文件逐项SHA256与归档读取校验通过；旧runs原件已删除。SHA256：`88dbb517bb8bce4e286500bdb861fbe9e469bcc7e357a1a600fe3098d3d5ff5d`。上次10月25日备份保留。
- v125以4096环境、24步/轮、每环境1024回合预算全新启动；不传resume或initialize-from。已完成第1轮98304样本（23.13秒），初始resume为空、Adam状态0、计数0，更新后13项Adam状态及策略有限。真实首轮contact平均每步+8.8144e-6，downward_force平均每步−0.00172565；TensorBoard两项奖励均已写入。成功0，尚未离桌。
- 限制：v123第799轮后的被动关节超速物理根因尚未确定，新奖励和时序修复不等于解决该求解器冲击；原守卫继续生效。历史训练的成功数仍为0，阶段进入lift不等于物体离桌。

## 2026-10-25 — 接触冲击精确复现与求解配置修复

## 2026-10-26 — 手对物体竖直合力及全新PPO

- 启用GPU批量ContactSensor同Wire过滤器的friction_forces_w；与法向力分别取反、按Z分量有符号求和，运行时质量×重力提供重量参照。
- 在每个live物理子步、复位前采样，输出每rollout分手均值、总均值、瞬时峰值及超过重量比例至monitor/TensorBoard/live_watch；不改观测、奖励或lift条件。
- 89项相关CPU/GPU单元测试通过，独立复核38项通过。scripts/verify_dh116_vertical_force.py在GPU480Hz验证作用反作用误差0、动量平衡最大误差6.23e-5 N。
- v122保存到554轮后停止；按用户要求v123全新策略/Adam/计数，从0启动4096环境，不用resume/initialize-from，旧训练保留。training_start.json确认无resume、初始计数0/Adam空；完成第2轮196608样本且策略/Adam有限。首轮18.39秒，第2轮竖直合力均值−0.8631 N、峰值+0.5130 N，重量1.1772 N，超过重量采样比例0，成功0。
- 限制：native摩擦归约有CPU同步，需实测吞吐；超重瞬时力不等于离桌成功，异常中断的部分rollout向上力暂不发布。

## 2026-10-26 — 限制PPO探索噪声并记录动作截断

- 监控识别v121第400轮高斯std9.26–17.33，而归一化动作会在[-1,1]截断；旧熵目标可继续推动无界噪声，理论最低截断比例93.78%。已保留检查点并暂停。
- 新增tabletop_policy_noise.py，逐Adam小批次与resume时投影有限log_std至std[.05,1]，仅清被投影坐标的Adam矩；拒绝非有限噪声。保留原始Gaussian采样/似然计算与执行端截断。
- 精确单向checkpoint契约迁移与恢复审计；不宣称分布等价。新增policy_std_min/max、action_clipped_fraction及TensorBoard/observer记录。
- 验证命令：DH116_NUM_THREADS=6 bash scripts/python.sh -m pytest -q tests/test_dh116_policy_noise.py tests/test_dh116_tensor_training.py tests/test_dh116_training_observer.py tests/test_dh116_tensor_env.py tests/test_dh116_terminal_monitor.py tests/test_dh116_tensor_task.py；90 passed。
- 第400轮CPU迁移确认actor/critic保持不变，20个噪声坐标投影；独立复核无阻断（另23项独立测试通过）。v122实际续训至405轮，std0.9316–1.0、实测截断比例51.40%，第403轮策略/Adam有限、12个网络参数张量已更新；新增3次有界联动误差复位，无全局停机，抓取成功0。

## 2026-10-25 — 接触冲击诊断与有限柔顺联动候选

- 保留v117第294轮关节限位停机证据；v118独立捕获接触冲击速度故障，TGS对照未通过。
- v120仅修改mimic自然频率0→400 s⁻¹，通过272控制步保存动作重放；原故障环境未复位，其他4环境发生有界联动误差复位。有限验证不等于所有故障根治。
- checkpoint_contract_changes加入精确单向迁移，固定求解器、频率、速度/驱动参数、材质及基座写入契约；保留全部物理保护。
- 验证：tensor_training/env、training_observer、terminal_monitor、measured_readiness、compliant_pads、velocity_limits共107项通过；runtime_mimic、mimic_compliance、protected_mimic_config、force_feedback、tensor_task共59项通过、1项跳过。复核收紧迁移后tensor_training 15项通过。
- v121正式PPO使用v117第294轮检查点继续，实测完成第306轮/30081024累计样本；新增12次更新（每环境288控制步），1次有界联动误差复位、无全局停机，抓取成功0。第296轮策略/Adam有限且策略已更新。未采用任何动作重放检查点；原拇指故障及长期稳定性仍待验证。

- **v116诊断完成，v117启动续训**：从v115第196轮真实策略连续采样800控制步（13.333秒/环境），未触发新速度守卫；主动速度峰2.31621 rad/s、全关节2.970024 rad/s、接触力31.27765 N，新增3次联动误差和2次超力受控复位。保留第229轮完整PPO更新、丢弃最后8步不足rollout；诊断无动作重放或物理参数覆写，独立复核确认已完成检查点可续训，策略/优化器有限。v117以4096环境从229轮启动，已实测完成第230轮新更新，力/摩擦/奖励未改。106项相关测试通过。该有限时长诊断未复现v115的8.834 rad/s，物理尖峰原因仍未隔离；已有修复确保下次每子步拦截并报告，历史异常数据不追认为有效。
- **v115运行超速与守卫修复（诊断中）**：监控发现第149轮子步实测主动关节速度峰值8.834 rad/s；旧逻辑仅在终止/复位时检查速度，observer也未检查物理峰值，因此直到第196轮人工发现并暂停。新增每子步全部关节3+1e-5 rad/s检查，超限在奖励与复位前全局停止；精确单向检查点契约迁移。observer补充致命峰值和遥测缺失/格式校验。相关101项测试通过，补充schema后observer11项通过；独立复核无阻断。v116从第196轮策略做诊断采集，正式训练暂未恢复；守卫修复不等于消除物理超速，也不能追认此前超速后的训练数据有效。
- **v115首批完整回合实测**：第113轮核对，相对真实第75轮检查点，4096个环境各新增1个回合；4078次超时、16次未建立夹持、2次受控超力复位，无全局停机，实际抓取成功0。逐子步峰值30.622 N；策略/优化器全部有限，13个策略参数张量发生更新。继续原配置学习，未将进入lift阶段计为抬起成功；只读后台观察器仍运行，不会自行修代码或重启。
- **实测夹持准入候选（v115已启用）**：v112诊断环境545曾双手各≥1 N且物体低速保持1.279秒，但按随机力目标跟踪判定的ready最长仅0.01667秒。新增可选 `--grasp-readiness measured_load`：每手≥1 N且≤1.25×配置最大目标+死区，沿用速度/滑移约束及0.2+0.3秒保持后允许尝试lift。力目标和过压暂停保持；奖励系数不变，但+3准入奖励的触发条件改变并写入检查点契约。仍须实际离桌30 mm稳定0.5秒才算成功。338项相关测试通过；独立复核无阻断，v115真实PPO已产生hold→lift阶段转换，尚未达到实际离桌成功标准。另新增逐子步物理峰值遥测，复位后仍保留峰值，相关51项测试通过。
- **退出状态监控修复**：将 Isaac 启动纳入异常处理，保留最后训练计数并原子发布退出状态；处理清理/写文件失败及非零 SystemExit，先保存结果再执行可能不返回的 Isaac 关闭。9项退出监控测试通过，独立复核无阻断。另修复异常路径丢失最后计数和未完成rollout物理峰值的问题；设备读回/写盘失败不会覆盖原始训练异常，非有限峰值显式标记。新增4项回归测试，训练与退出监控共22项通过；该遥测补丁仅下次启动生效，不为此重启v115。该修复不代表抓取物理故障已解决。
- **硅胶指垫柔顺候选（v115已启用）**：厂家CSV将 finger14/23/33/43/53_link 标注为硅胶A25；新增 `--hand-material grip_compliant` 仅将这5个末端碰撞体设为柔顺接触，K=10000 N/m、C=30 N·s/m为仿真估计，非实测标定。逐碰撞体USD绑定与逐link PhysX材料读回核验，检查点迁移仅允许已定义刚体物理配置。正式v114路径615步Isaac重放已通过，34个link×4096环境的运行时材料审计通过；K由PhysX负restitution编码验证，C验证USD配置，非独立运行时阻尼读回。不宣称稳定抓取。
- **v107历史故障与v115验证边界**：真实PPO续训到第75轮后，环境545出现50.872 N并保护停机。基座写入修复通过已知动作重放，但不是全局根治；第75轮策略已保留。v108已从同一第50轮检查点精确复现（615个控制步）；v109/v110仅放宽主动限速的诊断虽将峰值降到27.108 N，却出现4.415 rad/s主动关节速度，未采用。v112全手柔顺、v114正式仅软指垫均在原0.3 rad/s主动限速下通过615步重放；v114峰值11.311 N、主动速度峰值0.889 rad/s、最大联动误差0.01022 rad，新增物理复位0。v115从真实第75轮检查点续训；以上不代表长期稳定或抓取成功。未放宽过力保护。

- **固定基座重复写入修复候选**：v103精确复现v102的45.669 N冲击。v105保持原545步PPO动作、4096环境及物理/材料/奖励/保护参数，只跳过未变化的float32基座位姿写入，完整重放最高29.423 N，新增有界过力复位0（累计2来自源检查点）。正式实现按最后提交位姿比较，仅写变化环境；回合复位始终强制写选中环境。新增精确检查点迁移；304项相关测试通过（1个既有溢出测试警告），独立复核5项通过。正式生产路径v106完整545步重放通过且峰值与v105一致；原故障环境2645未提前复位（elapsed=3.0667秒），最后64子步峰值1.5695 N。补充乱序复位/实际reset接线后6项测试通过。v107从正式v102第50轮检查点恢复4096环境PPO，诊断重放模型未用于续训；长期稳定与抓取成功仍待验证。

- **后续监控发现仍有接触冲击**：v102在第50轮后环境2645出现45.669 N并保护停机。16次速度迭代只通过了原故障重放，不能视为普遍修复；已启动v103按同一第28轮检查点和随机状态精确复现，补充接触法向、接触点和驱动/几何记录。未放宽护栏。
- **接触求解修复（2026-10-25）**：v99精确复现v98的28轮指标和故障。原动作重放仅将创建场景时的速度求解迭代4→16（位置64、480 Hz不变）；v101通过675步，原故障环境未提前复位，末段峰值57.070→3.590 N。全4096环境峰值37.714 N，10次有界过力回合复位，仍无抓取成功；这验证了该故障轨迹，不代表所有探索轨迹永不失败。启动脚本固定16次，检查点记录参数并只允许精确迁移，原30/45 N保护、12 N即时目标、奖励不变。运行中改USD未生效的v100已排除。验证数据见 `runs/diagnostics_force_root/`、v99/v101目录。 新增配置传递/精确迁移测试先红后绿，299项相关测试通过（1个既有溢出保护测试预期警告）；`bash -n scripts/train_dh116_tensor_ppo.sh`通过。v102从本次全新v98第28轮模型恢复，实测到第44轮/4325376样本，全部4096环境均新增至少1回合；新增2次有界过力复位，其余未建立抓取超时，未全局停机。策略/Adam有限且策略已更新，成功仍为0；运行状态以活动记录为准。

## 2026-10-25 — 历史训练备份后清理并从零启动

- **按用户要求从零训练**：历史 `runs/` 与 `archive/` 已备份到 `/home/xk/coding/load/dh116_before_fresh_20261025_174131/training_history.tar`，tar逐内容比较通过并保存SHA-256后，已删除原目录（归档32398407680字节）。新训练不传resume/initialize-from，使用全新策略、Adam及回合计数。场景输入移至 `configs/dh116_training/` 并保持字节哈希一致；默认启动入口不再依赖旧runs。保留4096环境、每环境1024回合、12 N即时目标及close前1秒奖励。旧v97第243轮发生45.807 N冲击并停止，故障记录保存在备份；重新初始化不视为该物理问题已解决。
- **全新训练已实际更新（v98）**：启动命令无resume/initialize-from，核验第2轮/196608样本、完成回合0；策略和Adam状态有限，最大Adam步数80，未继承旧迭代/回合/优化器计数。当前无失败记录，尚无抓取成功。TensorBoard <http://localhost:6006> 选择 `fresh_instant_4096`；活动路径看 `runs/active_tabletop_run.json`。默认启动命令仍为 `bash scripts/train_dh116_tensor_ppo.sh`，如需强制从零可先清除DH116_RESUME环境变量。


## 2026-10-25 — PPO力目标立即执行

- **当前采用PPO力目标立即更新**：用户确认取消力目标爬升，PPO仍在每手0.1–12 N内自主选择，不固定12 N。新独立配置 `v2_pressure_12n_instant_candidate.json` 使用 `force_rate_n_s: null` 明确关闭目标限速；标量/NumPy批量/GPU控制一致。旧有限速配置与检查点保留。目标相关卸力门槛随之立即改变，实际力仍由接触、驱动及原关节/偏移速率决定；不宣称实测力能瞬间到12 N。

- v97从第225轮恢复至第229轮/22511616样本；运行合同确认目标速率为null，全部4096环境驱动读回通过，策略/优化器有限。已出现1个超力失败回合并按原规则复位，训练继续；尚无hold/lift或成功。297项回归通过（1个既有溢出保护测试预期警告），独立复核4项通过。TensorBoard选择 `v97_instant_force4096`。

## 2026-10-25 — 单手12 N目标与可达加力斜坡候选

- **单手力目标上限12 N候选**：新增独立 `v2_pressure_12n_candidate.json`，单手合计物体接触力目标0.1–12 N；为在6秒抓取超时内让斜坡目标可达，加力速度由0.5改为2 N/s（0.1→12 N需5.95秒）。关节力矩仍0.2 Nm，保护与奖励不变；这不是指尖12 N实测或官方15 N指标的标定复现。旧4 N配置保留。策略/Adam/计数续用，但力动作映射和相关观测归一化发生变化，不宣称行为等价。78项相关测试通过，实际受力及长期稳定性待验证。

- v96实际从第209轮继续至第212轮/20840448样本，运行合同确认为单手12 N、2 N/s；全部4096环境双手驱动限值读回通过，策略/优化器有限。当前无失败记录、仍close且成功0；这是短期更新验证，不是12 N实测承载或长期稳定验收。TensorBoard选择 `v96_force12_4096`。

## 2026-10-25 — 接触窗口起点按用户确认改为close开始

- 每回合从close初始化开始计算连续1秒，即使没有接触也消耗时间；窗口过后contact为0，最多5分，重接触和阶段切换不重开。
- 默认启动参数改为close_start；精确允许现有first_contact检查点迁移到close_start，保留150维策略、优化器和训练计数。拒绝同时改动窗口长度、环境数或观测尺寸。

- v95从第170轮策略/优化器恢复，实际核验第174轮：全部4096环境的一秒窗口已过期，该轮contact奖励为0；无失败，策略/优化器有限。61项相关测试通过。TensorBoard选择 `v95_close_window_4096`；仍未证明抓取/lift成功。

## 2026-10-25 — 每回合一次的一秒接触奖励窗口

- contact按实测双手总物体接触力计分：0–10N对应0–5分/秒，但只在单个连续1秒窗口内发放，每回合硬上限5分。默认首次有效接触启动，另支持close开始；接触丢失不暂停，重新接触/切换阶段不重开，分数在超时后不再发放。
- 取消新模式close/hold进度奖励，首次夹稳进入lift一次性+3；真实提升/保持及成功+25、失败−25保留。失败终止样本不发放新的contact/夹稳奖励。
- 观测新增窗口时钟和夹稳奖励标志（148→150维），原策略输入列和Adam状态保留、新列补零；合同精确迁移，拒绝未授权的尺寸/奖励/物理差异。
- 回归262项通过，含CUDA窗口、频率独立、零接触/丢失/重接触、选择性复位、5分封顶、一次性夹稳、真实RSL优化器迁移续训；独立复核28项通过。
- v92第134轮在新终止步发生3.00001788rad/s超限；不继续扩大保护容差。新增显式被动限速2.97rad/s候选，全部环境逐关节读回验证，v93独立运行旧奖励验证该物理改动。保留3rad/s名义保护，候选不代表完整实机标定或所有故障已解决。

- **新奖励实际续训（v94）**：从第151轮原生检查点迁移，核验到第155轮/15237120样本，策略及Adam状态有限、权重已更新；150维观测生效，部分环境窗口已到期，接触累计分最高1.4179（上限5）。当前无整批停机、成功0，尚未进入hold/lift。TensorBoard选择 `v94_contact_window_4096`；每30秒只读观察器继续记录，不自动修复代码。

## 2026-10-25 — 终止回合立即验证并复位

- v90在第74轮后因post_reset_or_terminal速度3.000018358rad/s再次停机。保留1e-5rad/s数值容差，不继续扩大。v91从原生检查点诊断，实际观察到普通超时后任务已冻结，旧驱动目标仍执行7个物理子步；未在v91首批回合重现v90的同一超限值，不能声称已重放原故障全部轨迹。
- 全部新终止行在终止速度、非有限、独立物理保护检查通过后立即物理复位；任务快照、奖励、终止观测和计数保留到控制步结束，届时只复位任务状态。物理恢复计数仍只记录有界力/耦合故障。复位后的真实物理步仍执行所有保护，禁止通过再次复位吞掉异常。
- 新增终止时序合同字段及精确加项迁移，保留策略/优化器；诊断记录任务终止、失败码、elapsed、当前子步和本步物理复位子步。原回合/动作/奖励/物理上限未调整。
- 13项生命周期回归覆盖提前超时/成功、快照所有权、七种复位后保护、末子步跨控制步检查、普通/物理失败混合及禁用恢复；另增加迁移测试。原实现10项失败；最终相关合集213项通过（10.06秒），diff及shell语法检查通过。
- v91/v92同起点实际第75–89轮指标完全一致；抽样64环境终止前轨迹一致，终止后旧路径最大速度3.0000019rad/s、修复版复位后1.86193e-6rad/s。v92实际至第122轮/11993088样本，恢复后新增12366回合，每环境至少新增3回合，无整批停机；策略/优化器有限且策略更新。仍无hold/lift或抓取成功。诊断轨迹和核验记录仅在本地runs保留。

## 2026-10-25 — 实测过载优先于PPO闭合目标

- v86第9轮后发生接触冲击并正确停机；v87逐物理步同动作复现，保留本地故障轨迹。压力偏置已饱和时PPO目标仍继续闭合，新增显式压力卸载目标约束；不放宽30N、耦合或终止速度保护。
- 过载时沿已标定方向卸力，其他屈曲关节允许打开但禁止继续闭合；暂停手掌微调与控制器提升，保留原关节包络、速度限制及耦合恢复优先级。检查点记录新控制合同，旧原生GPU合同仅允许显式增加该项并记录迁移，策略/优化器保留。
- 新增4项回归，与原GPU/复位/训练/TensorBoard共189项通过。v88续训至第14轮，仍有有限范围内物理失败复位；v89加载提升暂停补丁继续观察。未宣称所有接触故障解决或抓取成功。
- v89完成29轮后，在终止冻结行检查读到3.0000066757rad/s而停机；力5.3067N、耦合0.006607rad、净空2.97mm均在界内。新增共享终止/复位速度谓词，名义3rad/s加固定1e-5rad/s求解器数值容差，所有恢复/终止/复位路径一致，非有限和明显超限仍拒绝。记录独立合同字段和精确加项迁移；未将float32转float64当作超调成因。
- 容差边界、非有限、原故障值和迁移回归合并199通过（8.95秒）；v90从v89第29轮原生检查点续训。物理成功仍未验收。
- v90实际继续到第45轮/4423680样本：首批4094个超时回合完成复位，全部环境至少完成1回合，训练继续，每轮约16.5秒；本次另3个有界物理失败复位，未整批停机。仍全部close、成功0，不据此更改奖励或声明已学会抓取。

## 2026-10-25 — 完整GPU PPO与分阶段奖励

- 新增设备常驻TensorWorld/TensorTask/TensorEnv及RSL-RL PPO：4096环境×24步，保留148维观测/20维动作、原物理和测量保护；检查点版本化，支持同合同原生续训。
- close/hold/lift按实测状态切换，独立接触、稳定保持、真实提升和成功奖励；修复hold丢失接触后再close的奖励循环漏洞。
- 正常失败保留负奖励、批量局部复位；有限范围的力/耦合故障仅在独立保护均通过后恢复，复位后的实际物理步继续检查。无效策略数值、观测溢出等停机。
- 实际4096环境初始GPU/CPU净空一致、状态差<4.77e-7，映射及单环境复位隔离通过；首轮采样16.489s+更新0.182s，5896.8样本/s。16环境2轮更新约5.6s/轮。
- 实际v84误停在3.000000238rad/s，v85同动作复现；修复新迁移误将复位专用速度条件应用于正常live行的问题，保留终止/复位速度条件。v86从v84原生检查点恢复；未宣称完整抓取或lift成功。
- 21项新GPU路径测试通过，包括CPU对照和CUDA优化；修复后与原训练/复位/TensorBoard回归合并185通过。新运行状态和后续结果见README及本地monitor。

## 2026-10-25 — 全手物体接触与逐关节范围 PPO

## 2026-10-25 — v81 缩短 PPO 首次更新等待

- 明确记录 v80 的真实结束状态：225280 样本、0 更新、excess_force。
- GPU 入口新增 `--rollout-steps`，默认仍为 64，新训练选 32；续训保持检查点参数，拒绝显式覆盖。保留所有物理阈值与 close/hold/lift。
- 本地运行脚本可自动识别本次检查点续训并拒绝重复 Isaac 训练启动；v81 已启动，不能据此宣称力超限解决或抓取成功。
- 训练生命周期/TensorBoard 测试 140 通过，shell 语法、Python 编译和非法参数拒绝检查通过。
- 实际 v81 在 131072 样本完成首轮 PPO 参数更新（n_updates=1，loss=0.0261），KL 保护提前结束本轮优化，继续采样至 135168 时无安全停机；TensorBoard 已出现 train 曲线。


## 2026-10-25 — 接触加载耦合故障的同动作对照

- v78的4096环境在55个控制步因右手食指耦合误差0.0218446 rad停机；保留完整实际动作前缀和带符号关节诊断。
- v79在相同环境布局、同动作序列、同摩擦/驱动/速度/阈值下，将位置求解迭代16改64。55步全部完成，峰值误差0.0177825 rad，无安全停机；仅证明已记录故障前缀通过，不是抓取或长期训练验收。
- v80正式采用显式64次迭代初始化契约，4096环境、多核、每环境1024回合重新训练；v78没有完成过参数更新，未丢弃已训练优化结果。TensorBoard分开记录各次运行，监控器仅观察。


## 2026-10-25 — 多核4096环境恢复训练

- 按用户最新指令取消AGENTS中的单核/单线程要求，保留单Isaac实例。
- 启动脚本默认可用全部核心、按物理核心设置数值线程；GPU训练入口显式PyTorch线程配置和运行读回，未改变物理/奖励契约。
- v77保存77824样本/0更新检查点后退出；v78实际恢复4096环境、每环境1024回合，使用PyTorch6线程、全部12逻辑核心。未完成rollout重新采集，旧日志/模型保留。
- 19项TensorBoard/材质/回放/全手回归通过，shell语法、Python编译和差异空白检查通过；实际线程/亲和性和全部4096物体5.0/5.0材质读回一致。尚无稳定抓取/lift验收。


## 2026-10-25 — 物体强摩擦实验与显式材质回放

- 新增extreme_grip：静/动摩擦5.0/5.0，max组合、零恢复，显示粗糙度1.0；原rough_rod不变。
- 推理专用材质覆盖显式记录新旧参数，拒绝其他契约差异，不修改原检查点，也不放开训练续载校验。
- 15项材质与回放测试通过，单环境Isaac实际读回5.0/5.0/0.0，GUI运行最新已更新策略。尚无稳定夹持/lift成功；PPO优化暂停，耦合问题仍待解决。


- 新增全手物体过滤测力、五指/手掌分组、148维观测；旧四指垫观测保持原义。单环境与批量力反馈、奖励、阶段门控、局部过力检查采用一致数据。
- 新增逐关节动作范围；修复恢复指令越过新范围、扩展诊断环境索引缺键、坏接触数值导致整批异常三项问题，并添加失败后通过的回归。
- 新合同明确拒绝旧观测/动作模型静默续训；新PPO保持原物体位置和物理上限，2048环境、1024回合/环境已启动。
- 本次命令：`DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q tests/test_dh116_joint_envelope.py tests/test_dh116_free_grasp_regressions.py tests/test_dh116_tabletop_object_contact.py tests/test_dh116_tabletop_env.py tests/test_dh116_force_feedback.py tests/test_dh116_batched_task.py tests/test_dh116_env_expansion.py tests/test_dh116_tabletop_training.py tests/test_dh116_contact_reward_scale.py`：152通过、1预期警告。
- 实际Isaac：16环境8192步完成，37优化epoch，16训练回合完成/复位，无safety_stop；检查点权重变化、策略/优化器有限。初次smoke发现诊断索引KeyError，修复后上述第二次运行正常完成。评估仍是grasp_not_established，无抓取成功。
- 旧v71的mimic_error物理根因仍未证明；补充signed errors、命令、全关节状态、载荷诊断，保留原硬保护。历史全量失败未全部解决。后续持续观察目标规模实际训练与抓取成功率。

## 2026-10-25 — 自由夹持设计确认

- 核实v71因mimic_error停止；接触统计、力反馈和门控仍只覆盖四个指定指垫。
- 用户确认全手物体接触、新动作映射与新合同PPO；设计阶段完成，实现/离线/Isaac验证及真实抓取验收分别记录。
- 本次验证为monitor与进程状态、代码读取和git diff --check；下一步按实施计划修复及测试，尚未重新训练。

## 2026-10-25 — 项目快照 10-25：刚体高摩擦训练与吞吐优化

- 汇总尚未发布的10-23/10-24 PPO课程、控制、奖励、向量化、柔性物体、资产与测试代码，发布到独立 `10-25` 分支；既有分支及本地运行产物保留。
- 当前切回原位置实心圆杆；显式绑定手部1.0/0.8、物体1.5/1.2静/动摩擦并核对全部环境PhysX读回。属于训练假设，非实机标定。实际1024环境核验通过。
- 分离直接训练准入和抓取成功：两步有限安全检查后开始PPO，无接触经历允许学习，严格接触探测与最终成功标准保留。
- 新增真实单环境GUI策略播放，检查模型/物理契约，仅允许显示环境数变化；实际刚体场景已显示。运行管理脚本与模型保留本地，不上传。
- 全顶点净空计算采用等价高度公式。实际1024/2048/4096环境短测约774/804/742样本每秒，选择2048；保持物理、动作和奖励，显式环境扩容续训。修复容量探测在刚体预检拆分后缺少初始观测的问题。
- 实际首次更新检查点154624样本：权重改变、策略和优化器数值有限；KL保护在0.04处早停该轮，训练继续。当前仍无已验收的抓取成功，完整课程/实机能力待验证。
- 发布范围排除runs/archive/日志/缓存/密钥，新增graphify-out忽略规则；保留第三方资产许可证。
- 本次全量测试1632通过、73失败、1跳过（1预期警告）；仅修正旧approach测试中点时刻，生产时序未改。随后53项相关专项通过，剩余72项失败详见 `docs/validation_20261025.txt`，不标记全量通过。USD生成资产保持哈希原字节；差异空白检查通过。

## 2026-10-24 — 柔性PPO环境扩容续训（v67，本地未上传）

- 新增严格环境增长续训，契约JSON规范化消除tuple/list伪差异；保留策略/优化器/全局计数及原环境历史，新增环境零历史，禁止物理/奖励夹带变更和覆盖源运行。
- 修复零抖动批量复位重复全环境驱动写入与读回，子集复位一次完成；保留终止记录、RNG消费和独立复位，不改变物理与奖励。
- 新增 `--capacity-steps` 实际启动短测；1024与2048分别53.70/53.65 transition/s、3.61/4.08GiB显存，实际启动/隔离通过。选择1024以避免无采样收益的更长rollout；4096未测，不宣称最大容量。
- v66在67072样本/10优化epoch保存，v67从完整检查点继续；实际核验69120样本、新增2048、1024环境、保护未触发。每环境1024回合目标，原位柔性物体/20D动作/奖励/480Hz/64步rollout保持；扩容后首轮优化及抓取成功尚未验证。
- 验证命令：`DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q tests/test_dh116_env_expansion.py tests/test_dh116_tabletop_training.py tests/test_dh116_tabletop_vec_training.py tests/test_dh116_contact_reward_scale.py`（192通过）；`DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q tests/test_dh116_gpu_batch_reset.py tests/test_dh116_tabletop_vec_env.py tests/test_dh116_batched_task.py`（33通过，1预期溢出警告）。编译/diff检查通过。实际源检查点CPU扩容加载策略/优化器逐项相等，实际Isaac扩容恢复计数已核验。

## 2026-10-24 — 原位置柔性PPO集成（进行中，本地未上传）

- 新增批量CUDA Cosserat及原GPU PPO柔性物体适配器；保留原中心/手姿态/20D动作/奖励，用固定材料区域替代刚体底部和滑移指标。
- 法向接触观测与含摩擦COM反力分开；局部区域均值按材料长度积分，避免离散节点偏置。
- 原始蒙皮网格提供直接points发布兼容模式；已有GUI仅显示静态rest pose的问题尚待实际窗口验证修复。
- 83项相关回归、3项适配器测试通过；实际2环境Isaac启动检查通过/退出0。短PPO采样12条主动停止用于性能定位，尚无更新；近手查询经11项CUDA测试与实际手姿态512杆基准验证后，已启动v66的512环境/每环境1024回合新策略训练，已核验首批32,768条后的5个优化epoch，后续采样正常且无保护触发；未修改训练参数，不宣称抓取成功。

## 2026-10-24 — DeformX 原始柔性资产导入（本地未上传）

- 新增可搬移的原始橙色USDC资产包及Isaac USD入口，保留网格/骨骼/材质及许可证；删除引用资产附带的灯光影响，不添加刚体物理。
- 新增按Cosserat世界坐标驱动的独立骨骼动画，验证输入帧正交性与有效性；单环境运行入口加载全部34手碰撞体，将积分冲量转换到真实body COM后反馈PhysX。局部抬升测量固定材料区间，使用实际动态半径。
- 34项资产/运行时/材质/初始化测试通过；29项运行时/后端/物理测试通过。资产与接触集成代码复核通过。保留原刚体入口默认行为。
- 实际Isaac默认初态2步有限且无手接触（新半径导致3.9–4.4 mm间隙）。仅线X偏移4.5 mm的独立接触验证完成12步/0.025 s：真实接触、0.7626 N最大body步均反力、2.484 mm最大节点运动、5.42e-20最大冲量平衡残差；51.31 s墙钟时间、退出码0。命令：`DH116_NUM_THREADS=1 bash scripts/python.sh scripts/demo_dh116_deformx_wire.py --headless --device cuda:0 --steps 12 --contact-probe-offset-x-m 0.0045 --output runs/deformx_import_20261024_contact`。
- 修正验证程序默认关闭等待，使用训练入口同款post_quit/skip_cleanup并记录退出码。首次smoke仿真/保存已完成后关闭等待>30s，由SIGTERM退出；后续接触验证正常退出。
- 旧v65检查点2037760样本/301epoch/每环境10回合已保存并校验。按更换物体目标停止旧刚体训练并保留回退资料；尚未接通512环境柔性PPO，未宣称夹持/提升成功。记录原蒙皮拉伸/剪切近似、无自碰撞及罚函数摩擦限制；无commit/push。

## 2026-10-24 — 总接触力10 N对应5分，续训v65（本地未上传）

- 按用户指定扩大接触奖励线性区间：势值`5*clip(sum(pad_forces)/10,0,1)`，实际奖励为势差。标量和批量共用实现，其他奖励/物理/保护不变。
- 奖励契约升级为`object_support_progress_v4_10n_5`；显式允许仅该奖励版本迁移，其他契约差异仍拒绝。
- 237项测试通过，覆盖比例/饱和/力下降、无接触抖动刷分、标量与批量一致性、续训及保护。编译/diff检查通过；溢出保护测试产生一条预期警告。
- v64保存1710592样本、256优化epoch、每环境9回合后退出；v65恢复这些模型/优化器/回合统计，维持512环境、每环境1024回合目标。未优化的部分rollout不重放。后续核验1744384累计样本、261优化epoch，首轮新奖励优化已完成且无保护停机；不宣称10 N物理能力或抓取成功，无commit/push。

## 2026-10-24 — 每环境1024回合续训（v64，本地未上传）

- 用户明确512环境、每环境1024训练回合。新增显式回合目标，所有环境达标后在完整rollout更新结束时停止训练；继续使用64步更新周期，普通失败计入、预检/评估排除。
- 检查点保存已完成回合统计；旧检查点通过计数匹配的最终monitor迁移。恢复模型与优化器时尊重请求设备，避免GPU入口续训退回CPU。
- 161项训练/检查点测试通过，编译/diff检查通过。v63在229888样本、31优化epoch、每环境1回合保存退出；v64以该检查点和回合计数启动，活动指针更新，实际状态见实时monitor。
- 原物理/控制/奖励/硬保护保持，单Isaac和CPU0/单线程。未宣称真实抓取成功，无commit/push。

## 2026-10-24 — 512环境直接进入PPO（v63，本地未上传）

- 新增显式 `--direct-training`，跳过三批完整准入回合，保留初始复位安全、启动接触/隔离/驱动检查、逐步安全保护及最终评估。直接模式接触检测全部满足后提前结束预检。
- 128项向量训练测试通过，含失败回合学习、初始复位异常、逐步硬故障和参数冲突；编译/diff检查通过。默认准入模式不变，不修改抓取参数或奖励。
- v62在0PPO更新时停止，v63以相同512环境预算重新启动，活动指针更新。实际Isaac接触预检8控制步完成，进入PPO并已采集3072条样本，首次核验优化数仍0；不声称尚未发生的优化或抓取成功。历史记录保留，无commit/push。

## 2026-10-24 — 512环境GPU扩容（本地未上传）

## 2026-10-24 — 按用户要求恢复512环境训练流程（v62，本地未上传）

- 启动v62，沿用v54抓取控制/物理/奖励/硬保护，预算1048576 transition、checkpoint8192。v54没有学习更新或检查点，本轮全新策略；不导入工装试验大力矩参数。
- 活动运行指针更新，保持单Isaac及CPU0/数值单线程。预检/准入通过后自动开始PPO；启动阶段不能记为已有策略更新。本次无控制代码更改、无重复软件测试、无commit/push。

## 2026-10-24 — 单指固定工装加载试验（本地未上传）

- 新增独立Isaac加载入口、固定无摩擦测力板、指垫碰撞隔离、逐级力矩上限对照及480Hz持续载荷验收；记录实际接触、关节/联动误差和参数读回。PPO控制未改。
- v59/v60/v61完成56次加载（含双重复），18安全完成、38联动保护停止、0持续14N。64迭代改善低载稳定性但未建立额定持续输出；2Nm瞬时最高10.704N不算保持能力。关闭休眠无改善，稳态载荷传递尚未解决。
- 相关20测试通过、编译/diff检查通过、独立审查修正采样遗漏/容器碰撞节点/控制频率/日志问题。v54预检中停止、0PPO更新；所有历史记录保留，试验参数未部署。报告docs/dh116_fingertip_fixture_20261024.txt。

## 2026-10-24 — V2 加压几何与参数读回矫正（本地未上传）

- 修正旧闭合方向26–43 mm/rad向上漂移，增加基于源USD的直径方向与手掌中点补偿；实际动作仍受既有净空/跟踪/限速约束，日志记录补偿裁剪及误差。
- 新显式profile支持0.1–4 N每手目标载荷、480 Hz刚性线性mimic候选；CPU/GPU控制一致，旧默认及合同不变。实际PhysX逐实例核对驱动上限，记录质量/惯量/限位；隐式PD力矩与接触法向力明确标注，未冒充实机标定。
- v50驱动与v51/v52几何/求解器对照均未获得稳定离桌，不能将阶段到达计作成功。v53 GPU8实际24回合准入通过、0成功；相关253+56项测试通过，1 skipped，编译/diff检查通过，独立审查发现的旧.15 Nm读回回归已修复。
- 新v54启动512环境流程，1048576 transition预算、8192存档、全新策略；512自身准入通过前PPO更新为0。v47已停止，所有历史资产/检查点保留。详细限制和命令见docs/dh116_v2_correction_20261024.txt。无commit/push。

- 新增克隆GPU物理及批量任务入口；原CPU入口保留。保持奖励、控制、阶段和硬保护，单Isaac/CPU0/数值单线程。
- 定位GPU薄壁凸包CPU回退伴随零接触；归一化烘焙坐标并以USD缩放恢复原世界顶点，独立几何误差最大1.675e-11m，实际接触恢复。未改管壁、质量、惯量和摩擦参数。
- v44两环境6回合、v46三十二环境96回合准入安全；512容量实测约3.7GB显存。v47已启动512环境，1024只手检测到接触，仍在接触/物理准入，新增PPO更新0；准入通过后自动训练，尚无512训练完成或抓取成功结论。
- v41停止保存17764步/118优化epoch，v47配置同合同权重导入，优化器和计数重建。新增差分/边界测试，相关208项通过、审查通过。当前速度约107 transition/s，主机端/仿真吞吐仍限制加速。
- 实时状态：`runs/tabletop_gpu_20261024_v47_env512/monitor.json`；详见`docs/tabletop_gpu_20261024.md`。未上传或删除旧数据。

## 2026-10-24 反作用力反馈与抓稳后提升（本地未上传）

- 新增可选20D/136D反作用力PPO合同：最后两维为各手目标载荷，测得反力不足时限速闭合、过量回退；滑移时有界增压，暴露全部反馈状态与实际载荷。旧模式和旧检查点保留。
- 用实测载荷/速度/滑移连续稳定替代2.3秒自动lift，6秒未建立夹持视为学习失败；提升失压或滑动时暂停脚本上移，原硬保护和真正成功标准不变。准入验证安全复位/close探索，不要求策略先成功夹持。
- v35/v36定位驱动上限并非饱和瓶颈。v37..v40串行Isaac对照验证实际闭环、联动传力、阶段门槛及暂停重新加压。v40四回合无硬故障、三组到lift，但真实抓取0/4，未离桌。不得把诊断当作已训练策略。
- 新v41为4环境nf400严格保护的全新PPO，先准入，计划65536步/8评估；12回合严格准入已通过，首次核对3584步/14优化epoch，无硬保护停止；2048步检查点重载合同/维度/有限权重检查通过。初始力均值约.1225N，对应诊断可建立接触的范围；不会重置resume权重。
- 相关回归388通过、追加滑移时序1通过、审查修正后154项通过，编译/diff通过；未上传、未删旧结果，保持单实例/单核。

# 更新与阶段记录

## 2026-10-24 — 高摩擦吸管 PPO 试验（本地未上传）

- 新增可选高摩擦对象材质：静摩擦1.2、动摩擦1.0、恢复系数0、max摩擦合成；只对对象碰撞体绑定，保持视觉材质及其他物体材质定义。对象与桌面的有效摩擦也相应提高。
- 启动前验证全部实际PhysX形状的摩擦系数，参数纳入严格检查点合同，默认材质行为保留。相关软件验证222passed。
- v33实际完成65536步/472epoch，最终评估0/8；v34单实例4环境已通过物理材质核验和12个准入回合，实际进入PPO训练，尚未据此宣称抓取成功或性能改善。旧资产、检查点与日志保留，无上传。

## 2026-10-23 — PPO 调整夹点/刚度与中空吸管（本地未上传）

- 新增可选20D动作/125D观测，PPO控制手指、每手XYZ小范围位置和夹持刚度；真实接触点、物体底部高度及控制状态可观测。保留原物理保护和不要求四点接触的成功标准。
- 新增直径两侧的几何初始化、严格资产/初始化/动作合同，以及真实中空吸管USD生成器。对象外径16 mm、壁厚.5 mm、长.5 m、质量10.956g；32段凸碰撞保留通孔，质量/惯量已由实际PhysX加载值复核。新对象不沿用原120g成绩。
- 修复adaptive启动时再次预张开触及USD下限；v32失败发生在学习前。v33已通过12个实际准入回合并进入4环境PPO，首轮核对1388步/3epoch、无物理停止；尚未完成训练预算与抓取验收。旧v31主动停止并保存12,736步/103epoch，新模型不继承旧12D动作输出。
- 本次相关软件回归484passed；真实SB3假物理64步/2次更新和存档重载通过。实际抓取与长期训练仍待运行证据；没有上传源码或runs。

## 2026-10-23 — 共享物理时钟的多环境 PPO（本地未上传）

- 新增协同子步VecEnv及单实例多场景后端；独立碰撞组、同场景接触过滤、无隐藏推进的独立复位、终态观测/物理测量保留。
- 新增每env准入/配额评估、总步数检查点、严格恢复和约2秒更新的实时监控，报告物体高度、滑移、力与联动误差。
- 实际2env探针6回合安全；v27 4env训练8192步安全，但确定性评估3env联动失败。慢lift仍失败、禁lift安全；有限mimic柔顺对照跑完4env原失败前缀及完整延伸回合。只vector启用200 s⁻¹/阻尼1并写入contract，.02 rad硬阈值不变。
- v28实际完成8192总采样步/69优化epoch，随后8次确定性评估全部安全超时、抓取0/8；summary为completed_budget、exit0，passed与grasp_qualified仍为false。这是本次短程训练/评估安全证据，不是抓取成功。
- v29在累计33996步/本轮新增25804步/287优化epoch时因env2右finger22联动误差−.0247419 rad停止，exit2；检查点已保存，预算及最终8次评估未完成。[停止监控](runs/tabletop_vector_train_20261023_v29_4env/monitor.json)与[记录](docs/tabletop_vector_ppo_20261023.md)保留；介入依据是实测硬故障，不是回报低。
- v29原动作回放144步精确复现故障；100ms稳定等待虽901步无硬故障，却使3环境无抬升、另1个仅抬.208mm，拒绝该方案。仅加快预警恢复至.3 rad/s仍145步超限，也已拒绝。两者均未部署。480Hz/60Hz诊断完成901步、4环境安全超时、实际抬手约45mm；随后原v29策略在480Hz下8次确定性评估全部安全超时，抓取0/8，控制帧峰值mimic约.01018767rad，exit0，不宣称长期稳定或抓取成功。
- 生产入口新增480Hz物理步、保留60Hz控制；seed验证只允许240→480Hz并保留原来源。显式weights-only迁移严格保持其他契约，保留策略参数但重置优化器、计数与资格，记录源模型33996步及SHA。未修改奖励、硬保护、夹持控制或成功标准。
- v30通过准入后实际采样612新步、执行5优化epoch，随后env2训练回合3右finger22误差约−.020618rad，physical_safety_failure/exit2；检查点已保存，262144步预算及最终评估未完成，当前训练停止。该反例说明480Hz回放和确定性评估安全未覆盖训练探索，不能宣称问题解决。[停止监控](runs/tabletop_vector_train_20261023_v30_4env_480hz/monitor.json)及最终报告已保留。
- 外部运动学锚点原型经初始化修正后报告正确速度和约17.3µm最大跟踪误差，但同v30动作第152控制帧仍mimic超限，未部署。原后端的nf400候选分别重放v29/v30失败前缀并以零动作延长至901控制帧，均4环境安全超时、完整root_offset=.05m、物体未升；60Hz采样mimic峰值分别.0163013/.0184418rad，并非全子步峰值。
- 新增显式200→400 s⁻¹权重迁移：限定480Hz物理/60Hz控制、阻尼1，其余合同含布局/backend/seed严格一致；与物理升频迁移互斥，保留源SHA/合同/计数，策略权重不变、优化器和新计数清零，不继承成功资格。
- v30权重的nf400闭环评估在149控制帧/596总transition因env1 mimic误差.0215569431rad失败，exit2、physically_safe=false，计划8回合未完成。记录力约.8047N、最大关节速度约.195rad/s且数值有限，不能据此忽略原.02rad阈值。v31 nf400训练禁止启动，硬保护、奖励、控制和成功条件保持原值。
- 有限mimic故障立即终止/复位的临时原型在2869向量控制步中处理20次故障（env1/2/3，故障峰值.02106757rad），29次局部复位隔离检查均精确不变；无新增PPO采样或抓取资格。正式候选保持默认关闭，保留原失败−25奖励/终态，逐剩余子步校验；末子步复位持有独立pending证据，下一真实物理步任何guard失败均全停。CLI仅允许480Hz/60Hz与nf200；显式旧权重迁移只新增训练失败处理contract字段，优化器与计数重新开始，评估仍严格失败。
- 正式模块故障复位探针完成2259向量控制步/10实际故障（目标9，同帧并发达到10）、17次精确隔离检查，含1次zero-tail后续物理步校验，exit0；仍不授予抓取资格。主线程相关软件检查390 passed。v31 protected正式CLI随后启动：4env、480/60Hz、nf200、262144新transition预算、eval8、checkpoint2048，显式从v30复制权重并重置优化器/计数。首次核对为admission、0采样/0epoch，后续以 [实时monitor](runs/tabletop_vector_train_20261023_v31_4env_protected/monitor.json) 为准；nf400方案未部署。
- 本轮相关专项246 passed、1 skipped；补齐布局迁移限制的3项RED/GREEN回归后，定向49 passed。未再次运行完整专项，未跑全仓库；旧v25最终36268步失败记录保留，无commit/push。

## 2026-10-23 — direct PPO 联动恢复与串行续训（本地未上传）

- 主关节恢复目标改为 q_master + mimic_error / ratio，并按命名映射、耦合限位和原指令速率执行。仅此修正的 v23 在3754步仍失稳；后续同动作对照验证从动限速约束问题，direct 从动 solver limit 改为3 rad/s，主动仍0.3。原硬保护不变。
- 最终 v24 实际8192采样步、244优化epoch及5次确定性评估无物理故障，exit0；抓取0/5、物体未离桌，不声称抓稳。v25已从该检查点启动自动续训，最大8轮×65536新增步，物理故障不自动重试。
- 新增诊断回放入口、direct续训控制/seed参数与SHA验证；两类控制语义进入checkpoint contract。九文件专项186 passed、1 skipped；编译/diff通过，未重跑全量测试。记录在docs/tabletop_mimic_recovery_20261023.md。

## 2026-10-23 — 直接 close 与自主关节抓取（本地未上传）

- 用户明确取消训练中的 approach 和四点接触门槛：实测 handoff 状态作为新 close 初态，恢复命名关节/物体位姿与速度，真实物理稳定 12 子步后重新测量接触；不恢复接触缓存或成功标签。
- PPO 改为有界 12D 关节目标；108D 观测、固定目标包络与速率限制、预警主关节卸载、每手小幅位置调整。旧饱和残差模型不迁移。
- close/hold 定时后尝试 lift；资格按物体离桌、速度及相对滑移保持判断，无 pad 数量要求。奖励使用接触/高度进展和真实离桌保持，防止阈值以下挂机刷分；无支撑时不收取已抓稳后的滑移惩罚。
- 真实 v19 direct-close 探针3/3准入；v20 object-stability 探针3/3到达lift且安全，但无离桌成功。v21 在1048步右食指联动超限停止；v22 预警提前至0.004 rad，仍在1957步超限（−0.020175 rad），exit 2。检查点保留，当前停止训练，物理稳定性问题尚未解决。
- 奖励 v3 仅惩罚超出20 mm容许值的滑移；object-stability 的 close 实际成功正确写入通过报告。完整回合回报排序与报告回归均已验证旧行为失败、新行为通过；七文件专项134 passed、1 skipped。v18停止在第6轮，周期检查点337920保留，无该轮final summary；不称干净结束或夹持成功。

## 2026-10-23 — 接触恢复与自动续训（本地未上传）

- 接入预备姿态、物体 XY 跟踪、接触丢失后的有界下降、全轴耦合警告暂停和 104D/v4 观测；预抓取交接检查使用物理步后实测状态。原物理和抓取验收阈值保留。
- 新增显式权重迁移与串行续训入口：安全预算结束自动恢复同一模型，close 达标后训练 lift；物理故障不自动重试，预算耗尽不标成功。
- 实际 v16 累计 81,920 步仍 close 0/5；v17 权重迁移后新计数 8,192 步、307 epoch，评估 5/5 进入 close、成功 0/5。v18 已启动后台自动续训，最多 8 轮，每轮每阶段新增 65,536 步。尚无稳定 hold/lift。
- 新鲜专项验证 116 passed、1 skipped；全仓库测试未重跑。各 Isaac 实验串行、单核心/数值单线程；runs/logs 保留本地，无提交或上传。

## 2026-10-23 — PPO 训练准入、分阶段课程及真实短程更新（本地未上传）

- 修复提前接触后 approach 卡死；阶段完成与完整抓取成功分开。独立名义参考、可配置残差半径与残差变化率限制替代累计残差，94D/v3 观测包含滤波及阶段隐藏状态。
- 增加测量进展奖励、零残差/目标阶段随机探索准入、阶段晋级评估、周期检查点、契约校验、保存模型评估和按阶段续训。中断保存状态，物理失败（含评估）和准入拒绝以非零状态退出；课程未学会不伪装成进程崩溃。
- 实际 Isaac v5：512 采样步、20 优化 epoch，approach 5/5、close 0/5；v6 真实恢复至累计 640 步、25 epoch，close 0/5，未进入 lift。两次滤波配置运行未触发安全停止，只是短程接线/探索证据，未形成成功抓取。
- 保留 v2/v4 mimic 超限失败。v4 安全停止时累计 611 采样步、仍 20 epoch，实际退出码 2；最后不足一个 rollout 的采样没有冒充一次参数更新。
- 新鲜验证：五文件专项 101 passed、1 skipped；编译、diff 检查通过。全仓库测试未重跑；没有修改摩擦、成功阈值或使用物体附着。所有 Isaac 实验串行、单核心/数值单线程。
- 未完成：当前交接高度下的四点持续接触、hold/lift、正式长预算训练和泛化验收。详见 `docs/tabletop_training_20261023.md`。

## 2026-10-23 分支 `10-23` 开发快照

- 基于 `10-22` 创建新分支，保存当前源码、测试和方案；不覆盖历史分支，不合并。
- 新增 hand-only 桌面 MPC 参考残差 PPO 入口、奖励/阶段门槛、mimic 与关节保护；保存姿态同步、候选筛选、预载/恢复和既有教师学生接口修改。
- 修正强制下降：逐子步位移、接触停止锁存、桌面余量裁剪；张手到参考姿态平滑过渡，默认使用 PGS。没有放宽安全阈值或改变奖励权重。
- 实际验证限于下降/接触保持：两次约 15 秒 Isaac 运行无关节超差或穿桌，最终任务超时；最低净空 8.86 mm，记录最大 mimic 误差 0.00613 rad。未完成 close/lift、未启动 PPO，不标为成功抓取。
- 上传前专项命令：`bash scripts/python.sh -m pytest tests/test_dh116_tabletop_task.py tests/test_dh116_runtime_mimic.py -q`，9 passed、1 skipped。未重新验证全仓库；历史测试结果不作为本次通过证据。
- 排除 runs/archive/日志/缓存/密钥，保留本地原件。姿态 summary 与绝对资产路径仍有本机依赖。后续需完成闭合、提升及训练准入改造。

## 2026-10-22 — `10-22`分支阶段快照

- 新建`10-22`，上传到原GitHub仓库`ssdsj12/dexterous-hand-dh116`，基于当前`10-21`代码和既有提交历史；不覆盖原远程分支，不创建PR或合并。
- 汇总改进：CUDA完整Cosserat杆/网格接触与机器人反力、12D并行手部MPC与独立Panda搬运、实际v2专家记录与蒸馏接口、10ms执行窗口失载纠偏、恢复与指腹几何诊断。线性化MPC不是完整柔性杆候选rollout，CPU控制/源几何环节仍保留。
- 实际阶段仍为V23closePASS、完整1秒holdPASS、liftFAIL；没有整线离桌或合格成功轨迹，没有正式完成教师/学生训练。本次上传不启动新仿真、不改变物理参数或门槛。
- 本次真实验证：README所列CUDA/采集/策略/恢复/预载专项99passed in5.19s；全量`DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest tests -q --tb=no`得到874passed/71failed in29.03s，明确保留已知未通过项，不称全量通过。diff检查通过；不是仅引用旧测试结果。
- 上传源码、测试、接触配置与网格、既有资源；排除runs/archive/日志/缓存/密钥/1332，本地原件保留。部分配置仍绑定本机绝对路径/SHA，异机运行需重新定位校验。推送后由远程分支SHA与本地HEAD一致性确认上传结果。
- 下一步仍为经确认的掌心/整臂位姿调整与真实lift，随后才是成功专家与正式蒸馏；上传快照不代表总体训练目标完成。

## 2026-10-22 — V23终态与累计3mm局部姿态筛选

- 新增只读GPU受力分解及CPU参考核对：3个真实保存状态×3个明确标注的速度工况×6通道，共54项，节点/反力最大误差2.23e-16N，力矩最大3.47e-18Nm。相同CUDA几何输入，只核验受力/分配，不声称独立距离或全积分一致。
- 入口姿态重力0.98621N；即使反事实设定双手向上5mm/s、杆暂静止，摩擦约+0.098N，手对杆合力仍−0.437N。原CPU与GPU均为相同粘性正则化库仑摩擦，未丢失该项；未改摩擦、姿态控制器、增益、阈值或运行源码，无新Isaac/时间推进/成功标签。
- V23真实终态16922：closePASS/holdPASS/liftFAIL，右非指腹finger13源间距界0.999886mm<原1mm；6区域通过。反力已施加并执行PhysX，随后检查失败；16922仅为不完整失败步，16921为最后完整有效记录。滑移1.413684mm、wire未抬升。无新Isaac或相同配置重开。
- 实际完整加载170区间（21close/100hold/49lift）链检查通过，末区间失败，formal source正确拒绝；没有成功模型或正式训练。
- 原USD+实际全18D关节的离线支持平面复核给出约1.148mm，保留原运动包络；不是逐位实测PhysX证书，不据此降低1mm或改变运行器。
- 已批准Panda固定局部候选按最初V20点位累计3mm约束，保持原加载深度和所有source/pad检查。4个对齐目标试验3个静态通过、每手82个加载深度保持姿态扫描仍未找到向上承托的对齐候选；不把它们标为成功抓取。
- 只读双掌下移2mm对照的左finger13桌面余量0.966722mm失败，未部署或导出新初态。掌心/整臂调整需下一设计选择，未擅自改变Panda固定约束、摩擦参数或MPC架构。
- 本地诊断脚本/报告位于V23及`dh116_local_pinch_alignment_20261022_v3`；`support_scan_v2.json`取代漏根的初版诊断。脚本均实际exit0，编译与diff检查通过；未重复未改动的历史测试，不宣称全仓库通过。无commit/push。

## 2026-10-22 — V21实际lift失败；GPU单关节复现位置冻结

- V21终态27063/lift0.2363s：closePASS/holdPASS/liftFAIL，右thumb分离1.2154mm/s超过原1mm/s；仍有.9544N载荷，滑移.1852mm。PID157969/session21797已结束，无成功蒸馏。
- 终态实际回放272区间（148close/100hold/24lift）状态/时间链通过，末6.3ms明确失败，formal loader拒绝；不把持有载荷或阶段进入当成成功。
- 只读定位到Panda目标改变但部分实际关节角不动。独立GPU Isaac单关节1000步对照：原100µs/16次TGS位置迭代位移0，1次时位移.000476837rad（期望.0005）。仅为数值机制复现，不是整机/真实抓取验收。
- 独立诊断均已退出、数据和源码保留，未同时开多个Isaac。未改生产控制器、物理杆/资产/力矩/成功条件。下一步先原参数500步窗口接线基线，再单独验证迭代数修正的完整真实抓取。
- V22原参数接线基线自然500步结束，原逐帧检查通过，5个close完整区间/真实窗口来源链PASS；限长未完成技能，保持诊断标签。V23从同一14699状态、相同现有源码启动完整试验，位置迭代改1、预算200000，其余实际manifest一致。
- 两组实际前400帧原checks均PASS；16次迭代的双Panda七轴均零变化，1次各轴恢复非零变化，报告/复现脚本保留V23本地run，不是lift成功。V23实际decision300/400出现当前载荷恢复、前窗口仍欠载的情况，未误锁定，窗口修复的真实接线已核对。
- 唯一V23 PID167430/session76710实际2025/time.2026s通过close，在12025/time1.2026s完成原1秒hold进入lift；最新12348有效lift，elapsed.0323s、滑移.2859mm，站点rise仅约.0012/.0016mm，尚未抬升或整线离桌。新回放122区间（21close/100hold/1lift）原时间/状态链通过，2.6ms交接保留，不正式训练。可用备份仍是2024完整物理状态/SHAfb520d...16c365、不带phasecredit；额外hold12024备份因close-only校验拒绝、未创建，无校验修改或主实例中断。
- CPU固定core0；启动脚本正确设数值线程1，Isaac改写的OpenBLAS环境12不是live线程池测量，不据此重启或声称全框架单线程已证明。未安装额外诊断依赖或修改wrapper。
- 无commit/push，未重复未变化的专项；原138项结果不冒充本轮物理验收。

## 2026-10-22 — 原V21完成1秒hold进入lift，补齐原预载反馈的区间漏检

- 同一旧V21实际14700步/time1.4701s通过原100ms稳定夹持，24700步/time2.4701s完成原1秒保持并进入lift；最新24792有效lift，elapsed.0092s、材料滑移.0365mm。进入lift不是线缆抬升成功，尚未整线离桌，不正式训练。
- 旧实例未重载，不把它的进展归因于新代码：原manifest无新反馈标记，原采集无window字段。旧运行器/预载模块已保留源码及SHA备份。
- 小范围修复原“实测失载恢复调节”的约定：积累实际物理帧的minload/maxgap/known，在原MPC决策消费；90%原门槛、全部物理成功标准、力矩/增益/资产/GPU杆不变。
- 完整采集新增`goal.preload_window`及真实来源区间；新运行身份为`completed_physics_window_v1`。不改12D控制、Panda分工或旧网络维度。
- TDD新增8项预期RED，focused27通过；相关138 passed in7.52s，compile/diff通过。新反馈仍仅离线验证，未宣称全量全绿或新模式物理成功。
- 实际原V21回放249区间（148close/100hold/1lift）链验证通过，边界0.1ms如实保留，仍诊断资格。Panda搬运、hand_mpc_hold保持12D手部目标，不含Panda动作列。保存最后有效close14699，不导入成功credit；原PID继续实际lift，无新实例或提交/上传。
- 原14701物理帧/148close决策的离线对照中，旧掩码与实测全匹配；新窗口会补识别32次区间低载荷/接触问题且只读取决策前样本。这是反事实掩码，不是新动作/物理轨迹，不升级训练资格。

## 2026-10-21 — 局部指腹对齐静态筛选与V21实测启动

- 采用已批准≤3mm方案的Panda固定候选：左右真实mesh轴向错位4.163→2.588/1.755→1.029mm，最大点位移动1.672mm，距离差≤2.871µm；完整源几何及6个原pad区域静态通过，不是物理抓取成功。
- 仅拇指/食指关节调整，Panda关节/双掌/目标和非夹持手指保持；原USD、rod、物理配置和全部成功门槛不变。保留原完整杆动态、速度和驱动偏差，不继承阶段资格。
- 现有恢复/IK专项18 passed in1.01s；未修改生产代码，不宣称全仓库通过。生成状态保留本地runs，未提交/上传。
- 单实例V21从新close状态启动，PID157969/session21797，需实时复核进程和真实阶段；不重放approach。不采用离线掌心微调提案，也未新增Panda反馈控制器。
- 实际首帧源几何/四点接触通过；核对至11683步仍有效close，四点载荷约.898/1.038/.953/1.009N，尚无hold/lift成功。物理时间1.1684s不是hold完成。最长基本合格区间98.4ms，左拇指短暂<.9N重置原100ms计时；10ms控制采样与100us资格检查的时序差异已复现，待选窗口纠偏尚未实施。后续欠载已有实际释放保持，不称永久卡死。
- 保存1755及较新11517步完整实测close状态，未停仿真、不带phasecredit，旧备份保留。新核验101个完整v2 close区间（累计1.01s、末outcome10100），12D/时间/token/状态链通过；未终态或成功，仍为partial_action_diagnostic，正式源资格拒绝。同实例继续，不正式蒸馏。
- 尚待实际close/hold/lift及整线离桌，随后才能采集成功专家并正式蒸馏。
- 只读性能分解：当前每步约.35s，杆/接触约.10s，源几何前后各约.10s。静态CPU微基准3×10次：源几何24.8ms、已有缓存3.7ms、6pad区域64.6ms，静态筛查均通过；与单核Isaac竞争，不是整机同物理阶段性能验收。未启用live缓存或为此单独重载，不宣称已提速。

## 2026-10-21 — V20 lift实际失败，保留成功阶段及失败区间

- V20真实终态：close=PASS、hold=PASS、lift=FAIL。15805步/lift0.2701s，右拇指分离速度1.0943mm/s超过原1mm/s判据；仍有0.9295N载荷，gap约-.2973mm，材料滑移.2437mm，未达到整线抬升。
- 旧PID145546/session15643已结束、没有新Isaac；不是观察超时重开，也未降低判据。
- 终态实际v2回放160区间（32close/100hold/28lift）时间/状态/token链通过，末段0.1ms明确失败，正式训练资格被拒绝，不生成成功模型。
- 只读FK/实测跟踪显示目标上行、实际掌心下沉，起点目标/实测掌心相差约1mm；报告速度与位置差分不一致。下一步优先选择搬运位姿反馈或已批准的局部夹点方案，未擅自改步长/力矩/载荷/阈值。
- 无运行代码更改、无重复未变化测试，无提交/上传。

## 2026-10-21 — 原1秒保持完成，同一V20进入lift

- 第3104步开始hold，第13104步/time1.3105s完成原1秒实测保持并进入lift；没有重启、缩短持续时间或导入阶段成功。
- 最近13557步有效lift（约45.3ms），四点接触有效、材料滑移约0.110mm。wire站点上升仍只有约0.0016/0.0003mm，尚未满足抬升或整线离桌要求。
- 完整回放已读取32close/100hold/4lift共136实测区间，时间/token/状态链一致、区间检查通过；原阶段交接0.5ms短区间如实保留。仍为partial_action_diagnostic，不正式训练。
- Panda实际发布搬运目标，DH116保持12D夹持目标；不把手部保持伪装成新MPC规划。原姿态继续验lift，微调候选尚未部署。
- 本轮无运行源码改动，未重复未变化的测试，无提交/上传。

## 2026-10-21 — V20真实接触预载验稳通过，继续同一实例hold

- V20在3104步首次进入hold；最近4117步累计保持0.1013s，材料滑移0.0219mm。尚未完成1s保持或lift，不训练为完整成功源。
- 完整回放入口后续已读入32个close及41个hold区间，手部动作始终12D，Panda固定；hold所有权为hand_mpc_hold，不伪造新MPC规划。实际72个10ms区间和1个交接0.5ms短区间均保留真实时长，时间/状态链通过；未完成lift的正式源资格仍被拒绝。
- 保存close3103完整物理初态（SHA7ec4c4fd47c128b99af65b0b2568df0fafb00238dedb2dc1d3cd1d4181d26b9c），无成功credit。原实例不停止、不重放approach。
- 用户确认局部四点对齐备选（每点≤3mm）。复用现有MPC/源几何做只读筛选：大幅目标未来姿态左食指不通过；小幅0.111/0.072mm静态16姿态通过，但未改善双手实际mesh最近点对齐，不直接部署，也不称物理成功。报告保留本地run，设计已记录。
- 无运行源码修改，未重复未变化的测试；上一轮195项相关通过继续有效，不宣称全仓库通过。无提交/上传。

## 2026-10-21 — 同步执行中的 MPC 目标，去掉未执行名义坡道

- 按用户确认，TouchFirst/完整序列接受MPC实际载荷目标，mounted runner在MPC模式传入。第一帧即同步，仍须实测触碰/对指；非MPC默认坡道不变。
- 保留接触50ms+50ms、实际夹持100ms、hold1秒和全部载荷/滑移/抬升/整线离桌要求。测试覆盖目标不能证明接触、低载荷不能提升和无效目标不推进时间。
- 保存V19有效1752步后协作停止；V20单实例从close物理状态续验，不继承阶段资格。1062步已进入双手preload、controller目标直接就绪；最近1982步四点实际有效、载荷约.889/1.009/.923/1.028N，左手因低于原0.9N要求恢复调节，右手保持。19个完整v2区间，尚无hold/lift成功，不正式训练。
- 相关195 passed in14.26s，启动后再次195 passed in11.32s，编译/diff通过；全仓库未宣称通过。未提交/上传。

## 2026-10-21 — 移除额外预载门槛，保持原物理判据

- 预载暂停与原GraspEvidenceGate的90%目标载荷对齐，不再要求额外95%；接触/载荷不足即恢复调节。原成功、持续时间、碰撞、滑移要求不变。
- V18保存有效500步后正常停止；V19从该close状态继续，最近607有效步、双手目标保持、四点连续有效57.3ms，两手均进入oppose，载荷约.912/1.017/.911/1.027N，已写6个完整v2控制区间。仍未最终稳定抓握/hold/lift，不正式蒸馏。
- 受影响专项184 passed in13.54s；此前全量858/71不冒充此修改后的全局通过。编译/diff通过，无提交/上传。

## 2026-10-21 — 接触边界的1ms实测位姿速度窗口

- V17四点实际有效接触已有，但孤立100µs差分尖峰重置持续计时；保存有效1000步后正常停止，不授予成功。
- 新增因果位姿速度窗口（当前10步=1ms）：位置和物理步长不变，首帧保留报告twist，状态切换清空历史，记录实际估计区间。保留真实快速运动，不改接触/滑移/碰撞门槛。
- V18单实例从保存的close状态继续；最近93步有效，连续四点有效7.2ms，载荷约.939/1.058/.938/1.045N。仍无hold/lift成功，不正式训练。
- 专项183 passed in15.12s；最终全量858 passed/71 failed in58.93s，同旧失败集合。编译/diff通过，未提交/上传。

## 2026-10-21 — 修正接触边界的虚假运动与恢复入口

- 实测V15右Panda位置保持，但报告qd映射出的拇指速度约3.5mm/s，位姿差分约.005mm/s。增加显式一阶pose_delta接触边界，保存实际差分与原报告twist；首帧不伪造零速度。这是仿真近似，原GPU杆物理/参数、接触阈值不变。
- 恢复初态筛查使用真实保存的wire形状/体积守恒半径，修正默认直线与已受载姿态不匹配的误拒绝；同姿态真实余量1.2428mm通过原1mm门槛。
- V15在1412步正常停止，保存有效1000步；V16启动前筛查退出，未步进；V17从保存状态续验。最新114有效close、89帧四点严格接触有效，尚无稳定抓握/hold/lift，不作为成功训练源。
- 专项182 passed in13.31s；边界修改后全量856 passed/71 failed（同旧失败集合），恢复入口后的最终全量待完成。无提交/上传。

## 2026-10-21 — 预载后的逐手目标保持

- V14有效状态续至2891步仍为close；保存1200步后正常停止并重载控制修改，不重跑approach。
- 基于实际已知接触/载荷捕获每只手的当前指令：95%目标捕获、90%保留，载荷/接触下降恢复调节；不导入抓取成功。保持约束进入MPC候选指令轨迹，保留真实Jacobian/受载预测；学生/教师使用同一交接。
- 保持时从实测深度更新参考，避免未执行力误差积分；完整采集保存保持状态，所有物理成功/碰撞判据不变。
- 单实例V15从保存的V14接触状态继续；最近378有效close、双手已保持，但尚无稳定抓握/hold/lift，也未正式蒸馏。
- 专项179 passed in13.49s；全量854 passed/71 failed in59.33s，失败集合与上次相同；编译/diff通过。未提交/上传。

## 2026-10-21 — 原区域内的指腹中心对准、载荷反积分饱和

- V13在468步因左食指局部余量.9985mm停止，最后四点约.950/1.058/1.090/1.213N；外层非指腹检查通过，仍非成功抓握。
- 新增离线接近终点对准，限制掌心移动/转角并保持实际间隙；不改标定区域、USD或wire模型。候选双掌均<1mm、转角1.407°/.835°、间隙差<.06µm，左食指局部余量约1.578mm；50体源几何/6区域/名义初态静态检查通过，不是运动或物理成功证明。
- 初态重构保留wire动态、关节速度及主动驱动偏差；新诊断不导入阶段成功。载荷达到目标后参考回算到当前接触深度，防止历史参考继续要求过闭合，原力矩/成功判据不变。
- V14实际续验最近1430步有效close，四点.974/1.082/.948/1.042N，但稳定接触未通过、未hold/lift；下一步处理预载达标后的目标保持交接，不直接放宽判据。
- 专项171 passed in6.39s；全量846 passed/71 failed in58.58s，与上次失败集合相同，编译/diff通过。未正式蒸馏、提交或上传。

## 2026-10-21 — 因果跟踪校正、预载参考恢复与受载限位

- 基于实测 q 和上一已执行指令校正 MPC 受载预测偏差；有界/因果，不修改实测力或成功判据。V11 记录的离线核对 RMSE .001077→.000152 rad；并非物理成功证明。
- 受载平衡命令保留1mrad预测限位余量，仍逐步限速。支持显式恢复预载深度及预测偏差，从已完成控制区间估计时排除未来/失败/短末区间。
- V11 failure2815/lastvalid2814：右拇指耦合下限处停止；V12携带2400步接触进度续验，四点 .761/.926/1.066/1.153 N 后因右手跟踪误差略超1度在200步停止。无 hold/lift 成功。
- V13从V12有效199步恢复，模拟 Kp6→10，比例目标重算保留初始力矩，原 .15Nm cap 不变；旧增益偏差失效但预载参考保留。最新读53步有效close，非实机参数标定，无 approach 回放。
- 完整源训练同时要求整线离桌标记；不更改原站点提升诊断契约。
- 专项159 passed in12.56s，编译/diff通过。最新全量840 passed/71 failed in58.54s，与上次失败集合完全相同，无新增失败，全仓库仍未通过。V13随后259步仍有效close，四点.769/.913/1.063/1.211N；成功专家、正式蒸馏与实际模型验收未完成。

## 2026-10-21 — 完整区间采集、反馈历史模型与 V11 专家续验

- 新增 `ContactDecisionCollector` 和 `dh116-contact-decision-v2`，保留 q/qd、上一动作、接触反馈/已知标记、几何/Jacobian、完整杆动态、目标/阶段及 MPC 预测。下一控制观测关闭上一个实际动作区间；hold/lift 显式保持，末端不完整区间不冒充成功。
- 新版采集已接实际 Isaac。V11 从 V10 最后有效 step1030 开始 close，原 USD/GPU 杆不变，无 approach 重跑；已校验 13 个真实 10 ms 区间。最近检查 step1325 有效、四点约 .608/.701/.815/.886 N，仍在 close，未 hold/lift 成功。
- V10 临界局部 patch 粗采样下界略低于 1 mm；同姿态细采样约 1.20–1.21 mm。新增自适应局部采样和可移植真实接触拒绝测试，不扩大原区域或降低间距/穿透标准。
- 新增 267 维三帧历史学生与 1265 维（40 单元杆）特权教师，保留手部 12D 上一指令残差和停放关节契约；完整当前杆动态只供教师，未来结果不进入输入。新 v2 checkpoint/整机适配与旧格式显式分开。
- 训练入口默认完整新版区间，要求成功源及 close/hold/lift；旧格式须 `--legacy-contact-replay`。当前 V11 的实际入口拒绝测试符合预期，未启动正式训练或创建模型。学生观测目前来自仿真，不宣称已接实机感知。
- 本轮相关 **150 passed in12.29s**，编译/diff 通过；包含使用合成测试数据的训练 CLI/模型重载测试，不是物理抓取成功。全量未重跑，上次 811 passed/71 failed 仍有未解决项。
- 未完成：成功实际专家抓握/抬升、成功数据正式蒸馏、教师/学生独立 Isaac 验收和留出评估。无提交/上传。

## 2026-10-21 — 受载预测与教师/学生接线，继续约 0.5 N 物理接触

- 手 MPC 加入实测法向载荷映射的 PD 平衡偏差；仿真 Kp=6、原力矩上限不变，并支持按旧/新增益比例重算初态目标以避免比例力矩跳变。原始 q/qd 与柔性杆动态保留。
- 接通新 `.pt` 教师/学生加载、角色/控制周期校验、训练范围元数据和未参与手指的精确停放。旧 JSON 保持可加载。部分模型 CUDA 推理探针约 0.556/0.679 ms，尚无成功轨迹模型或实际模型抓取成功。
- 修复 pinch-only 同步 lift 误要求中指接触；middle-hook 默认要求未放宽。
- V8 四点载荷约 0.50–0.55 N，485 步受非前面保守距离下界拦截。细化复算从 0.999842 mm 改善到约 1.20 mm；仅对临界线段细化，不改实际间距/区域/穿透标准。
- V9 从 V8 第 484 步继续，含载荷反馈暖启动，259 步在 finger13_link 粗估余量 0.826 mm 处终止。凸包平面包络精化后该失败姿态完整源几何下界为 1.0694 mm，原 1 mm 要求不变。已保存最后有效 step258，无运行中 Isaac。
- 离线手部求解可用拇指两轴配合保留 wire 接触并将相邻非指腹指节余量提升至 1.6 mm；这是几何可行方向，尚非物理路径。下一步给手部 MPC 加入非目标指节避让目标，继续真实抓握与提升。
- 专项 111 passed / 4 deselected，新增凸包/源几何专项 27 passed / 1 deselected；最新全量 **804 passed, 71 failed in 29.00s**，编译和 diff 通过。未保持/抬升成功，未将既有失败标解决。

## 2026-10-21 — 完整 Cosserat CUDA 计算与实测 close 接续

- CUDA 杆包含剪切/伸缩、弯曲/扭转、惯性、阻尼、桌面/运动网格接触、摩擦及独立的杆/机器人线性和角冲量；USD 不改，不静默回退 CPU。外层 Isaac 交换、几何检查与证据仍有 CPU 工作，未宣称实时。
- 本次 CUDA 专项 27 passed in 5.26s；含控制、状态初始化、数据读取/学习的专项合计 93 passed in 10.41s；编译和 diff 检查通过。实际全场景 V6 四点接触约 0.012 N，尚无夹持/抬升成功。
- 全量 788 passed / 71 failed in 53.79s，存在旧四通道测试/资源与六通道实现契约冲突及旧 approach 时序等问题；未宣称全仓库通过，未将实现退回四通道。
- MPC 接触目标改为法向优先，允许必要切向运动；根据实测载荷和间隙调整闭指参考，不改接触刚度或伪造负载。Panda close 冻结，lift 仍由 Panda 负责。
- 新增 `--close-initial-state` 和保存脚本，从有效 close 行恢复实际 q/qd/目标与 GPU 杆完整动态状态。保存 V6 第 4727 步并从该状态启动 V7，跳过 approach；不导入阶段成功证据，不声称恢复 PhysX 隐式求解缓存。
- 未完成：持续夹持、Panda 抬升、成功轨迹训练和学生整机验收。已有 28 条真实 MPC 决策的小样本训练仅是 partial_action_diagnostic。

## 2026-10-21 — DH116 hand-only parallel MPC and GPU Cosserat

- 新增 12D 双手主动关节 hand-only 状态/动作契约和批量并行 MPC；明确从 grasp 开始，跳过 approach，Panda 状态不进入规划或回放。
- 新增教师/学生回放采集脚本，记录 privileged/non-privileged observation、previous action、MPC target、action、reward、phase，并生成 SHA256 manifest。
- 新增 `TorchCudaCosseratRod` 显式 CUDA 后端；CPU PyElastica 路径保持不变，CUDA 不可用时拒绝启动，不做隐式 CPU fallback。该实现标记为仿真近似，未替代实机参数。
- 验证：hand-only MPC 9 passed，GPU backend 2 passed，短回放 smoke 通过；尚未完成 Isaac 抓取/lift 物理验收。
- 新增 IsaacLab hand-only wire 接入探针；双手和回放动作可加载并推进 PhysX，但首轮姿态/碰撞稳定性验证失败（接触冲量异常大），因此未标记 grasp/lift 成功。
- 验证器可直接运行 student checkpoint，并对 Isaac 关节状态/接触冲量做早停；24 动作/2400 步稳定运行无 wire 接触，调低 wire 后首步冲量超过 500 N。接触位置仍需标定，尚无抓取/lift 成功证据。
- 增加 wire 高度参数用于定位；0.36 m 测试产生 363.99 N 接触峰值并在 8 步内非预期抬升 33.65 mm，安全门禁停止，仍不算 grasp/lift 成功。

## 2026-10-20 — 性能优化 Task 1：阶段计时与运行身份

- 新增 `--profile` 和 `PerfMeter`，按步记录控制、杆/接触、源碰撞保护、反作用提交、PhysX、传感器状态和证据写入耗时，并原子写出 `perf_summary.json`。
- 运行清单补充 `source_commit`、设备、`physics_dt`、wire `py_dt`、接触采样数、CPU 亲和性和数值线程环境；profiling 默认关闭，物理参数和单核约束保持不变。
- 本阶段验证：`bash scripts/python.sh -m pytest tests/test_dh116_perf_meter.py tests/test_dh116_sequence_runtime.py -q` → **31 passed**；脚本编译和 diff 检查通过。
- 未完成：完整回归、同物理时长的三次性能对照、正式 Isaac 性能测量，以及双手持续夹持和抬升验收。

## 2026-10-20 — 性能优化 Task 2：墙钟 wire 显示调度

- 新增 `should_publish_wire` 墙钟调度和 `--ui-hz`（默认 15 Hz）；物理循环每步固定 `sim.step(render=False)`，到期 GUI 帧再发布 wire 并调用一次 `sim.render`。
- `WireMeshPublisher` 缓存首次拓扑，后续只更新点和包围盒；节点或拓扑改变会抛出错误，避免静默写入错误网格。
- 本阶段验证：`bash scripts/python.sh -m pytest tests -q` → **734 passed in 24.68s**；编译和 diff 检查通过。
- 未完成：正式 Isaac 三次同物理时长测量、快照/反作用缓冲、证据缓冲，以及双手持续夹持和抬升验收。

## 2026-10-20 — 性能优化 Task 3：快照与反作用缓冲

- 新增不可变 pre/post `RuntimeSnapshot`，每步 token 独立递增；反作用记录绑定 DeformX post token。
- 新增 `ReactionBuffer`，左右臂的力、矩、作用点及环境索引张量预分配并原地清零，避免每步重新创建整组张量。
- 本阶段验证：`bash scripts/python.sh -m pytest tests -q` → **736 passed in 24.57s**；编译和 diff 检查通过。
- 未完成：证据缓冲策略、正式 Isaac 三次同物理时长测量，以及双手持续夹持和抬升验收。

## 2026-10-20 — 性能优化 Task 4：证据 flush 策略

- `EvidenceJournal` 增加 `audit`/`buffered` 策略、按条数或墙钟时间 flush，以及公开 `flush()`；阶段切换、失败与关闭强制落盘。
- verifier 增加 `--evidence-flush-policy`、`--evidence-flush-steps`、`--evidence-flush-seconds`；默认仍为 audit，轨迹行数和字段不变。
- 本阶段验证：`bash scripts/python.sh -m pytest tests -q` → **739 passed in 24.52s**；静态配置、编译和 diff 检查通过。
- 未完成：三次同物理时长 Isaac 性能测量、可复现性能报告，以及双手持续夹持和抬升验收。

## 2026-10-20 — 性能优化 Task 5：可复现性能比较工具

- 新增 `scripts/run_performance.py`，按 variant/phase/repetition 串行启动单个 verifier，固定物理身份、单核环境和独占输出目录。
- 新增 `scripts/collect_performance.py`，要求每组至少三次完成运行，校验 `physics_dt`、wire `py_dt`、接触采样和步数一致，再计算中位数、P95、SPS、RTF并标记不可比组。
- 本阶段验证：runner 专项 **3 passed**；完整回归 **742 passed in 24.56s**。
- 未完成：实际 Isaac 三次串行测量、报告审查，以及双手持续夹持和抬升验收。

## 2026-10-20 — 性能优化 Task 6：Isaac 三次对照证据

- 在 RTX 5070、core 0、固定 `physics_dt=1e-4`/wire `py_dt=1e-5`/16 点采样下，baseline、buffered、GUI 15 Hz 各完成三次 1200 步 settle/diagnostic。
- 中位墙钟：baseline 36.953951 s，buffered 36.933635 s（1.000550×，不可视为确定提速），GUI 44.014520 s（0.839585×）。所有运行 1200/1200 步通过物理门。
- 报告：`docs/superpowers/reports/2026-10-20-dh116-performance-report.md`；原始数据留在本地 `runs/perf_20261020_settle/`，不上传运行目录。
- 结论：当前主要热点是 DeformX 杆/接触求解；证据缓冲和显示调度不能证明整机加速。close/hold/lift 与真实抓取仍未验收。

## 2026-10-20 — 抓取控制与半隐式接触开发快照

- 新分支 `feat/dh116-simulation-10-20`，汇总当前控制代码、配置、测试与设计计划。
- 改进包括前伸/阶段内对称姿态、分阶段速度控制、wire 渲染发布与接触查询优化、实验半隐式求解及杆动力学/Jacobian 适配。
- 本次验证：`bash scripts/python.sh -m pytest tests -q`，726 passed in 26.94s；`git diff --check` 通过。
- 阶段限制：半隐式仅有离线验证，正式 Isaac 尚未切换；实际持续夹持和抬升未完成，整机提速未验收。后续继续长时/实际指腹验证、接线与真实抓取测试。历史 v1 配置保留来源状态，包含未完成配置；路径仍需异机适配。
- `runs/`、`archive/`、缓存、日志及临时进度碎片 `1332` 留在本地，不随本次上传。

## 2026-10-19 — Isaac closure-pilot 接触前探针

- 独立单核 Isaac 实例启动成功，真实桌面与 DeformX wire 物理步进稳定；短跑输出保留在 `runs/isaac_contact_probe_20261019/`。
- 该模式固定在预姿态，只做轻触诊断，不会自动执行 approach；约 `0.108 s` 仿真时间内 wire 支撑误差约 `2.5 µm`、四指负载均 `0 N`、指腹与 wire 间隙约 `19.5–22.4 mm`，因此没有宣称接触或抓取成功。已安全停止，避免无效长跑。
- 下一步应运行满足 `>5.1 s` 约束的正式 approach 轨迹，再进入闭指/保持/抬升；半隐式后端仍未切换到 Isaac runner。

## 2026-10-19 — 10 ms 步长减半与移动边界复核

- 静态和移动边界各完成 `0.01 s`、`2.5→1.25 µs` 内步长对照；两组均满足位置、峰值力、线性冲量与反作用平衡门槛。
- 新后端末状态差异（`2.5 µs` 对 `1.25 µs`）：静态位置最大 `3.283e-7 m`，移动边界 `3.284e-7 m`；姿态分量分别 `4.748e-6`、`4.756e-6`；角速度分别 `7.938e-4`、`7.948e-4 rad/s`。
- 报告保留在 `runs/implicit_contact_10ms_*_20261019.json`。这是离线合成杆/网格数值证据，不是正式 Isaac 或真实抓取验收；下一步仍需在独立实例中接入/回放并记录接触、闭指、夹持、抬升证据。

## 2026-10-19 — 半隐式后端短时数值门

- 新增 `scripts/benchmark_dh116_implicit_contact.py`，对比旧显式与实验后端的静态/移动边界短时工况。
- 内步长2.5 µs、0.001 s：静态位置差1.65e-7 m、峰值力差0.00468 N、冲量差2.83e-6 N·s；移动边界位置差6.43e-6 m、峰值力差0.00523 N、冲量差6.36e-5 N·s；线性反作用平衡为0。两组短时门通过，原始 JSON 保存在 runs（不上传）。
- 全量回归 **726 passed in25.54s**。短时结果不能外推到100 ms或真实抓取；下一步延长工况、执行2.5→1.25 µs步长减半，再考虑CLI/Isaac切换。

## 2026-10-19 — 半隐式接触实验后端接线

- 新增 `co_sim/implicit_moving_contact.py`。它复用原 Verlet，但移除旧显式 `RodMeshContact` 注册，在每个子步把桌面和四个指腹的网格样本联合送入接触求解器，再一次写回杆节点外力/材料力矩；指腹网格获得相反反作用，桌面反作用单独记账。
- 通过真实 PyElastica 后端 smoke：无接触与原 `MovingContactRodEngine` 状态逐字段对照；指腹反作用峰值及 solver residual/step count 检查。专项7 passed，全量 `bash scripts/python.sh -m pytest tests -q` 为 **726 passed in24.88s**。
- 这是实验后端，尚未运行 Isaac 或通过细步长收敛门；未宣称替换旧运行、实时或抓取成功。下一步对静态/运动细步长参考做力、冲量和角冲量平衡，再决定是否开放 CLI 选择。

## 2026-10-19 — 路线1单次接触冲量组件

- 用户选择保持物理参数的路线1；新增 `co_sim/implicit_stepper.py`，将联合求解器输出经真实接触 Jacobian转换为广义接触力、线性反作用与单次冲量，数组只读，调用方负责一次提交。
- TDD专项5 passed；全量 `bash scripts/python.sh -m pytest tests -q` 为 **724 passed in 34.84s**。
- 仍未接入 `MovingContactRodEngine` 的 Verlet 调度，未做新算法细步长对照，未启动 Isaac；不能将该组件当作整机提速或抓取证据。下一步接入实际 step callback，并增加非接触等价/重复施力拒绝测试。

## 2026-10-18 — 实际杆质量响应与接触Jacobian适配

- 新增 `co_sim/implicit_contact_adapter.py`，保持节点世界坐标与材料转动坐标定义，旋转响应为invI*dilatation，接触Jacobian保留原节点插值与力臂；不修改活跃杆。
- `bash scripts/python.sh -m pytest tests/test_dh116_implicit_contact_adapter.py -q`：先8项预期RED，后8 passed。直接逐列对比实际杆加速度核，并对比1/4/16子采样的原接触力/力矩分配及功率共轭。`bash scripts/python.sh -m pytest tests -q`：719 passed in26.09s。
- 自审确认桌面原力律也是RodMeshContact；后续应读取实例参数而非默认值。尚未接一次kick与通道冲量统计、未做新积分收敛、未切Isaac。下一步独立实验后端接线，不重复施力。

## 2026-10-18 — 半隐式接触纯数值核心（未接运行）

- 用户确认详细设计，按 Inline Execution 开始；新增 `co_sim/implicit_contact.py`，联合求解共享速度的法向阻尼及库仑限幅切向力，保持积分权重，不修改输入，失败不返回合格解。
- TDD：19 项缺失实现预期失败，实施后通过；补充旋转/力臂/无粘附检查，针对性 22 passed。`bash scripts/python.sh -m pytest tests -q` 本轮 711 passed in 25.90s；`git diff --check` 通过。
- 仅离线核心，尚未适配实际 Cosserat 动态响应、步进、反作用记账与细步长验收。当前 Isaac PID31220 未重启/热加载，抽样 step60834 close、载荷0；夹持和抬升仍未完成。
- Inline自审：此核心接受完整SPD响应及接触Jacobian，但不能由解析单元测试推断实际杆合格；dense差分Newton的性能未测，不宣称实时。后续实际响应必须包含 dilatation 倍率。

## 2026-10-18 — 运动参考验证与预测冗余优化

- 新增私有已校验输入预测路径，外层仍独立校验/冻结动作，公开接口仍全校验，预测输出继续检查有限性、旋转及不可变数组；不修改动力学方程。后端指纹增加 moving_contact_models.py。
- 2 项预测测试及 1 项指纹测试先失败后通过，针对性 80 passed，全量 689 passed in 51.68s。1000 次预测三组交替微基准中位数 .388637→.208689 s，非整机性能证据。
- 独立指腹竖直 10 mm/s、100 ms，内步长 2.5→1.25 µs 的位置、峰值力、冲量差均通过容差；证据保留 runs/forward_gui_render10/moving_dt_halving_probe.{json,py}。没有真实机器人闭环抓握资格。
- 旧实例保持运行，尚未加载新优化；稳定接触积分和状态迁移方案等待用户选择，未改写旧状态。实际夹持/抬升尚未完成。

## 2026-10-18 — 静态双侧细步长参考建立

- 原摩擦、材料、真实指腹网格不变，100 ms 工况中 2.5→1.25 µs 比较：位置差最大 6.24497e-7 m、峰值力差最大 0.00095003 N、冲量差最大 1.35468e-7 N·s，通过预设工程容差。
- 耗时 47.489/94.795 s；只作为后续稳定积分优化的参考，不能声称实时或整机闭环验收。旧生产步长和实例未改变，夹持/抬升未完成。
- 原始报告/复现脚本保留于 runs/forward_gui_render10/inner_dt_finest_halving_probe.{json,py}；此次是数值诊断，无新增生产实现或新全量测试声明。

## 2026-10-18 — 摩擦阻尼步长敏感性定位

- 仅在独立消融诊断中关闭切向摩擦，10→20 µs 位置差降至 0.000749 mm；不将无摩擦模型用于抓取。生产参数未改变。
- 初始冻结几何的质量归一阻尼分析得到最快时间常数约 1.313 µs；仅局部线性诊断，不冒充完整系统稳定界。
- 保持原摩擦时 5→2.5 µs 位置差 0.123491 mm，但峰值力最大差 0.058707 N，超出 0.05 N 阈值，整体未通过。原始证据/脚本保存于 runs/forward_gui_render10，未覆盖旧报告。
- 本轮无生产实现改动，不重复声称上一轮测试为本轮验证。旧实例仍在 close/touch；后续需更细参考和稳定积分验证，真实夹持/lift 未完成。

## 2026-10-18 — 内部步长收敛风险

- 静态双侧真实指腹网格、100 ms 物理时长、外层 100 µs 下，内部 20/50/100 µs 相对 10 µs 的中心线差为 1.585/6.887/13.425 mm，均未满足 0.2 mm 容差。10→5 µs 减半差仍为 0.548 mm，基线精度收敛尚未证明。
- 仅离线数值实验，生产参数、旧状态和运行实例均未修改；不把数值有限当作精度通过。证据和复现脚本保存于 runs/forward_gui_render10/inner_dt*_probe.*。
- 只读核对旧 close 状态外部 SHA、来源配置及接触点定义，未改写旧 pin。需先明确迁移并重新验证，再切换优化版；实际夹持/抬升仍未完成。

## 2026-10-18 — 保守接触粗筛编译优化

- 新增编译版点到网格 AABB 距离筛选，保留浮点保守余量、内部/近表面候选及原精确网格查询。实际姿态四路候选 80→66；不降低子采样、不改力学参数/时间步长。
- 测试先 4 项预期失败，实施后移动接触针对性 49 passed；全量 686 passed in 32.96s；后端 rod_mesh_contact 43 passed in 5.37s；diff 检查通过。
- 静态双侧接触 5 ms、三组交替基准中位数 1.908→1.591 s；采样位置、累计冲量逐位相同。存在并行 CPU 竞争，不能替代整机性能或闭环验证。证据留在 runs/forward_gui_render10/broadphase_ab.json。
- 旧 Isaac 保持运行、不热加载；实际仍在 close 且载荷零。后续需合法切换优化版、验证移动反馈与真实夹持/lift，未上传。

## 2026-10-18 — 独立耦合间隔诊断

- 固定真实指腹网格对比落体 60 ms、单指/双侧各 10 ms，外层 100/250/500/1000 us，杆内 10 us；运行证据保留在 runs/forward_gui_render10/coupling_*_probe.json。
- 落体耗时 8.022→7.455 s，双侧 1.753→1.804 s；各一次计时且与 Isaac 并行，不能称为可靠提速。静态边界数值一致不代表机器人闭环稳定。
- 针对速度/视觉/接线测试 32 passed in 0.51s。生产 dt 未改、实例未重启；夹持/lift 未完成。下一步优化内部接触查询并补移动边界验证。

## 2026-10-18 — 分阶段速度运行接线与分配诊断

- 新增显式 `--grasp-speed-profile staged`，默认 baseline 保留。候选 profile 写入配置指纹，实测四路网格间隙传入状态机与恢复流程；不接受缺失或非有限间隙。
- 同步抬升速度可配置，首轮 staged 为 10 mm/s；原高度差、接触、滑移和力矩保护不变。较大的自由空间法向速度与带载力反馈速度分别限幅，保留原关节限速。
- 运行记录新增 Jacobian 分配前/后速度和共同缩放系数，明确不是物理实测速度。先 5+2 项预期失败，后针对性 58 passed；完整回归 682 passed in 27.41s。
- 真实基线 step 43896 的离线 FK 诊断：共同缩放 3.019，候选左/右拇指 9.327/1.000 mm/s，分配 3.089/0.331 mm/s。共同限速可能拖慢近侧，后续必须验证可行速度而非仅调高请求值。
- 尚未切换 Isaac；认证接近路径的时间参数尚未加速，耦合步长仍保持基线，真实夹持/lift 尚未完成。

## 2026-10-18 — 实测间隙分阶段速度组件

- 新增不可变 `GraspSpeedProfile` 与 `ContactSpeedGovernor`；按实际指腹表面间隙计算制动包络并限制启动加速度，接触/载荷优先切换力反馈，撤退和停用通道优先于平滑增速。
- `TouchFirst` 新增可选 profile 与实测 gaps 输入；默认保留原结果。坏间隙在阶段更新前拒绝；未扩大驱动力矩或关节限速。
- TDD 先验证 13 项缺少功能失败，后针对性 35 passed；全量 `bash scripts/python.sh -m pytest tests -q` 为 675 passed in 27.31s。
- 候选模式尚未接入实际运行入口，不能称为已完成仿真提速。下一步接入控制分配缩放诊断、状态机/恢复和抬升参数，再进行数值与实际抓取验证。

## 2026-10-18 — 抓取提速第一步与当前实际阶段

- 用户确认动作分阶段提速与计算优化设计，CPU 允许使用全部核心；保留单 Isaac、柔性杆、反作用力和碰撞保护。
- 新增 `WireMeshPublisher`，缓存 USD 属性并仅初始化一次拓扑；运行循环仅在渲染帧发布视觉网格，物理与保护仍逐步运行。初始和状态恢复后的可视化发布保留。
- 测试先观察 4 项预期失败，后针对性 9 passed；完整 `bash scripts/python.sh -m pytest tests -q` 为 655 passed in 32.09s，`git diff --check` 通过。
- 离线 USD 视觉发布基准 1001 循环、3 次中位数：旧逻辑 3.25348 s、新逻辑 0.05485 s。仅说明视觉部分减少重复开销，不能推导真实总仿真加速倍数。
- 实际 `forward_gui_render10` 检查 step 41119/close，通过当前检查，但四路载荷仍未有效建立；接近已通过，闭指/夹持/lift 未完成。当前进程未热加载新实现；GUI 暂停恢复事件已单独记录。
- 下一步：参数化分阶段速度、分段计时与耦合数值对比，再合法接续验证。未上传 GitHub，未合并或覆盖其他未提交工作。

## 2026-10-18 — 前伸抓取离线候选与用户要求的仿真关闭

- 本轮完整回归：`bash scripts/python.sh -m pytest tests -q`，647 passed in 25.43s；`git diff --check` 通过。未执行物理仿真，测试通过不代表夹持或 lift 已成功。

- 用户要求关闭 Isaac；`runs/symmetric_gui2` step 6518 记录 operator stop，进程退出，保留轨迹，不视为算法成功/失败结论。
- 新增 `forward_pose_report`，使用标定掌心 +Y/+Z 检查向前、朝下和腕部在抓点后方。先观察缺少模块失败，再完成实现；方向与镜像针对性 19 passed in 0.12s。该模块不修改实际姿态或物理控制器。
- 离线候选 1–14 保存在 runs。候选 14：source 最小下界 1.3057 mm；四个指腹区域检查通过；镜像位置误差 0.7819 mm、姿态误差 4.1452°。左/右指腹径向距离为 [8.7593,8.3124]/[7.4034,7.8310] mm，径向法向误差 [4.7521,3.2220]/[3.9461,2.6309]°。
- 以上不是全部验收：候选 14 的手指前伸方向偏转约 34–36°、掌心倾斜约 18°，不满足原 15°方向门槛。已向用户展示 `runs/forward_feasibility14/pose_preview.png` 并请求确认是否接受倾斜；尚未改门槛、发布配置、验证接近路径或运行物理抓取。
- 当前搜索诊断脚本在 `/tmp/dh116_forward_probe.py`，是本机一次性工具而非发布生成器。逐次报告保存参数与资产层指纹，旧候选保留。后续发布前需将生成流程纳入可复现工具；不能仅凭候选报告或图片宣称抓取成功。

## 2026-10-18 — 镜像运行首步失败定位与带限位余量的新配置

- 运行接线回归 634 passed（51.93s）；这是离线结果，不代表抓取成功。镜像仅用于抓取/lift，通用运输分配仍允许非对称动作。
- `symmetric_gui1` 首步实测右 finger12 为 0.007912546396 rad，低于耦合下限 0.007912622218 rad；原初态只留约 1e-8 rad，碰撞模型严格拒绝。失败原件保留，未 clamp 实测值或放宽限位。
- 重求预抓配置，优化边界内缩 0.005 rad。命令 `bash scripts/python.sh /tmp/dh116_symmetric_approach_probe.py --name symmetric_approach_probe2 --joint-margin .005`：130 次路径检查通过，预姿态/张开姿态 source 下界约 1.017 mm。
- `bash scripts/python.sh /tmp/dh116_symmetric_jacobian_probe.py --name symmetric_approach_probe2` 通过，最大 USD FK 有限差分误差 1.135e-10 m/rad。生成独立 `configs/symmetric_grasp_v2`，旧 v1 不覆盖；临时诊断脚本及 runs 证据仍为本机文件。
- 启动单实例 `runs/symmetric_gui2`，CPU affinity 0。真实 PhysX 接近、闭指、夹持、lift 尚待验证；新增余量不等于物理成功，运行驱动参数仍为仿真参数。

## 2026-10-18 — 新姿态USD Jacobian核验与lift状态机组件

- 新预姿态在初始杆中心z=0.318 m检查通过，source最小下界1.238 mm。prepose/open/grasp三姿态共12个指腹点的13维Jacobian对比USD FK有限差分通过，最大误差1.274e-10 m/rad；接近限位处采用不越界的单侧二阶差分，未放宽模型限位。报告 runs/symmetric_approach_probe1/jacobian_and_initial_check.json。
- TabletopGraspSequence新增显式 symmetric_lift（默认关闭），只在lift调用高度协调器；其他阶段输出零运输速度。两侧高度差超过1 mm时重置抬升成功保持计时，达到2 mm停止运输并记录abort；原非同步模式保持不变。
- 先因缺少模式参数观察3项失败，再实现；状态机、原证据门槛和恢复相关50项测试通过（1.84s）。运行器CLI与新配置仍未切换，实际PhysX点Jacobian及动态夹持/抬升尚待验证。

## 2026-10-18 — 镜像静态候选与接近路径取得通过证据（非物理验收）

- feasibility8 显式约束优化收敛；掌心镜像位置差0.900 mm、朝向差3.180度，径向法向误差左4.500/2.691度、右4.500/2.677度，同手轴向错位左0.0251 mm/右0.0011 mm。低于设计1 mm/5度门槛。
- 四指区域均通过，非指腹下界分别2.756/2.971/1.735/2.498 mm；其他source静态下界1.315 mm。仍仅为离线源几何证据，不是PhysX/DeformX动态抓握证据。
- 从候选构造向外张开2 mm目标，再求解抬高30 mm预接近姿态。runs/symmetric_approach_probe1 保存 setup.json 和 sweep.json；自适应全关节盒路径检查130次，static_wire_source_sweep_passed=true。原有 approach_target 速度检查通过。
- 仍需检查初始落桌杆位置、配置指纹和实际USD点Jacobian，完成运行接线后才能启动新候选；没有覆盖旧配置或热修改GUI，没有宣称实际夹持/抬升成功。

## 2026-10-18 — 镜像约束求解进展，仍不发布运行配置

- feasibility5 增大法向/掌心权重后，掌心镜像差1.130 mm、朝向2.954度，四指区域通过，但拇指径向误差仍约5.05/5.29度，未达标。
- feasibility6 首次显式约束优化在数值边界触发严格耦合限位拒绝。保留失败目录与日志，未放宽模型限位；后续求解边界向内缩1e-8 rad，避免有限差分查询越过源限位。
- feasibility7 显式约束结果掌心差约1.000000002 mm、朝向3.040度，全体source静态下界1.343 mm；优化器达到迭代上限，且约束仍存在极小负残差。未将其标成已收敛/合格配置，需留出数值余量并复核指腹、路径。
- 所有运行均为离线单核探测，当前GUI未改动，连续物理抓取/抬升尚未通过。

## 2026-10-18 — 镜像候选指腹区域与掌心复核

- feasibility2 四个指腹区域静态检查均通过，但按已标定掌心功能轴比较，镜像位置差22.135 mm、朝向差14.722度，未满足设计；不能仅凭指腹目标对齐采用。
- feasibility3 将两掌镜像纳入联合 IK 后，掌心位置差0.790 mm、朝向差1.381度；右拇指压入超过1 mm，被指腹检查拒绝。其余全体 source 静态筛查通过不能抵消该失败。
- feasibility4 增加接触点位置权重后，四个指腹区域检查均通过，非指腹下界2.205–2.789 mm，source 静态筛查通过；但掌心镜像位置差1.682 mm，拇指实际径向法向误差左5.812/右5.944度，仍超过本轮1 mm/5度目标，未生成运行配置。
- 功能轴采用已有掌心标定：将 wrist-to-middle (+Y) 映射为功能X，掌心外法向作为功能Z，再重构右手基，避免把反射当旋转。原始诊断报告在各 runs/symmetric_feasibility*/region_and_palm_check.json；没有修改旧标定或碰撞阈值。
- 尚需改善候选、验证连续全臂路径和实际夹持/抬升；当前 GUI 保持旧版本，不将离线静态通过冒充物理成功。

## 2026-10-18 — 镜像指腹目标离线可达性探测

- 读取实际 USD、现有标定点和限位，分别求解两侧 Panda+拇指/食指活动关节；未复制左右关节角、未发布运动指令。两个诊断结果保存在 runs/symmetric_feasibility1 和 runs/symmetric_feasibility2。
- 20 度接触倾角候选的目标位置误差为左1.863 mm/右1.151 mm，法向误差超门槛，拒绝作为运行候选。
- 5 度候选：目标位置误差左0.084 mm/右0.610 mm；按实际局部线缆圆心重算的法向误差左0.451/0.688度、右4.000/4.521度；同手轴向错位左0.014 mm、右0.041 mm。
- 第二候选非目标全体 source 静态筛查通过，最小下界1.038 mm；目标指腹 bodies 在此检查中显式委托，不代表指腹前/后区域检查通过。功能掌心镜像、指腹区域、连续全臂路径与动态夹持/抬升均未验证，严禁直接当作合格配置启动。
- 现有 GUI 未热修改；检查到step6592/approach/passed=true，仍为旧几何。数值可达性改善不等于物理抓取成功。

## 2026-10-18 — 阶段内高度同步数值组件（未接线）

- 新增 SynchronizedTransport：实测双侧高度、1/0.5 mm 等待滞回、2 mm 拒绝门槛、5 mm/s 渐变速度、1 mm 目标领先上限、独立速度输出。接触失效或非法证据拒绝运输；方向改变重新初始化协调状态。
- 只供选中的 lift/可选放回阶段调用；通用分配器仍允许非对称动作，未改模型或 Jacobian，未热修改当前 GUI。
- 先因组件缺失失败，随后 20 项通过，包括不等跟踪速度的数值闭环；完整回归 629 passed in 53.47s，退出码0。该闭环为数值测试，不是 Isaac 抓取成功。
- 当前 GUI PID24035 仍是旧策略，检查到 step4473/settle/passed=true。新镜像 IK、全路径筛查、运行接线、状态恢复和实际抓取抬升验收仍未完成。

## 2026-10-18 — 镜像抓取组件与非全局约束（尚未运行接线）

- 用户限定镜像仅用于抓取/lift，其他任务可非对称。已同步修改设计和实施计划；同步放回为显式选择的演示分支，不自动替换后续任务。
- 新增功能坐标镜像，验证右手旋转合法性而非直接镜像四元数；双手分配器新增 (2,3) 独立运输速度，保持原共用 (3,) 输入及共同限速。未修改 USD、Jacobian 或在线 GUI 控制。
- 测试先观察新双侧速度输入拒绝、镜像模块缺失，再实现；针对性 38 passed，完整回归 607 passed in 52.12s。随后增加一手保持/另一手运动以及不同方向运动的非镜像回归，针对性 40 passed in 0.75s；该两项新增测试不包含在先前 607 项计数中。
- 尚未完成：实测高度同步协调器、新镜像 IK/路径配置、阶段接线及物理夹持/抬升验证。组件通过不等于已在仿真中对称抓取。

## 2026-10-18 — 实际 Isaac GUI 演示启动（未验证抬升成功）

- 用户明确要求 Isaac 仿真画面，不以数据回放代替。协作停止 PID23148，trial2 summary 标记 operator stop requested、step6400；settle 已通过，approach 中止，后续阶段未运行。记录及阶段文件未删除。
- GUI 根因：运行器固定 sim.step(render=False)。先加入 GUI/headless 开关测试并观察失败，再改为 render=not args.headless；GUI 设置桌面相机和照明，不改轨迹、反作用力、碰撞保护或额定参数。
- 针对性验证：GUI 与 prepublish 测试 3 passed in 0.14s。实际窗口标题 Isaac Sim 5.1.0，截图确认两臂、灵巧手、桌面与橙色 wire；PID24035、CPU affinity 0，唯一仿真实例，记录在 runs/touch_first_gui1，检查到 step325/settle/passed=true。
- 后续完整回归：bash scripts/python.sh -m pytest tests -q --tb=short，595 passed in 52.51s，退出码0。检查 GUI 到 step714/settle/passed=true，仍未证明闭指、夹持或抬升成功。旧失败 lift 的数据回放保存在 runs/failed_lift_replay，仅为诊断动画，不是 Isaac 录像。

## 2026-10-18 — 独立目录真实启动修复与轻触试验启动

- `runs/touch_first_trial1` 在物理步进前报 ModuleNotFoundError：隔离时漏迁仍引用的 scripts.verify_fixed_dual_panda_o6_moving_contact.prepublish_contact_gates。所有阶段 NOT_RUN，不算物理验证通过。
- 将该纯检查函数原逻辑抽取到 dh116/prepublish.py，主运行器改用此模块；不带回O6资产或其他任务，不修改检查阈值。新增测试先因模块缺失失败，修复后2项通过；完整回归594 passed in 24.51s，退出码0。
- 原PID6780已退出后，独立新目录 `runs/touch_first_trial2` 启动唯一Isaac，PID23148、exec session86317；检查时step361、settle，真实物理步进开始。日志 `/tmp/dh116-touch-first-trial2.log`；启动失败记录保留未覆盖。
- 新策略完整抓取验收仍未完成。右手联合几何校正方案待确认，当前试验只验证已批准的touch-first接线，不冒充已修正右手几何。旧阶段状态与新代码指纹不同，未篡改旧pin用于恢复。

## 2026-10-18 — 原实例结束：右拇指非指腹区域间隙保护停止抬升

- 已确认 PID 6780 退出，summary.json 最终 passed=false，failure_step=76165；settle/approach/close/hold=PASS，lift=FAIL。并非日志超时推断，也没有重启。
- 定位：measured_source_guard_failure.pad_regions.right_thumb 的 nonfront_lower_bound_m=0.000999939582238412，低于1 mm门槛；其他三个 pad region 通过。此为保守间隙下界失败，不等同已证明碰撞穿透。未修改保护阈值。
- 最后完整步76164：两站升高1.139/0.219 mm、滑移1.332 mm、四指有效接触，向上合力约0.387 N；未达到30 mm抬升目标。失败候选反作用力未发布。
- 原证据目录 `/home/xk/coding/M1/assets/dh116_v2/table_margin_raise1mm_trial1` 保留，包括 stage_approach/close/hold/lift.json、summary.json、point_probe.json 和 trajectory.jsonl。新策略尚未物理验收，旧状态 pin 不伪造兼容。
- 下一步：右手联合姿态/闭指几何修正方案待用户确认；优先改善同截面对夹与非指腹间隙，再做路径和物理验证。单纯加力不能消除此已观测几何限制。

## 2026-10-18 — 抬升传力与抓取对齐诊断（未改在线控制）

- 原实例 step 71486–71586：日志 `force_w` 是机器人侧反作用力，取其四通道合力的负值得到线缆受指腹的向上力；均值约 0.0792 N（范围 0.0719–0.0861 N）。按实际配置 L=0.5 m、r=0.008 m、密度1000 kg/m³、g=9.81，线缆质量约0.10053 kg、重力约0.98621 N。此窗口不足以证明离桌抬升，法向夹紧载荷不可等同向上托力。
- 使用原进程参数中的 postpilot7_point_references_v1.json、contact_hulls_v2/manifest.json，结合 step 71806 mesh_end_poses 和 rod_position 做世界坐标比较；先扣除 body_local_mesh_com_m 再变换指腹点，未混用 mesh COM 与 link 原点。
- 左手两指中点相对 material station 36 的偏移为 [-0.014, -0.872, -0.181] mm；右手相对 station 4 为 [-0.166, -2.422, 1.251] mm。右拇指较其最近局部圆心高2.472 mm，右食指高0.032 mm；左拇指/食指分别低0.136/0.223 mm。
- 两指法向夹角：左171.94°、右160.10°；各指法向与最近圆心方向误差约3.77–5.81°。参考点不是分布式碰撞的完整压力中心，不能只凭这些几何量断言根因。
- 下一步优先验证右手同截面对夹对齐与实际向上传力，同时保留原实验。未改变资产、厂家参数、在线控制、旧 checkpoint 或碰撞门槛；尚未证明抬升成功。
- 后续窗口 step 75901–76001 的线缆向上分力均值分别为左拇指 +0.0933 N、左食指 +0.1270 N、右拇指 **-0.0240 N**、右食指 +0.1860 N；右拇指有持续向下分量。step 76001 两站升高1.074/0.199 mm，hold基线后滑移1.279 mm，仍在 lift、passed=true。现象支持进一步检查右侧偏斜，但未证明唯一根因，不以局部升高冒充完整抬升。

## 2026-10-18 — 原实例通过 1 秒 hold，进入 lift

- `table_margin_raise1mm_trial1/stage_lift.json`：source.step=68872、passed_stages=[settle, approach, close, hold]、imported_from=null。
- checkpoint SHA256：`81864ffb7274d6e01faf41ff3286d639ac5beb5515ff9cb6276c9a6ccd368e65`。未修改旧状态或指纹。
- step 68935 passed=true、stage=lift，四指载荷约 0.965/0.918/1.040/1.043 N，hold 基线后滑移 0.411 mm。
- 仅完成阶段切换，尚无正向有效抬升高度，更未完成 30 mm + 1 秒持续验证；不声称抓取抬升成功。原单核实例继续运行。

## 2026-10-18 — 原实例闭指通过，进入 hold（抬升仍待验证）

- `table_margin_raise1mm_trial1` 的物理阶段 checkpoint：entry_stage=hold，source.step=58872，passed_stages=[settle, approach, close]，imported_from=null。
- 原件 `/home/xk/coding/M1/assets/dh116_v2/table_margin_raise1mm_trial1/stage_hold.json` SHA256：`96923780e4b9da470121302eeb3d63a282e0cdfde5a57aeed1e417bf4d13f2d6`。保留原 pin，不转写为新代码可恢复状态。
- step 58968 passed=true、stage=hold，四指载荷约 0.948/0.896/0.992/1.014 N；相对 hold 基线滑移 0.00764 mm。
- 尚未通过持续 1 秒 hold、抬升与最终验证；新 touch-first 运行器也尚未物理验收。单实例继续，未热更新或重新初始化。

## 2026-10-18 — 轻触优先运行接线（本地，物理验收待完成）

- 阶段：Task 3 运行接线与离线回归完成；Task 4 新控制器 Isaac 验收未执行。
- 改进：实测位置、法向和全关节低速持续达标后闭指，不再依靠固定 4 秒启动；实测姿态与上次指令限速交接；每手轻触/对向闭指/预载输出接入协同分配器；恢复后重新建立接触证据，运行指纹加入 touch-first-v1。
- 本次验证：`bash scripts/python.sh -m pytest tests/test_dh116_state_triggered_sequence.py tests/test_dh116_stage_reset_runtime.py tests/test_dh116_grasp_sequence.py -q` 为 35 passed；`bash scripts/python.sh -m pytest tests -q --tb=short` 为 591 passed in 50.14s，退出码 0；`git diff --check` 通过。
- 原实例证据：step 53492 四指腹有效法向载荷约 0.228/0.184/0.265/0.250 N，reaction_applied=true，仍处于 close。此为旧控制器证据，不能作为新轻触策略或稳定夹持/抬升成功证据。
- 未完成：原实例收尾、新策略实际接触/夹持/抬升验证。保留单 Isaac、单核、旧 checkpoint 指纹；未修改厂家额定参数或伪造绑定。
- 后续启动校验修正：将新连续抓取入口的最低目标载荷与轻触组件对齐为 0.05 N，拒绝在 Isaac 启动后才发现不兼容；新增测试先失败，修正后运行器/状态触发/恢复针对性 49 passed in 0.45s。上面的 591 项完整回归发生在此小修正之前，未冒充修正后的完整结果。
- 提交前重新完整验证：`bash scripts/python.sh -m pytest tests -q --tb=short`，**592 passed in 51.03s**、退出码 0；运行配置预览退出码 0。原实例 step 54945 仍在 close，四指载荷 0.471/0.426/0.495/0.489 N，最大观测滑移约 0.182 mm；尚未完成夹持/抬升。

## 2026-10-18 — 轻触优先组件（功能分支，尚未运行接线）

- 阶段：Task 1/2 组件实现；运行器仍使用旧双侧闭指逻辑，不能声称已按新方式抓取。
- 改进：到位/低速去抖组件；每手独立 touch/oppose/preload；接触丢失重新轻触；1 N/s 预载目标增长上限；显式法向速度分配保持另一指腹的世界点与掌心朝向。
- 验证：测试先红后绿；针对性 55 项通过（1.26 s），包含实际 USD Jacobian 的食指点保持测试；本次完整回归 579 passed in 51.11s。新增组件未运行接线，不构成新版本物理验收。
- 待完成：组件接入运行器、实测速度输入、平滑接管、阶段恢复、新版本 Isaac 及真实夹持/抬升。
- 原试验未热更新、未启动第二个实例；厂家力矩参数未改。

## 2026-10-18 — 首次独立上传：离线迁移已验证，真实抓取待验收

- 当前阶段：DH116 独立工程离线迁移完成；原 Isaac 桌面余量试验仍在 approach。检查时 step 24241，尚未闭指、夹持、抬升，不宣称抓取成功。
- 改进：代码、必要 USD 及其相对依赖、DeformX 后端、厂家资源和新接触配置集中管理；左手路径增加 1 mm 桌面余量的配置纳入工程，保留碰撞保护。
- 上传治理：排除历史归档和日志；添加每次上传必须记录阶段、改进、验证和下一步的规则。
- 本次验证：`bash scripts/python.sh -m pytest tests -q --tb=short`，559 passed in 51.52s，退出码 0（本地包含未上传的历史测试资源）。`bash scripts/python.sh scripts/run_current.py` 配置校验/命令预览退出码 0，未启动第二个 Isaac。上传清单 413 个文件约 106 MB，无超过 50 MB 的单文件，未发现常见 GitHub token 或私钥签名。新增文档的 whitespace 检查通过；厂家原文件及既有代码存在尾随空白，保持原字节，未为上传修改其内容。
- 未完成：原试验结果收尾、新目录实际 Isaac 验证、真实闭指/夹持/抬升验收、跨机器配置与测试夹具的完整可移植性。
- 下一步：根据原试验安全门结果推进，不绕过接触、反作用力和碰撞保护。
