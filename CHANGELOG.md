# 更新与阶段记录

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
