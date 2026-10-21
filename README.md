# DH116 独立工作目录

## 2026-10-21 close 直入与恢复姿态保持

从 `stage_close.json` 恢复时，`restore_validate` 阶段保持当前实测关节姿态，校验通过后直接进入 `close`，不会重新回放 `approach`。这样可以把接近、闭指、夹持和抬升作为连续阶段运行，同时保留源几何碰撞保护与真实接触负载证据。

## 2026-10-20 连续 approach 的 wire 计算策略

正式连续序列支持 `--approach-wire-mode frozen`：`settle` 阶段正常推进 DeformX，使 wire 落桌并稳定；进入 `approach` 后复用该稳定快照，只推进 Panda/DH116，并持续执行源几何与碰撞保护；进入 `close` 后自动恢复 DeformX、指腹接触冲量和机器人反作用力，随后连续进入 `hold/lift`。不需要人工暂停或重启。默认 `dynamic` 保持原行为。该模式只减少 approach 的 wire 求解，不代表已完成闭指、夹持或抬升验收。

## 2026-10-20 性能优化 Task 1

已加入可选 `--profile` 阶段计时：输出逐步 `perf.jsonl` 与按阶段汇总的 `perf_summary.json`，记录控制、DeformX 杆/接触、反作用提交、PhysX、状态传感器、源碰撞保护和证据写入耗时，并把提交 SHA、设备、时间步、接触采样数、CPU 亲和性和数值线程环境写入运行清单。默认关闭 profiling，不改变物理参数、单核约束、接触反作用或碰撞保护。

本阶段验证：`bash scripts/python.sh -m pytest tests/test_dh116_perf_meter.py tests/test_dh116_sequence_runtime.py -q` → **31 passed**；`python -m py_compile scripts/verify_fixed_dual_panda_dh116.py src/go2_pvcnn/control/dh116/perf_meter.py` 通过；`git diff --check` 通过。尚未完成完整回归和 Isaac 性能对照；实际闭指、持续夹持、抬升仍未验收。

## 2026-10-20 性能优化 Task 2

wire 的 GUI 发布改为墙钟调度，默认 15 Hz；每个物理步仍固定调用 `sim.step(render=False)`，到期后只调用一次 `publish_wire` 和 `sim.render`。网格拓扑在首次发布时缓存，节点数、面计数或索引变化会拒绝复用。物理步长、接触反作用、源碰撞保护和单核约束未改动。

本阶段验证：新增调度和拓扑测试，完整回归 `bash scripts/python.sh -m pytest tests -q` → **734 passed in 24.68s**；脚本编译和 `git diff --check` 通过。尚未做正式 Isaac 三次同物理时长性能对照，也未完成闭指、持续夹持、抬升验收。

## 2026-10-20 性能优化 Task 3

新增不可变运行快照和可复用反作用缓冲。每个物理步保存独立的 pre/post token，接触反作用记录对应的杆状态 token；左右臂反作用张量和环境索引只分配一次，后续原地清零复用。接触反作用仍按每步一次提交，安全门、重力保持和碰撞保护不变。

本阶段验证：`bash scripts/python.sh -m pytest tests -q` → **736 passed in 24.57s**；脚本编译和 `git diff --check` 通过。尚未完成证据缓冲、三次同物理时长 Isaac 性能对照，以及闭指、持续夹持、抬升验收。

## 2026-10-20 性能优化 Task 4

`EvidenceJournal` 增加 `audit` 与 `buffered` 两种 flush 策略。默认 audit 逐行落盘；使用 `--evidence-flush-policy buffered` 时按 `--evidence-flush-steps` 或 `--evidence-flush-seconds` 刷新，阶段切换、失败和关闭始终强制 flush。轨迹行数和字段保持不变。

本阶段验证：`bash scripts/python.sh -m pytest tests -q` → **739 passed in 24.52s**；证据、静态配置和脚本编译检查通过。尚未做正式 Isaac 三次同物理时长性能对照，也未完成闭指、持续夹持、抬升验收。

## 2026-10-20 性能优化 Task 5

新增 `scripts/run_performance.py` 和 `scripts/collect_performance.py`。runner 按 variant、phase 和 repetition 串行启动单个 verifier，固定资产、`physics_dt=1e-4`、wire `py_dt=1e-5`、16 点接触采样和单核启动约束；collector 要求每组至少三次、检查物理身份和完成步数，再输出中位数、P95、SPS 与 RTF。输出目录必须位于 `runs/` 且单次运行独占。

本阶段验证：`bash scripts/python.sh -m pytest tests/test_dh116_performance_runner.py -q` → **3 passed**；完整回归 `bash scripts/python.sh -m pytest tests -q` → **742 passed in 24.56s**。尚未启动正式 Isaac 对照，闭指、持续夹持、抬升仍未验收。

## 2026-10-20 性能优化 Task 6：首次 Isaac 对照

按相同物理时长对 baseline、buffered 和 GUI 15 Hz 各串行运行 3 次，全部 1200/1200 步通过 settle/diagnostic 物理门。baseline 中位墙钟 **36.953951 s**，buffered **36.933635 s（1.000550×，处于噪声范围）**，GUI **44.014520 s（0.839585×）**。因此当前证据 flush 没有确定整机提速，GUI 显示增加墙钟成本；主要热点仍是 DeformX 杆/接触求解。详见 [性能报告](docs/superpowers/reports/2026-10-20-dh116-performance-report.md) 和本地 `runs/perf_20261020_settle/report.json`。

这组运行只覆盖 0.12 s settle/diagnostic，没有进入闭指、持续夹持或抬升，不能作为抓取成功证据。下一步需针对杆/接触求解热点单独设计优化，并保留完整物理验收门。

## 2026-10-20 上传快照

分支：`feat/dh116-simulation-10-20`。当前阶段为固定双 Panda + DH116 的抓取控制开发、半隐式接触离线验证和 Isaac 接触前探针验证；持续夹持与抬升尚未通过实际验收。

本次汇总：前伸掌心朝下与阶段内对称姿态配置、分阶段速度控制、按渲染帧发布 wire 网格、接触粗筛与运动预测优化，以及实验半隐式接触求解器、杆 Jacobian/动力学适配和基准脚本。半隐式后端尚未接入正式 Isaac runner，尚无整机实时提速结论。

本次上传前验证：`bash scripts/python.sh -m pytest tests -q` → **726 passed in 26.94s**；`git diff --check` 通过。配置仍含本机路径；`forward_top_grasp_v1` 为历史未完成配置，使用前需核对各配置验证记录。下一步继续半隐式长时与实际指腹数值/性能验证、运行接线以及接近、闭指、夹持、抬升实测。

运行数据 `runs/`、历史归档 `archive/`、缓存和日志保留本地，未包含在此上传快照中。下方为历史阶段记录。

## 最新 Isaac 探针

独立单核 Isaac `closure-pilot` 已启动并正常步进，wire 保持在桌面上；该模式固定预姿态，约 `0.108 s` 仿真时间内四指负载均为 `0 N`，指腹距 wire 约 `19.5–22.4 mm`。它不是 approach，因此尚未进入闭指或抓取。输出：`runs/isaac_contact_probe_20261019/`。下一步运行带 pinned approach 的正式序列。

## 2026-10-19 最新数值验证

半隐式接触实验后端已完成 10 ms 静态与移动边界的步长减半检查（`2.5→1.25 µs`）。两种工况相对旧显式后端均满足位置、峰值力、冲量和线性反作用平衡门槛；两组新后端末状态差异为：静态位置 `0.33 µm`、移动边界位置 `0.33 µm`，反作用平衡均为 `0 N·s`。报告：`runs/implicit_contact_10ms_{2p5us,1p25us, moving_2p5us,moving_1p25us}_20261019.json`。

这只证明合成杆/网格工况的离线数值一致性；实验后端尚未切换正式 Isaac CLI，也没有证明 DH116+Panda 的真实闭指、持续夹持或抬升成功。

## 开发中：先轻触再夹持

路线1短时数值门已跑通：实验后端与旧显式后端各 0.001 s、内步长 2.5 µs 对照。静态工况位置差 `1.65e-7 m`、峰值力差 `0.00468 N`、冲量差 `2.83e-6 N·s`；10 mm/s 移动边界位置差 `6.43e-6 m`、峰值力差 `0.00523 N`、冲量差 `6.36e-5 N·s`；线性反作用平衡均为 0，短时门槛通过。证据 `runs/implicit_contact_short_{static,moving}_20261019.json`。这不是 100 ms 收敛或真实 Panda 抓取证明，下一步仍需延长时段并做步长减半。

路线1 Task3 已接入实验后端 `co_sim/implicit_moving_contact.py`：不注册旧 `RodMeshContact`，在每个 Verlet 子步联合桌面与四个指腹接触，使用真实杆质量/伸长响应和 Jacobian，一次写入节点力/材料力矩并记录网格反作用。专项 **7 passed**（含无接触与原显式后端逐状态对照、指腹反作用单次提交），全量 **726 passed in 24.88s**。尚未切换正式 Isaac；仍需多接触冲量平衡、细步长收敛和速度基准。

路线1继续：新增单次接触冲量提交适配器 `co_sim/implicit_stepper.py`，通过实际 Jacobian 将联合接触力转成广义力与冲量，明确由调用方只提交一次；5项专项测试通过，全量 **724 passed in 34.84s**。该组件尚未接入 MovingContact/Isaac，仍不能宣称速度提升或抓取成功。

最新实际杆适配：已实现带伸长因子的质量/转动响应及接触点Jacobian，8项先失败后通过，直接对照DeformX实际加速度核和接触力核（1/4/16子采样），全量 **719 passed in 26.09s**。下一步为一次施力的Verlet接线及反作用统计；目前尚未启用新算法运行，也未取得稳定夹持或抬升证据。

最新接触积分进度：详细方案已确认，新增独立联合半隐式速度求解器（尚未接入杆后端/Isaac）。19 项先失败后实现通过，补充旋转、转动力臂与分离检查后针对性 22 passed；本轮 `bash scripts/python.sh -m pytest tests -q` 为 **711 passed in 25.90s**。保留原摩擦律，多个接触共享速度求解，不收敛明确报错。下一步适配实际杆的质量/伸长转动响应、桌面及一次冲量提交，再做细步长对照。没有整机提速或抓取成功结论。下文待确认等记述为此前历史状态。

最新：独立指腹以 10 mm/s 竖直运动的 100 ms 工况中，2.5→1.25 µs 对比也通过容差（位置差 0.000649 mm，峰值力差 0.000122 N）。不是实际机器人闭环抓取证明。已去掉运动预测内部对已冻结输入的重复校验，保留公开输入及完整输出检查，并将预测模块加入后端指纹。先失败后实施，全量 **689 passed in 51.68s**；函数微基准约减时 46%，不是整机速度。旧 Isaac 未热加载，尚未稳定夹持/lift；新的稳定接触积分方案待确认。

最新参考解：静态双侧接触 100 ms、原摩擦参数下，2.5→1.25 µs 的位置差 **0.000625 mm**、峰值力差 **0.000950 N**、累计冲量差最大 **1.35e-7 N·s**，三个工程容差均通过。仅证明这一固定工况的细步长收敛；耗时约 47.5/94.8 s，不是实时提速方案或实际机器人抓取证明。生产内步长未改，后续需以此参考验证稳定求解及移动边界。

数值诊断补充：仅诊断关闭切向摩擦后，10→20 µs 的中心线差降至 0.000749 mm，表明主要敏感性在摩擦阻尼路径；生产摩擦未改。保留原参数时 5→2.5 µs 的位置差 0.123 mm 已满足位置容差，但右拇指峰值力差 0.0587 N 超过 0.05 N，整体仍未通过收敛门槛。后续需更细参考解或改进阻尼积分，不能把关闭摩擦当抓取方案。实际夹持/抬升仍未完成。

**最新精度限制：**100 ms 静态双侧接触内部步长实验中，20 µs 相对 10 µs 的线缆位置差约 1.585 mm；10 µs 与 5 µs 仍差约 0.548 mm，超过设计的 0.2 mm 容差。不能直接扩大内部步长提速，当前 10 µs 的该工况数值收敛也未确认。旧 Isaac 保持原参数运行作调试证据，不据此宣称最终抓取验收。新配置状态迁移尚未执行，旧状态未改写。

接触查询优化已实现：保守点到 AABB 距离粗筛编译化，剔除包围盒角落伪候选，不改真实网格、接触参数或采样数。全量 **686 passed in 32.96s**，另后端接触测试 **43 passed in 5.37s**。三组静态双侧后端交替基准耗时中位数 1.908→1.591 s，采样位置与累计冲量逐位一致；不是 Isaac 整机实时速度证明。现有旧实例未热加载，最新 step 47263/close，载荷仍零，夹持/抬升未完成。

最新后端诊断：固定真实指腹网格，外层 0.1/0.25/0.5/1 ms、杆内 10 µs 的落体/单指/双侧对比已完成。双侧四路峰值约 0.642/0.631/0.970/0.767 N，数值基本一致，但耗时 1.753→1.804 s，未测得提速。单次静态网格测试不含机器人反馈，不批准更大运行步长。下一步优先优化杆内部接触查询。实际 Isaac step 45802 仍为 close、载荷全零，未完成夹持/抬升。本轮针对性测试 32 passed in 0.51s。

最新状态：用户已接受候选 14 的倾斜前伸姿态，`configs/forward_top_grasp_v2` 在 `runs/forward_gui_render10` 实际运行，接近通过并保存闭指起点；本次检查 step 41119，仍在轻触闭指，未完成有效夹持/lift。当前实例曾发生 GUI 暂停恢复，不作为不间断端到端验收。

提速实施中：线缆视觉网格按渲染帧更新，静态拓扑只提交一次，物理/接触/保护仍逐步执行。新增测试先失败后通过，完整回归 `bash scripts/python.sh -m pytest tests -q` 为 **655 passed in 32.09s**。离线 USD 视觉发布基准 1001 次循环、3 次中位数为旧逻辑 3.25348 s、新逻辑 0.05485 s；不是整机加速倍数，当前 Isaac 未热加载新代码。分阶段动作提速、较大耦合步长及真实抓取验收仍未完成。用户已授权全核，覆盖历史单核要求；不能将开放 CPU 亲和性等同于自动并行。详见 grasp-speed 设计和计划。

最新组件进展：已接入显式 `--grasp-speed-profile staged`，包含实测间隙调速、载荷闭合、10 mm/s 候选同步抬升、恢复模式接线和关节速度分配日志。自由空间触碰速度包络最高 20 mm/s、最后 2 mm 为 1 mm/s、带载速度上限 1 mm/s；初始认证接近路径尚未重新定时。旧默认行为不变，当前实例未切换新模式。针对性 58 passed，全量 **682 passed in 27.41s**。离线当前姿态分配发现共同限速约 3.019 倍，候选拇指 9.327/1.000 mm/s 变为模型分配 3.089/0.331 mm/s；不是实测跟踪，不能称已达到人手速度。后续仍需速度/跟踪可行性、耦合对比和实际抓取验收。

以下为历史阶段记录，不能覆盖以上最新状态。

对称抓取调整实施中：镜像仅用于本任务抓取/lift，不约束后续非对称操作。功能坐标镜像、双侧独立运输速度分配、lift 高度协调和运行接线已实现；运行接线后完整回归 634 项通过（历史验证日志 `/tmp/dh116-symmetric-runtime-regression.log`）。`runs/symmetric_gui1` 实际首步因右拇指初态紧贴耦合下限失败，不是抓取成功。新 `configs/symmetric_grasp_v2` 在预抓求解中留 0.005 rad 关节余量，130 次离线路径检查及 USD FK 点 Jacobian 校验通过；已启动 `runs/symmetric_gui2` 实测，尚未完成闭指/夹持/lift 验收。旧证据保留，未放宽限位或碰撞保护。

最新演示：按用户要求切换到真实 Isaac Sim GUI，`runs/touch_first_gui1` 已显示双 Panda、DH116、桌面与橙色 DeformX wire，并开始实际步进。修复了 GUI 仍固定 `render=False` 的问题，增加观察相机及照明；未修改物理控制参数。针对性测试 3 passed in 0.14s，完整回归 **595 passed in 52.51s**；尚未完成抓取抬升验收。原 `touch_first_trial2` 在 step 6400 通过协作停止退出，原件保留，属于操作员中止而不是抓取算法失败。

独立目录实际启动发现并修复遗漏的共享接触检查导入；已补纯 DH116 模块且保持原检查逻辑。修正后完整回归 **594 passed in 24.51s**。`runs/touch_first_trial1` 为启动失败证据，`runs/touch_first_trial2` 已实际物理步进（检查时 step 361，settle），**尚未完成新策略接近/闭指/夹持/抬升验收**。新实例从初态验证新指纹版本，不把旧 checkpoint 冒充兼容恢复。

功能分支已接入“实测到位低速 → 拇指轻触 → 食指闭合 → 平滑增载”，包括实测关节速度、限速接管和恢复后重新确认接触。前次完整离线回归 **592 passed in 51.03s**（包含启动载荷校验修正）；**新版本尚未完成 Isaac 物理验收**，不应把测试通过当作抓取成功。旧控制器实例已结束，始终未热更新。细节见 CHANGELOG。

## 最近上传：2026-10-18

当前阶段：独立工程隔离与离线验证；实际双手接近试验进行中，尚未确认闭指、持续夹持和抬升成功。

本次改进：独立整理 DH116 控制、USD 依赖、接触配置和 DeformX 后端；增加上传排除规则及持续进度记录。本次本地回归 **559 passed in 51.52s**，运行配置预览通过，未启动第二个 Isaac。详细变更见 [CHANGELOG.md](CHANGELOG.md)。

GitHub 版本不包含约 4.7 GB 的 `archive/` 历史证据、运行日志及缓存；原件保留本地。部分测试和配置重建工具依赖该归档，**仅克隆仓库不能复现全部本地测试**。当前运行配置仍含此机器的绝对路径与指纹，异机需重新生成配置，不可直接修改校验值冒充兼容。厂家资料与第三方资源保留原有权利归属，本仓库不授予额外再分发许可。

当前为固定双 Panda + 普通 DH116 V2.0 + 桌面 DeformX 柔性线缆项目。不是 DH116S，不包含机器狗任务或 O6 资产。`fixed_dual_panda_o6` 命名空间内仅保留 DH116 使用的接触桥和线缆可视化两个共享模块。

## 当前迁移状态

- 1380 个静态来源文件已复制并 SHA 校验，原目录未删除；详见 `migration_manifest.json`。
- 主模型的 5 个 USD 层已隔离到 `assets/`，依赖相对引用；`OmniPBR.mdl` 由 Isaac 软件环境提供。
- 新运行配置已重新校验指腹表面坐标、接触几何、源路径 94 个区间和 loader 指纹。
- 主运行入口读取本目录的 DeformX；Python 启动脚本重置 PYTHONPATH，仅使用本项目及安装的软件库。
- 主运行、空载验证、资产生成及测试资源路径已适配，独立目录全量 **559 passed in 50.95s**，见 `regression.log`。共享测试中仅排除了不适用的旧 O6 资产专项测试，DH116 接触身份测试保留。
- **实际运行迁移尚未完全验收**：新目录实际 Isaac 验证尚未执行。原目录当前试验结束前不启动第二个实例。
- 原活动试验 `table_margin_raise1mm_trial1` 未纳入静态归档，结束后才能同步最终证据。

## 内容

`src/` 控制与共享依赖；`scripts/` 运行/校验；`tests/` 回归；`assets/` 新运行 USD；`configs/` 新配置；`vendor/DeformX/` 实际后端；`resources/manufacturer/` 厂家资料；`archive/` 历史原件；`docs/source/` 设计与记录；`tools/` 可审计迁移工具。

归档中的旧路径和 SHA 保持不变。它们不是新目录可直接恢复的状态；禁止修改旧 stage pin 冒充兼容。新配置的模拟参数不代表实机额定参数。

## 环境和命令

使用已有 `/home/xk/miniconda3/envs/go2/bin/python`、Isaac Sim/Isaac Lab、GPU0。可用 `DH116_PYTHON` 指定兼容 Python 环境；不复制整个 conda 环境。所有入口通过 `scripts/python.sh` 限定 core0，数值库单线程。

```bash
cd '/home/xk/coding/dexterous hand'
bash scripts/python.sh -m pytest tests/test_migration_copy.py tests/test_asset_isolation.py tests/test_standalone_layout.py -q
# 完整回归（包含真实 DeformX 后端测试，不启动 Isaac）
bash scripts/python.sh -m pytest tests -q
# 只打印命令并验证配置，不启动仿真
bash scripts/python.sh scripts/run_current.py
# 原实例结束后再运行；存在其他 DH116 verifier 时会拒绝启动
bash scripts/python.sh scripts/run_current.py --run
# GUI 版本需指定独立输出目录
bash scripts/python.sh scripts/run_current.py --run --gui --output runs/gui_trial1
```

这些入口尚待新目录实际物理验收，不能将命令可打印当作抓取成功。

## 抓取进度

**本次原实例已结束，整体未通过。** step 76165 的右拇指非指腹区域间隙下界降至0.99994 mm，触发1 mm保护；不是已确认发生穿透。最终 settle/approach/close/hold=PASS、lift=FAIL。最后完整步两站升高约1.139/0.219 mm、滑移1.332 mm，未完成目标抬升。保留全部阶段状态和失败现场，下一步修正右手对夹姿态并验证全程间隙；不放宽阈值。

原实例进一步在 step 68872 完成连续 1 秒 hold，保存 `stage_lift.json`，进入 lift。step 68935 检查通过，四指载荷约 0.965/0.918/1.040/1.043 N；仅刚启动抬升，**尚未证明线缆抬起或达到 30 mm 高度目标**。

2026-10-18 原实例 `table_margin_raise1mm_trial1` 在 step 58872 通过 close 并保存 `stage_hold.json`，进入实际夹持保持。step 58968 四指法向载荷约 0.948/0.896/0.992/1.014 N，hold 基线后滑移约 0.008 mm。这是原运行控制器的历史阶段证据，不代表新 touch-first 版本已完成 Isaac 验收。

旧恢复试验成功加载状态，但接近末段桌面余量不足触发保护，未闭指/夹持/抬升。当前调整为左手路径上移1mm，名义桌面余量约2.20mm；实际试验仍在原项目运行。未使用绑定伪造抓取，1mm安全阈值不变。
