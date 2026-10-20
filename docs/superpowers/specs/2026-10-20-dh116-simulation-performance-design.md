# DH116 仿真性能优化设计

日期：2026-10-20  
范围：固定双 Panda + 双 DH116 V2.0 + Isaac Sim/Isaac Lab + DeformX Cosserat wire  
基线：分支 `feat/dh116-simulation-10-20`，单核启动约束保持为 CPU 0

## 目标

在保持桌面 wire、接触反作用力、碰撞保护和抓取证据完整的前提下，减少重复的主机工作，获得可复现的阶段性能数据。优化首先面向稳定的 `settle`、`approach`、`close`、`hold`、`lift` 阶段，不能用更早退出、减少安全检查或放宽成功条件换取速度。

本轮的直接交付物是：

- 每个物理步和每个阶段的分段墙钟计时；
- 可选择的 wire 显示限频、状态快照/缓冲、证据日志缓冲；
- 与原基线匹配物理时长的三次对照结果；
- 环境、源提交、设备和线程策略的可复现记录。

## 保持不变的物理边界

- 外层 `physics_dt=1e-4 s`，线缆内部 `py_dt=1e-5 s`；
- 16 点空间接触采样、接触反作用力、重力保持和 source-clearance/collision guard；
- 每个物理步的 post 状态安全检查和一次反作用提交；
- 四路有效接触、连续夹持、滑移、抬升高度及整根 wire 离桌判据；
- `scripts/python.sh` 的 `taskset -c 0`、数值库单线程和单 Isaac 实例；
- 原始逐步 JSONL 证据的行数、字段和阶段时序。

改变步长、接触采样、求解器迭代数或控制频率属于后续独立数值实验，不混入本轮等价性能优化。

## 组件与数据流

### PerfMeter

新增 `src/go2_pvcnn/control/dh116/perf_meter.py`，使用单调墙钟计时，不主动插入 CUDA 同步。每步记录：

- `step`、`stage`、`completed`、仿真时间和墙钟时间；
- `control`、`rod_and_contact`、`source_guard_pre`、`reaction_submit`、`physx_step`、`state_and_sensors`、`source_guard_post`、`wire_visual`、`evidence`；
- 每段耗时及总步耗时。

计时器同时维护阶段计数、平均值、P50/P95、SPS 和 RTF，结束时写入 `perf.jsonl` 与汇总文件。异常步必须带 `completed=false`，不能被当成正常性能样本。

### pre/post 快照与缓冲

在控制前建立本步唯一的 pre 快照，在 `sim.step` 和 `arm.update` 后建立独立的 post 快照。快照带 `step`、`stage`、`sample_time` 和唯一 token。只读消费者共享当前快照；写入日志或阶段状态的对象复制数据，避免后续 in-place 更新篡改历史。

初始化时缓存关节名到索引、body 索引、主动关节映射、耦合矩阵、指腹局部点/法向和反作用提交数组。每步清零并复用反作用缓冲，仍只提交一次当前步的力和力矩。

### wire 显示调度

保留初始化时的网格拓扑和第一次发布。固定 40 单元、环向采样不变时只更新 points 和 extent。GUI 以墙钟 `ui_hz` 调度 wire 发布和 `sim.render()`；物理步、接触、保护和 post 证据仍每步执行。headless 模式不进行显示发布。

默认显示频率为 15 Hz，并记录实际刷新次数和 `visual_publish_mode`。显示调度不能调用第二次 `sim.step`，也不能让相机观测频率与显示频率混用。

### 证据日志策略

增加 `evidence_flush_policy`：

- `audit`：沿用每步 flush 的原始模式；
- `buffered`：仍每步编码和写入 JSONL，但每 512 行或 0.5 墙钟秒 flush，阶段转换、异常、STOP_REQUESTED 和正常结束立即 flush。

文件独占创建、非有限数拒绝、最后一行完整性、异常上下文和阶段状态保存保持不变。buffered 模式的潜在尾段丢失必须在 manifest 中明确记录，不能覆盖 audit 证据。

## CLI 与运行身份

新增性能运行入口或等价参数，保留当前 `run_current.py` 的原始默认行为。性能配置必须显式记录：

- `device`、`tensor_device`、`physics_dt`、线缆 `py_dt`、接触采样；
- CPU affinity、OMP/OpenBLAS/MKL/Numba 线程数、Kit/PhysX 线程配置；
- Isaac Sim、Isaac Lab、PyTorch、Numba、驱动版本；
- `ui_hz`、显示模式、flush 策略、随机种子、源提交和本地差异摘要。

本轮不取消单核限制。CPU/GPU 后端比较作为独立实验入口，不能混入默认抓取运行。

## 对照实验

每次只改变一个变量，使用相同资产、材料、初态、种子、阶段和物理时长；同一机器串行运行，每组至少三次，报告中位数、P95、冷启动和波动：

1. A0：原始基线；
2. A1：wire 显示调度；
3. A2：pre/post 快照与缓冲复用；
4. A3：buffered 证据写入；
5. 若 profile 支持，再单独测试 CPU/GPU 后端或接触热点。

加速比只对匹配阶段或相同物理时长计算。若候选更早触发保护或失败，实验标记为不可比。

## 验证与回退

每个候选阶段运行完整回归，并增加/保留以下负例：临界桌面间隙、非目标指腹接触、载荷丢失、滑移超限、NaN、日志写入失败和 operator stop。对照报告必须比较：

- 完成步数、阶段和终止原因；
- 四路接触负载、反作用冲量平衡、wire 穿透和滑移；
- 最小几何间隙、source sweep 和非目标接触；
- wire 中心线/姿态/速度误差；
- 阶段 SPS、RTF、平均和 P95 单步耗时。

发现漏检、重复施力、时序混用或关键物理指标恶化时，只回退对应独立提交，不修改历史配置 SHA 或运行证据。

## 未纳入本轮

不在本轮改变 `physics_dt`、`py_dt`、接触采样、材料参数、摩擦参数、求解器迭代、抓取目标载荷或双手动作策略；不以关闭 GUI、关闭反作用力、减少碰撞保护或绑定 wire 端点作为性能方案。

