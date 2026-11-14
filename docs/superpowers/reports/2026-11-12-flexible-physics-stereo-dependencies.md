# 柔性物理、视觉采集与逐抓法 Teacher 依赖拆分

基线 `546130fe756d2a979d6dbf1fc6426eb25b759438`。本包仅变更调度依赖与 CPU 回归。

- 原生接触按左拇指、右拇指、单手对向接触、双手同杆递进。既有单刚体生命周期和已锁定指腹 SDF 属于左侧前置证据。
- `flex.f7-physics-state` 需要同杆接触、真实物理状态、原 24D 动作、Reset、终止和 R0 核验。Pinch、Tripod、Power 各自直接依赖物理状态。F8 聚合独立验收结果。
- `flex.f7-stereo-capture` 额外依赖当前任务相机的同步和退出资格。F7 保留为物理与同步采集的汇总入口。Student 同时需要该入口与合格 Teacher。
- WireSeg 子集及全量基准保留独立入口。旧 drop/hang 仍可在全量队列查询原始失败证据。两者均与无相机 Teacher 的执行无依赖关系。
- 所有新增原生包保持待验证。原生成功标准、动作合同、奖励、模型及历史证据沿用现有版本。

CPU 回归命令为 `python -m pytest -q -p no:cacheprovider tests/test_dh116_flexible_execution_dependencies.py tests/test_dh116_work_packages.py`。新增测试在基线因缺少独立接触包失败，拆分后通过。
