# DH116阶段状态、柔性杆恢复及运行时接线

父节点：[T500/D4阶段恢复](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。计划：[stage-reset](../../docs/superpowers/plans/2026-10-16-dh116-stage-reset.md)。Inline执行，全部数值测试core0/单线程；无新Isaac实例，无实机操作。

## 输入、输出与合同

- stage_state.py：独占原子发布JSON，外部SHA加载；完整named q/qd/target、固定根、限位、杆动态/原始rest数组、固定材料ID、锚点、落定高度、载荷、资产/代码/配置pins与前序资格。
- 外部后端`co_sim/free_rod.py`：export_physical_state/restore_physical_state；MovingContactRodEngine继承，必须全新实例且只能注入一次。模型/接触/网格身份匹配，静止数组必须等于原模型；写动态数组后重算几何/应变/内部力、清旧外力。不重放冲量，不将变形杆当无应力模型。
- stage_runtime.py：StageResetSequence独立restore_validate，接近/闭指重新验证.1s，保持/抬升重新验证1s预载与保存锚点滑移；物理时钟从0开始，轨迹查找使用单独局部阶段坐标。固定根匹配、live限位匹配后按名字写q/qd，刷新kinematics。
- runner：--save-stage-states在实测相邻阶段转移、前序资格与已执行点J探针通过时导出stage_approach/close/hold/lift.json。--restore-state与--restore-state-sha256仅新实例使用；来源标IMPORTED，阶段试验不得授予continuous_grasp资格；失败不自动重启。
- 没有降低碰撞、压入、速度、滑移门，没有绑定wire；不改夹力/材料/硬件假定。

## 验证

TDD：状态工件最初12缺模块RED，后补缺model SHA拒绝RED；真实后端8缺接口RED；阶段运行时7缺模块RED，后补runner接线和非法高度测量RED。各轮实现后通过。

真实后端对照：变形杆保留rest_lengths/rest_sigma/rest_kappa，恢复后应变/内部力相同；重力单步位置/速度一致；四网格接触后端新实例缓存为空、外力清零；非零接触的继续推进与新实例恢复推进位置/速度/director/omega/冲量在测试容差内一致。不是PhysX精确断点续算证明。

首轮DH116回归406 passed/2 failed：厂家Excel旧路径/home/xk/下载不存在。只读查找确认两文件在/home/xk/桌面/dh116，改测试环境路径而非产品代码。

最终命令：OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 DH116_TABLE_SOURCE_DIR=/home/xk/桌面/dh116，PYTHONPATH含当前项目和Isaac omni.usd.libs、LD_LIBRARY_PATH含go2/lib、USD/bin、Isaac kit；taskset -c 0 /home/xk/miniconda3/envs/go2/bin/python -m pytest tests/test_dh116*.py tests/test_fixed_dual_panda_o6_free_rod.py tests/test_fixed_dual_panda_o6_moving_contact.py -q --tb=short。

结果：**489 passed in52.14s**，完整输出/tmp/dh116-stage-reset-regression-v2.log。实际非法CLI缺restore SHA退出1，报external SHA required；未创建输出目录、未启动Isaac。审查方式为Inline自查，没有独立子代理审查。

## 实际试验边界与下一步

原trial2_load115 session90016/PID15485持续，最后核验step18806、approach、passedTrue。没有重启、没有热加载新代码。旧进程启动时没有stage导出功能，旧JSONL缺原始完整恢复合同，不能假称可直接恢复。

下一实例才启用--save-stage-states，生成真实合格入口后使用--restore-state PATH --restore-state-sha256 SHA；沿用同配置/代码pins，输出新目录。当前旧实例结束前不启动第二个Isaac。Task5实际恢复瞬态/完整连续夹持抬升/3重复未验收；下降放回仍未实现。

Git Baseline/Candidate Ref：5e3bcc4 +既有dirty worktree，未提交。Key Files：stage_state.py、stage_runtime.py、grasp_sequence.py、verify_fixed_dual_panda_dh116.py及对应测试；外部PyElastica-Mesh/co_sim/free_rod.py亦未提交，需随项目部署保留。
