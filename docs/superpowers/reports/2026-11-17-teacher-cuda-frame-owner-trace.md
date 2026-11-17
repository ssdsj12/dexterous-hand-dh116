# CUDA初始化帧上游持有者：第三层诊断准备

## 原生问题与本包目的
基线11-17 HEAD dd90dca58edbc74329fd6e049a7dad8cea43205a。上一轮一次原生实验确认STOP后manager字段已清空，但双手Articulation/Data仍保留共享SimulationView/backend；构造帧的上游出现torch.cuda._lazy_init。SIGSEGV仍发生在app.close返回后的解释器finalization阶段。上游CUDA帧由谁保留尚不清楚；这些观察不等于崩溃因果已经证明。

本包仅增加该帧上一层的只读诊断和CPU验证。没有运行Isaac、初始化CUDA、执行控制/求解器/Teacher/PPO或改变清理顺序、SDK、物理、资产与模型。

## 实际修改
- tabletop_simulation_owner_cleanup.py：新增_probe_cuda_frame_owners；probe_simulation_owner_edges增加trace_cuda_init_owners=False。默认仍为原两层行为。显式开启时从已扫描的上游frame中选取函数名_lazy_init的候选，最多跟踪1个frame，再记录其Python referrers。
- diagnose_dh116_teacher_initialization.py：原before_clear/after_stop写入点开启此选项；不新增物理步或生命周期边界。冻结输入增加实际安装torch/cuda/__init__.py，避免源码变化被静默复用。
- test_teacher_cuda_frame_owners.py：先写10项失败测试，全部因功能、接线或源码pin缺失失败；最小实现后10项通过，联合485项CPU回归通过。

## 边界与资格
每次probe最多新增1次gc.get_referrers调用；两个既有边界合计最多新增2次。每层扫描/描述最多16个referrers。gc.get_referrers自身仍枚举所有直接Python referrers，不能声称整个枚举过程有常数耗时上界。

第三层排除当前探针的临时list和自身frame；输出只有ID、类型、函数/文件/行号、受限匹配键和数量，不持有实际资源。已有字典描述最多扫描128个键、输出8个匹配键，扫描不完整会标记。生产代码不读取f_locals，不清除frame，不新增gc.collect，不调用native方法或CUDA初始化。

函数名选择只用于诊断，不构成源码/原生资格。输出明确保留实际frame文件名；未知持有者、超出描述上限和看不到C++持有者的情况不得解释为已释放。重复候选与额外未跟踪候选有计数。root_cause_proven、native_release_verified和native_owner_qualified保持false。

## CPU证据
10 red → 10 green。联合485 passed、0 failed、0 skipped，包括此前相关force/初始化/几何/ownership/worker/清理和截图建议验证。

真实Python frame构成的CPU模型显示：第三层能定位持有_lazy_init frame的external字典，结果JSON可序列化且不持有对象。清除模拟外部持有者后，在测试内GC确认资源回收。测试中的gc.collect不是新增生产清理。另验证默认两层不扫描第三层、严格布尔开关、扫描上限、重复候选只追踪一次、看不到Python referrers时仍标unknown、worker接线与实际Torch源码pin。

确切命令、环境、失败日志、通过日志和SHA在readiness。本包是观测能力的CPU资格，不能作为真实上游持有者或SIGSEGV已经修复的证据。

## 后续最小原生观察
完成Windows发布和最终冻结后，再申请一次新预算：GPU2、单实例、1环境、2 SDK初始化步、读取34个collision owner；零控制、求解器、模型加载、Teacher和PPO。保持180秒软上限、无自动重试/强制退出。旧原生预算已耗尽，不复用。

新证据仍通过before_clear.owner_edges.json和after_stop.owner_edges.json输出，在selected_owner_edges的upstream_frame_owners中包含第三层。不得为旧实验回填该字段。若持有者来自不可见C++引用，只能记录未知并准备下一项有证据依据的对照，不清理任意Torch/SDK全局或改碰撞资产。

主线程提前CUDA初始化仍只是候选，未实现或执行。Teacher准入与30mm Lift、5s Hold、0.5mm Slip标准保持不变。PR2保持Draft。
