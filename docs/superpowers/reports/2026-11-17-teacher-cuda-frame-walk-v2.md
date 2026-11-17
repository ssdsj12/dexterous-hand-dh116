# 有界CUDA上游frame链追踪v2

## 为什么继续扩展
基线2b343d5e9fe13cfc2e1c2eebbfc10763c08e048f。前次一次已批准原生实验已发现_register_triton_kernels frame引用_lazy_init frame，但最终保留者仍未知；五个隔离CPU控制没有复现原生保留。不能因此禁用Triton、清空算子注册表、删除SDK字段或宣称根因已找到。

本包让下一次诊断在明确上限内追踪剩余frame链，避免每个原生批次只增加一层。本包没有新的Isaac、CUDA初始化、物理、控制、求解器、Teacher、模型或PPO执行。

## 实际代码
probe_simulation_owner_edges新增cuda_frame_owner_budget，严格整数1..8。trace_cuda_init_owners默认false时仍为原两层。旧的true且不传预算模式仍只观察1个frame。独立诊断worker显式选8，仍在同样before_clear/after_stop边界写入原文件。

_probe_cuda_frame_owners改为有界队列：从一个_lazy_init候选根开始，只把可见的frame referrer加入队列，记录非frame持有者的受限描述。按ID去重，重复边和环不会重新扫描。每个node记录parent_frame_id和depth_from_cuda_frame，可还原观察到的图边；这些是单进程瞬时ID，不是永久身份。

本worker每次probe最多扫描8个不同frame，每个frame最多描述16个referrers。两个既有边界最多16次新增gc.get_referrers枚举。最大hop上界为10（原2层＋8个frame节点），不是声称实际观察了10层。达到节点上限时保留pending数量和frame_walk_truncated，不把未跟踪候选当作不存在。默认1节点也如实标记后续候选被上限截断。

gc.get_referrers本身会枚举所有直接Python referrers，枚举大小/耗时不具常数上界。遍历过程中会短暂持有frame别名，临时待处理节点数量有保守上界1+8×16=129；返回及异常时全部别名释放，返回JSON不持有frame、资源或native handle。不可见C++持有者和没有可见Python owner始终unknown，native_release_verified/root_cause_proven/native_owner_qualified不因此变true。

不读取f_locals、不清除frame、不新增生产gc.collect、不调用native销毁方法。默认训练路径、SDK、清理顺序和物理不变。冻结输入新增实际torch/__init__.py与torch/library.py，连同已有torch/cuda/__init__.py绑定注册实现。

## 测试与限制
11项先失败，再最小实现后与旧10项共21项通过。联合496 passed、0 failed、0 skipped；准确命令、日志和SHA在readiness。

验证包括真实Python frame链到字典持有者、清理模拟持有者后资源回收、1节点向后兼容、1..8严格参数、2节点截断、重复边/环、异常路径释放别名、worker预算接线和新增源码pin。CPU测试使用GC检查模拟对象，生产路径没有新增GC。

这些结果只证明观测器的CPU行为，不能替代真实上游持有者、自然退出或Teacher成功。此前原生失败证据和消耗的预算保留，禁止用新字段回填旧记录。

## 下一次原生范围
完成发布并冻结后再请求新一次批准：GPU2、单实例、1环境、2 SDK初始化步、34个collision owner读回；0控制/求解器/模型/Teacher/PPO。180秒软上限，失败不重跑。监督器不自动kill；如获明确预授权，保存完整栈和映射、核验PID后仅清理该回执绑定的SIGSEGV故障进程，并记录为失败。

这次变化仅增加有界Python持有链观察和源码pin，不改变原严格hull校验、碰撞、执行器或30mm/5s/0.5mm成功标准。正式Teacher仍blocked，PR2保持Draft。
