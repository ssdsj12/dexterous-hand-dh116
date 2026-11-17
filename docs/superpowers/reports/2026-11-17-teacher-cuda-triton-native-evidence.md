# 原生第三层追踪：Triton注册帧仍引用CUDA初始化帧

## 单次已批准执行
执行11-17源码0c342aff1dfdc3855f3c5f7fd92c4d156a99cf2f，冻结SHA c75fd2880f4644088849dd5e8d194621cbc13e423706383f87aca8fd95454c0b。GPU2、单Isaac、1环境，仅2个SDK初始化步；0控制/求解器/模型加载/Teacher/PPO，无重试。授权同时包含保存完整栈、映射并核验PID后，仅清理本轮SIGSEGV故障进程。

两手初始化读回通过，34个collision owner各返回1个cooked hull；原严格校验仍0/34。SDK时钟在采样前、采样后及STOP后始终是2、0.004166666883975267s。

worker在66.965663s保存结果，app.close返回。随后约69.79s在PhysX Tensor插件SIGSEGV，主线程bt40实际可用20层栈包含Py_FinalizeEx；完整线程栈、动态库映射和原始数据已保存。按预授权核验并清理inferior725876及GDB725746，132.60s时监督器退出1；GPU2恢复9MiB，单实例锁释放。未自动重跑，也未运行下一实例。

原始监督器对外部手工清理不知情，force_termination_used仍false且gdb_returncode为0；本包以单独cleanup_receipt明确force_termination_used=true，绝不将GDB0当成被测进程自然退出0。

## 新的真实持有链
before_clear与after_stop均观察到同一条链，箭头表示引用的方向：

_register_triton_kernels frame → _lazy_init frame → SimulationContext构造frame → runtime factory frame/locals → 仿真与双手资源。

_lazy_init位于实际torch/cuda/__init__.py:346；它的上游是同文件_register_triton_kernels:1676。该位置是第二个torch._TritonLibrary.registerOp调用。两阶段frame ID一致，第三层枚举2个referrers，剔除诊断容器后显示1个真实上游frame，没有扫描截断。

STOP后manager字段为None、root ArticulationView已失活/回收，但两手Articulation/Data与共享SimulationView/backend仍活着。当前还没有观测_register_triton_kernels frame之上的最终持有者；不能据此把Triton注册本身定为SIGSEGV根因，也不能清空Torch算子表或禁用Triton。

实际安装源码还确认：torch/__init__.py:2630–2641定义_TritonLibrary，registerOp调用Library.define/impl并保存operator wrapper。包装函数不等于真实持有frame的最终容器；相关源SHA保存在readiness，无SDK源码修改。

## 五个额外CPU控制：没有复现原生持有
每组均为独立CPU子进程，CUDA_VISIBLE_DEVICES为空，没有Isaac。注册组使用真实Torch CPU dispatcher注册两个SparseCsrCUDA operator，但不执行CUDA kernel：

| 控制 | CPU scene是否被保留 | CUDA运行 |
|---|---|---|
| 无注册 | 否 | 禁止 |
| 构造函数内实际Triton注册 | 否 | 初始化入口fail-closed |
| 构造前实际Triton注册 | 否 | 初始化入口fail-closed |
| 构造内实际Python_lazy_init＋真实注册 | 否 | C++CUDA初始化用spy替代 |
| 构造前实际Python_lazy_init＋真实注册 | 否 | C++CUDA初始化用spy替代 |

最后两组确实执行了Torch的Python初始化控制流程，其Python_initialized标志暂时为模拟true；实际CUDA初始化器没有执行，结束前恢复子进程标志。不得把这个模拟标志当作GPU验证。

五个控制进程均exit0，但均未复现原生frame持有。因此“单独Triton注册Python流程必然泄漏”不成立；真实App/原生注册上下文仍需调查。不能将CPU阴性结果描述成原生修复或多线程根因已证明。

## 保护、发布与下一步
原生后相关CPU回归151 passed、0 failed、0 skipped；命令与日志SHA在readiness。

原始模型、528个保护文件、物理、USD、关节限制、动作和成功标准全部不变。Teacher准入仍blocked；原生预算已消耗。

下一步应一次性有界跟踪剩余frame-only上游链，直到看到Python容器或达到明确节点上限，避免每次只多一层的低效原生循环；不可见C++持有者仍标unknown。未来真正改变初始化顺序只能作为独立单变量对照，不是本包已经做出的修复。每个新原生版本仍需冻结后授权，PR2保持Draft。
