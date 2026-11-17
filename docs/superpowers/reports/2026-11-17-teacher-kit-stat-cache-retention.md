# Kit文件stat异常缓存：可复现的frame保留机制

## 本轮原生结果（仍失败）
执行源码5d37b09b64cb7b4cc8ac00fb4680d6b827a012a4，冻结SHA e7397e4322d4f97f648b7517addfc74407c156d845bbc70437e40134686eafb7。GPU2、单实例、1环境、2 SDK初始化步、34个碰撞体读回；0控制/求解器/模型/Teacher/PPO，无重试。

worker结果在67.012720s保存，app.close返回，约69.71s仍在PhysX Tensor插件SIGSEGV。bt40请求得到的完整可用主线程栈含Py_FinalizeEx。按预授权保存栈、映射并核验PID后，仅清理inferior750298和GDB750168；监督器约171.03s退出1。被测进程没有自然退出0，GDB返回0不代表通过。两步SDK时钟在采样前后及STOP后保持2、0.004166666883975267s，未推进额外物理。

34个owner各返回1个hull，严格校验仍0/34。两手初始化读回通过不代表接触、抓取或生命周期通过。原生失败证据保留，预算已消耗。

## 新的原生frame链
两个边界均观察到同一组8个frame：

_lazy_init ← _register_triton_kernels ← importlib.util.find_spec ← _find_spec ← PathFinder.find_spec ← _get_spec ← FileFinder.find_spec ← _path_isfile。

箭头表示右侧frame引用左侧调用者。每次达到8节点上限后还有3个候选未追踪，frame_walk_truncated=true；不声称已经观察到最终持有容器。其中_path_isfile的下一层是_path_is_mode_type，位置与文件stat异常处理相符。

## 安装源码审计：不是猜测backend.release
实际安装的omni/ext/_impl/stat_cache.py：
- _os_stat_cached同时缓存os.stat返回值和异常实例；缓存值为(result, exception)。
- 再次抛出缓存异常会保留traceback，traceback的frame及f_back可继续保留上层调用栈。
- refresh_stat_cache在App update number变化时清空字典，并把importlib._bootstrap_external._path_stat指向缓存函数。
- 现有reset_stat_cache将缓存update number置为-1，再调用refresh_stat_cache；它读取update number，不调用App.update。

ext_settings.py默认开启/app/extensions/pathStatCacheEnabled；_internal.py在extension预启动路径调用refresh_stat_cache。完整源SHA在readiness，安装源码没有修改。本轮原生没有直接读取cache字典或调用reset，因此尚未把某个原生cache entry与frame ID绑定为最终载体。

## 三个严格区分的CPU控制
使用实际安装stat_cache.py的Python函数和真实Torch CPU dispatcher注册；仅Kit get_app/update-number对象为CPU stub，CUDA初始化入口fail-closed、CUDA_VISIBLE_DEVICES为空，无Isaac或GPU上下文。

| 模式 | GC后Scene仍存活 | reset后/第二次GC | cache |
|---|---|---|---|
| 不启用SDK cache | 否 | 否 | 0 |
| 启用cache并保留 | 是 | 仍是 | 12条，其中3个FileNotFoundError |
| 启用cache，再调用现有reset_stat_cache | 是 | 否 | 12条降至0 |

全部3个子进程exit0，实际App update=0、CUDA初始化=0。异常traceback中包含_path_is_mode_type和_os_stat_cached，与原生已观察链相符。这明确复现了一个持有机制，并证明现有API能解除CPU模型保留；不是任意外部持有者的手工模型。

结合前一包“单独Triton注册不保留Scene”的五个CPU控制，缺失的环境因素指向Kit stat cache。CPU因果已复现；真实SIGSEGV是否由该cache造成，仍需原生单变量对照。不得把这两个资格混为一谈。

## 修正方向与保护
可准备一个默认关闭、独立入口的修正版：在runtime引用释放及CUDA同步之后、现有ownership/GC检查点之前调用源码绑定的现有reset_stat_cache；采集cache前后标量与SDK时钟，不新增生产GC或App update。旧入口保持原样，不编辑SDK、不清除任意frame、不调用虚构backend.release、不关闭保护或重做碰撞资产。

原生后相关CPU回归162 passed，0 failed，0 skipped。确切命令、环境与日志SHA在readiness。

原始模型、528个保护文件、物理/材料/执行器和30mm/5s/0.5mm标准不变。Teacher仍blocked。只有完成代码、CPU回归、发布、冻结并获新预算后，才执行修正版原生对照。PR2保持Draft。
