# Kit stat-cache单次原生对照：缓存链解除，正常退出1，几何仍阻断

## 执行身份与预算
实际执行提交de84c4dfb018f9ef2ff79a06a7c3f537fd9f304c，独立入口diagnose_dh116_teacher_stat_cache_release。批准冻结SHA：2108a5bfbfc315f1032abd7aa80ad80023f2c0725e833fcd4a75c84bf5e62a8a，1366项冻结输入。GPU2、单实例、1个Pinch环境、2个SDK初始化步、34个碰撞体；0控制/求解器/Teacher/PPO/模型载入。只执行1次，180秒软上限；本次预算已经消耗，不重试。

原生证据位于服务器数据盘evidence/teacher-stat-cache-release-native-20261117T131229Z；外层监督与授权位于teacher-stat-cache-release-execution-20261117T131229Z。完整路径、命令和SHA见readiness。安装SDK、资产、物理、关节保护及成功标准均未修改。

## 三种结果不能混同

| 项目 | 本次实际结果 | 资格边界 |
|---|---|---|
| Python stat缓存 | 2841条→0；603个缓存异常；24条runtime-related；现有reset API调用1次 | 本路径实际观测到缓存清空；不是任意native资源释放证明 |
| 弱引用 | 原GC后左右Articulation/Data/view消失；STOP后SimulationContext与SimulationView/backend的Python包装均消失，manager为None | Python可见对象gone；没有直接测量C++析构内部状态 |
| 退出 | app.close返回；被测Python自行退出1；没有SIGSEGV或其他终止信号，监督器退出1；GDB本身退出0 | 正常非零退出，不是自然exit0；不授予正式生命周期或Teacher资格 |
| 几何 | 34个owner/34个stored hull读回，严格合格0/34 | 原拒绝原因仍保留，不能把“读回”写成“通过” |
| 资源 | 910609、910478均自行消失；单实例锁释放；GPU2返回约6MiB占用/24076MiB空闲 | 未使用预授权强制清理，没有SIGTERM/SIGKILL或重跑 |

worker保存结果用时68.513179s，外层71.615459s，未超过软预算。没有故障停点，故障取证分支没有触发，forensic.log为空；不伪造本次C++故障栈或动态映射。旧SIGSEGV栈/映射仍保留在此前证据目录。

## 本次首次直接观察到原生缓存载体
stat_cache_before.json直接读取当前运行中、通过源码与loaded-code核验的SDK cache字典：
- 2841条全部在4096条扫描上限内，entries_scan_truncated=false。
- 603条保存异常，其中24条具有runtime相关frame；按16条记录上限保存，exception_records_truncated=true。不能把16条写成完整603条异常轨迹。
- 一个真实FileNotFoundError的traceback/f_back链包含：
  _path_is_mode_type → _path_isfile → FileFinder.find_spec → PathFinder/importlib → _register_triton_kernels → _lazy_init → SimulationContext.__init__ → create_runtime → create_force_runtime → operation。
- 这次不是单纯猜测Triton注册，也不是只在CPU夹具中创建的外部持有者。原生cache本身确实保留了运行时构造调用链。没有读取f_locals或清除任意frame。

调用现有reset_stat_cache一次后，cache_entries_total=0，post-observer复核为0；最终写盘后只读guard正常返回，没有stat_cache_error.json，最终错误仅来自原严格几何拒绝。App update counter在API前后均16，SDK时间均2步/0.004166666883975267s，没有为释放缓存新增App.update、物理步或GC。

before_clear的CUDA上游frame探针不再发现_lazy_init；左右手/ArticulationData弱引用已gone。原clear/STOP后，SimulationContext弱引用gone，所有被观察SimulationView/backend包装弱引用gone，SimulationManager当前view为None。原有订阅显式unsubscribe、clear_instance返回；未使用臆测backend.release。

## 时钟可用性
初始化、capture前后、缓存API前后及before_clear最后可用SDK时钟均为2。STOP后SimulationContext已经销毁，clock_available=false，sdk_step/sdk_time为NA；这是对象消失后的不可读值，不能补零、沿用旧值或声称又测到了tick2。shutdown_clock_consistent字段的原逻辑没有把NA当作时钟逆跳；本报告单独保留NA，避免把不可读误写成实测未变。

## 几何对照和保留错误
按queried_prim_path匹配，再对每个cooked_hulls[0].raw做完整JSON哈希比较，本次34/34与上一轮e7397e...运行完全一致；不是比较不存在的hulls字段。本次没有重Cook或更改网格来改变数据。

原persist_capture继续抛出incomplete 34-source cooked capture: ONE_OR_MORE_COLLIDERS_CENSORED，具体collider包含COOKED_HULL_TOPOLOGY_INVALID_0。完整readback和原错误保留在stored_cooked_capture.json/result.json。capture_complete、process_exit_verified、native_lifecycle_qualified、grasp_qualified和teacher_admitted保持False。并未为了取得exit0吞掉异常。

## 可支持的结论与限制
此次受控变更解除了一条真实Python缓存持有链，并在这一条初始化/几何失败清理路径中避免了先前finalize SIGSEGV。这比CPU模型增加了直接原生证据，但只有1次、不构成重复性证明，也未证明所有相机/求解器/长回合都能安全退出。C++释放没有单独插桩，native_binary_release_instrumented=false；完整工作负载SIGSEGV的唯一根因仍不宣称已证明。

Teacher未执行策略查询或PPO，不能获得抓取、Lift、训练或示范资格。严格几何0/34仍然阻断后续接触资格。下一项是继续离线核对stored cooked polygon表示与严格validator的API/坐标语义，保留现有校验、原网格与失败证据；若需新的原生运行，应先给出实质修正、CPU证据和新的冻结预算，不复用本次批准。

## 回归与发布
原生之后相关CPU回归188 passed、0 failed、0 skipped（包含原Registry55及新增缓存26，彼此重叠，不能加到此前联合522当作新增独立测试）。确切命令、env、elapsed与日志SHA见readiness。无额外原生、无GPU训练。

528个保护文件与三份旧模型逐SHA核验；两个历史/tmp Checkpoint此前已不存在，仍不谎报校验通过。原始图像、原生JSON大包、GDB日志和模型只保存在数据盘。报告/readiness/Registry与必要摘要经Ubuntu commit、Windows导入同commit push并核对远端；PR2保持Draft。

元数据更新后的Registry CPU回归55 passed、0 failed、0 skipped；与188项套件重叠，不累加。独立只读复审无证据分类阻断；其未重算大型raw hull，34项raw哈希比较由主代理执行并保留汇总与原始SHA。
