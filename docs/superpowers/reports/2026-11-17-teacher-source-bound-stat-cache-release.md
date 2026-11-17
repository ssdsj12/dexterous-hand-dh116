# 源码绑定的Kit stat缓存释放：独立诊断修正版

## 范围与已知结果
基线98ea761adb510d3fbe6d1d7b778a7a9abea55132，Ubuntu工作树/mnt/mydisk/xk/projects/dh116-teacher-display-owner-20261117，分支11-17。沿用已发布的Kit缓存CPU机制证据；本包未启动Isaac、CUDA、Teacher或PPO。上一轮冻结e7397e4322d4f97f648b7517addfc74407c156d845bbc70437e40134686eafb7的单次原生预算已执行并耗尽，故障进程已按预授权清理，不能用于本修正版。

新入口是scripts/diagnose_dh116_teacher_stat_cache_release.py。旧diagnose_dh116_teacher_initialization.py默认仍不传hook；不存在旧模型自动启用此修正的行为。

## 实际修改
- tabletop_stat_cache_release.stat_cache_source_contract绑定当前安装stat_cache.py、ext_settings.py和_internal.py的真实SHA。
- _validated_fields除文件与module身份外，将实际loaded function code/defaults/globals与固定源码直接compile得到的code比较；不exec安装模块，不运行SDK导入。settings检查实际module、lru_cache wrapper、__wrapped__、_get_setting及reentrant predicate正文与身份。此项是Python源与载入正文核验，不等于原生对象/二进制释放资格。
- snapshot_stat_cache最多扫描4096条、保存16条异常记录；traceback上限32、每链f_back上限32、每异常最多128个标量frame描述。只保存类型、ID、函数、路径和行号，不读f_locals、不保留异常/frame对象。key仅记录hash。
- release_owned_stat_cache要求缓存已经启用且importlib hook已激活；不启用关闭的功能。读实际SDK2与App update counter，调用现有reset_stat_cache最多一次。API只读counter，不新增App.update。
- verify_stat_cache_empty在after observer后和最终receipt写盘后复核。最后一次clock reader返回后直接检查当前真实dict长度，之后不再写盘或调用observer；缓存重建、时钟变化均失败，不再reset。
- make_before_collect只使用已加载的SDK module与弱引用SDK clock owner，写独立before/after/release/error JSON并flush/fsync。release receipt明确标记尚需完成post-write guard，不能用该文件的存在单独证明最终guard成功。
- run_owned_call新增可选before_collect=None，在原同步、异常traceback处理和owner字典释放后，原唯一GC之前调用。普通hook错误仍经原GC/checkpoint；中断经原GC/checkpoint后传播。没有新增生产GC、没有调整原STOP/App.close顺序。

## 测试先行及审查
使用当前安装SDK的真实Python函数AST、真实importlib缺失扩展路径检查、明确CPU fake Kit App/settings。没有创建GPU context，也没有用手工外部持有者代替这个缓存机制。

26项新测试覆盖：真实cache持有Scene及一次reset后的回收；源SHA、loaded module、code正文、settings predicate拒绝；disabled/inactive hook拒绝；SDK步数、App counter、时钟变化；弱引用owner；诊断IO失败、中断；原同步→reset→原GC→checkpoint顺序；独立版本与冻结指纹拒绝；after observer、receipt writer与最后clock reader正常返回但重建cache时的失败关闭。IO故障仅添加错误，不能绕过已有清理。

测试先行记录保留在外部证据目录：
- 初始缺API：red.log，16 failed。
- 扩展hook/CLI：red-reviewed.log，20 failed。
- 独立审查两类缺口：red-review-gaps.log，4 failed、21 passed。
- 最后clock-reader窗口：red-last-reader.log，1 failed、25 passed。
- 最终专项：green-final.log，26 passed、0 failed。
- green-initial.log的1次失败是CPU fixture使用不可weakref的SimpleNamespace；改为可weakref的ClockSim，并额外测试拒绝不可weakref对象，没有加入强引用fallback。
- green-review-gaps.log保留1次错误计数断言：worker多次只读clock被误计为reset调用。现按实际refresh_stat_cache调用身份单独计数；执行行为不变。
独立Agent只读复审确认上述三项窗口已闭合，未执行SDK/GPU。
最终联合与Registry结果及确切命令、env、日志SHA记录于readiness。

## 接口约束与失败语义
独立CLI默认只freeze；真正execute必须携带重新计算一致的完整批准SHA、独立版本、真实源码来源、独占新输出目录和相同参数身份。新入口复用旧AppLauncher、单实例锁、原runtime与readback，不复制另一个仿真控制器。

严格几何校验不变。上一轮34个stored hull均读回但0/34严格合格；即便本修正去掉SIGSEGV，原几何失败仍必须正常报告非零退出，不能改成exit0。app.close返回、Python缓存清空、Python对象回收、native view释放、外层自然退出是不同资格。当前native_release_verified、native_SIGSEGV_cause_proven和teacher_admission全部False。

## 保护与后继
528个保护文件、原三种模型、安装SDK、USD、材料、碰撞、执行器、动作/观测合同、30mm/5s/0.5mm标准保持原样。历史两个/tmp Checkpoint已不存在，不能声称读回校验通过。

本包先CPU回归、保护校验、Ubuntu commit，再Windows导入同commit正常push并核对SHA/tree/文件清单，PR2保持Draft。原始日志、模型、264KB完整freeze plan、故障栈与数据均只保存在服务器证据盘，不上传原始数据。

后续只申请一次新原生反事实诊断：GPU2、单实例、1环境、2 SDK初始化步、读取34 owner，0控制/solver/Teacher/PPO，180秒软上限、不重试。旧预算不复用。新freeze SHA在发布核验及全部readiness写入完成后生成。原生自然退出与严格几何仍须分开判定，不自动进入训练。

最终联合CPU回归522 passed、0 failed、0 skipped（含26项新增）；Registry元数据回归55 passed、0 failed、0 skipped，55项与联合套件重叠，不累加成577。保护528/528一致，原Pinch/Tripod/Power三模型SHA一致。

源码提交462e241239bda7aae58b7494ae18e7e9e559e24b已由Windows导入并正常push，远端tree58913ad714be3a1be4fdc2ae63dfc5e0bbd74dae、12文件一致；PR2 Draft=true、merged=false。此后仅追加发布核验元数据，不改变已测试源码。
