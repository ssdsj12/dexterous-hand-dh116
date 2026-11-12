# C5 telemetry hook 所有权单变量实验预检

基线 b2de6ec4a0e1879b55052b84e0a2a412d7a0cf7c。本工作包只读检查 Ubuntu 已安装 Isaac Sim 4.5 / Kit 106.5 源码和既有 C4 栈，新增独立入口、配方与 CPU 测试。没有启动 Isaac、GDB、训练或采集。C3/C4 旧额度已耗尽，C5 原生执行仍须精确冻结后的新批准。

## 发现与证据边界

既有 C4 在默认窗口不存在时，于 _extensions+0x1e445、_carb+0x4213d、_carb+0x425de、Py_FinalizeEx 路径 SIGSEGV，GDB 外层 255。来源为私有目录 evidence/native-attempts/flex.f6-camera-c4/a5d3eccdd4d0da813208b9ddafd7462ff11fe642e63a0c2ac7540d60009fcf0e/attempt-iqt7vw5q/native.log；日志 SHA 和映射换算见前一份 C3/C4 audit。

只读 objdump 检查已固定 SHA 的 _extensions ELF：0x1e431 从持有结构加载指针，0x1e435 清零该槽，0x1e442 读取 vtable，0x1e445 是 call *0x18(%rax)。这支持追查原生对象释放路径，但没有类型符号或对象地址证据证明该帧属于下面的 telemetry holder。不能把附近动态符号名称当作函数身份。

| 对象/路径 | 源码发现 | C5 处理 |
| --- | --- | --- |
| 原探针 _native_snapshot | manager、settings、viewport 等只在快照函数局部存在，输出为 JSON 标量/列表 | 复用 C4 读回 |
| SimulationApp | SIGINT 闭包持有 self；self 持有 _app、_framework、_carb_settings | 保持原样 |
| app_iface.get_app_interface | lru_cache 保留 IApp | 保持原样 |
| omni.ext._impl.get_extensions_interface | 函数属性 extensions 缓存接口 | 保持原样；当前 C4 是否填充未测 |
| telemetry_helpers._telemetry_hook | _initialize_telemetry 把 subscribe_to_extension_enable 返回值保存为模块全局；该文件没有清理该全局的路径 | 唯一实验变量：close 前设为 None |
| ExtensionModules.shutdown | 清理扩展实例 __dict__；应用退出时跳过模块移除分支 | 不修改 |

SDK omni/ext/_extensions.pyi 177–202 行将 subscribe_to_extension_enable 返回值描述为一对 holder，299–315 行说明 hook 在 holder 存活时有效。omni/kit/app/_impl/telemetry_helpers.py 7、13–33 行持有这对 holder；SimulationApp.close 563–623 行最终调用 _app.shutdown() 和 _framework.unload_all_plugins()，注释中的模块清理未启用。模块全局对象可因此活到 Python import teardown；旧日志没有记录这个特定全局的实时值。

**未证实假设：** C4 某个已注册 telemetry hook holder 延迟到插件卸载后析构，与 _extensions 退出故障有关。C5 只测试此候选。若运行时没有这对精确类型的 holder，本次实验无效并失败关闭；不得把未发生干预的自然退出记为修复。若 C5 成功，也只获得此诊断条件下的退出证据，不能单凭一次结果声称原始故障所有者已唯一确定。释放订阅会停止它的回调通知，这是该单项生命周期干预的直接效果；其他 telemetry 缓存保持原样。

## 独立实现与采样

- scripts/probe_dh116_camera_c5_ownership.py 使用原 C4 相同配置和 experience，只有上述模块属性赋值改变所有权。观测只通过已加载 sys.modules，不导入目标模块、不调用 getter、不新建 hook，不运行全局 GC。
- before_hook_release.json 只保存模块存在性、容器类型、holder 类型字符串和对象 ID。函数返回后不保留 native wrapper；CPU 析构测试验证释放先于 close。pair 不存在或类型不符、close 重新注册 hook、close 抛错均不能通过资格。
- 四个时点保存未经筛选的 /proc/self/maps：显式 update 前、释放前、close 前和 close 返回后。原 C4 窗口、experience、stage、库路径和扩展读回仍保留。
- scripts/dh116_camera_c5_debugger.py 固定 C5 GDB 命令。Py_FinalizeEx 断点保存自然结束路径的当前栈、全线程栈、共享库、GDB maps 和原始完整 /proc/PID/maps；stop() 返回 False 自动继续。SIGSEGV/SIGBUS 捕获同样资料并继续交付原信号。自然 finalize 不是退出成功证明。
- scripts/run_dh116_camera_c5_work_packages.py 导入原调度器，仅选择 C5 GDB 命令。原调度器文件未修改。普通旧入口不设置 C5 GDB 标记，C5 会在导入 SDK 前拒绝执行。
- GDB 自然采样重新读取 before_app_close.json 和 app_close_returned.json 到 gdb_finalize.json。配方要求全部 GDB 命令成功、完整 maps 非空、C4 设置读回符合、干预已发生及 close 后未重新持有；外层独立验证 inferior 和调度退出为 0。信号不能被内层完成标记覆盖。
- 采样错误明确记为错误，不制造回溯/映射。若故障发生在 finalize 前，正常采样文件可缺失，原生资格必须失败。

旧 probe_dh116_simulation_app_lifecycle.py、run_dh116_work_packages.py 和 C0–C4 五份配方逐字节与基线一致。旧探针 SHA-256：49e3812e14c279b5c4c7aaaf183edea69dadf20c32e61b986984b89dc36edf76；旧调度器：06d9a8e45a52961f198f4b5b752eb894f251cd41ef9995f36f32a5fee037f21b。不得经 C5 入口重跑旧已消费额度的格子。

## CPU 验证与保护

先写测试并实际观察 12 项因缺失 C5 实现失败；实现后 12 项通过。增加配方及自然采样产物校验后，实际观察缺失字段/配方失败。配方验证另发现 .gdb 不在发布路径白名单，改为独立 Python 命令模块后全通过。最初直接 Python 启动旧边界测试因缺少 launcher 的 src 路径发生 collection error；改用既定 scripts/python.sh 后旧基线 30 项通过，未改测试绕过问题。

实际命令：

    DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_camera_c5_ownership.py tests/test_dh116_simulation_app_lifecycle.py tests/test_dh116_render_capture_boundary.py

实际 **44 passed，0 failed，3.67 秒**：C5 14 项，旧生命周期与边界 30 项。GDB 回调在 CPU 替身下执行，覆盖正常/信号采样和错误记录；没有启动 GDB，不记为实际调试器或 Isaac 验证。git diff --check 通过。528 份保护文件与两份历史 Checkpoint SHA 均符合既有清单；SDK/物理源码未改。

## 冻结与待批准预算

配方 configs/dh116_execution/deformx_camera_c5.json：

- command spec SHA-256：b2111b376320ba4dfb4babd302034e77a0de12678f080d5386903ff335855322
- code/config SHA-256：94ffe53d74185ef49725924b95d0d9497953e21c081723a7ecddf57be952fc6d
- 上述身份使用本报告命令所列三份 package tests；调度器按源码集合排序。注册须使用相同测试集合，其他更改后须重新冻结。
- SDK 源码、接口 stub 和 ELF 输入 SHA 已加入配方。telemetry_helpers：b0c0ce45b8a1603a1b978382f3a93b0b7e9505d8ee9003e824b6c30055609a31；_extensions ELF：695ea005b3a39a433c13a4f589ed137a51b604a9563b6bfc031b6216d9a075a6。

拟议资格范围仅 camera_c5_telemetry_hook_lifecycle。**一次尝试、GPU 3、一个环境、180 秒、0 物理步、0 PPO updates、至多一次显式 app.update、0 采集帧/评估回合。** SDK 构造与 close 内部 update 沿用原行为，显式上限不等于 SDK 内部调用总数。保持最少 4096 MiB GPU 空闲、1 GiB 磁盘空间、独占 Isaac 锁及每指纹最多一次尝试。

批准后必须通过 C5 专用调度入口运行；实际 child argv 为 bash scripts/python.sh scripts/probe_dh116_camera_c5_ownership.py --output <唯一私有目录> --execute-native，GDB 由原调度 wrapper 选择。当前没有 C5 新批准或原生退出结果，RGB/深度/Native ID 正式双目采集仍阻塞。后续根据真实 owner 观测、映射、回溯和独立退出收据判断假设，不能直接转入采集或训练。
