# 实际Teacher Worker异常路径缓存释放修复
基线38bd608222e57c463171a0f0629e6193d14a25c1；只改实际Worker的Python生命周期组合，不改物理、SDK或模型。

## 已证实的缺口
v5真实退出栈在PhysX Tensor plugin/_physicsTensors/Py_FinalizeEx；after_app_close快照仍持constructor帧、hands与native SimulationView。
actual_provider._run_isaac.acquire原先只有_acquire成功才cache_release；异常绕过这一调用。
v5目录确实没有stat_cache_release.json。本包修复该明确的路由缺口，不把静态关联当成已证明的唯一C++根因。

## 实际改动
新增私有_run_lifecycle_with_stat_cache真实组合边界，_run_isaac主入口已调用。
在单worker bindings作用域包装捕获的原run_owned_call，不递归调用已替换的模块属性。
向原函数已有before_collect传入一次性source-bound缓存释放：
同步GPU -> clear失败traceback帧 -> 清NativeOwners -> 缓存释放 -> 原有GC -> 原生命周期STOP/App。
原成功路径独占调用删除，成功时只保留completed_sdk_step整数；异常时读取同live sim实际步数，不伪造end。
hook使用weak sim reader，临时强引用在finally清除；实际步须为整数2..冻结attempt.spec上限，成功end须同实际一致。
拒绝外部pre-GC callback覆盖，一次执行后不重试。继续使用原make_before_collect、Kit源码SHA和时钟/空缓存验证。
没有额外gc循环、重绑native字段、关闭碰撞、物理频率变化、os._exit、信号或强制清理。
封存dh116_fault_safe_lifecycle_v3.py及Tripod Reset实施来源SHA保持不变，无需篡改历史manifest。

## CPU行为证明
新测试调用真实run_lifecycle与run_owned_call，只将不可用的Kit缓存I/O替换成显式CPU缓存。
该缓存保留已完成资源constructor帧，复现owner资源字典清空后对象仍活着；hook清理后在STOP前相关weakrefs失活。
覆盖成功、acquire故障、cache故障和双故障；双故障保留原PPO错误与cache错误，关闭和回调各最多一次、bindings必恢复。
另外检查step0/越界、完成end不符、缺失sim、make_sim提前失败、外部hook覆盖拒绝。
新边界缺失时先得到3个失败；实现后初始3项通过。扩展首轮66通过1失败为测试错误假设make_sim故障不STOP；
原生命周期事实上仍STOP一次，已修正测试而不改封存逻辑，失败日志保留。
最终与旧生命周期/Provider共68通过（8.93s），加当前配方/启动/合同/Registry135通过（35.37s），两组不重叠共203通过、0失败。

## 复核与限制
独立只读Agent核查真实主入口、删除旧成功调用、冻结上限来源、callback时序和bindings恢复，规范/质量无重大问题。
审查Agent未运行测试；Antigravity仍无额度，未冒称其新回复。
make_sim/trace.watch在进入run_owned_call之前的异常不会执行这个hook；原生命周期仍如实失败并关闭，不泛称全部启动异常已修复。
本包没有运行Isaac，因此不能声称SIGSEGV根治、自然exit0、PPO更新或抓取成功。
v8独立绑定当前564文件闭包，指纹0edf9c3c8392846174c34bc0e4a24c919668dfbc5373cb3fd13589a2c733985d，尚未执行。
原8控制/66SDK/768内部solver子步/1更新、180/240s预算不扩大；v5测得的wall剩余问题仍须解决，不盲目再跑。
下一包优先核验已有control数据复制/解析热点与完整预算可行性，再取得实际native释放证据。
528保护文件、3份原Checkpoint核验原SHA；固定双RGB、24D/371D与30mm/5s/0.5mm标准保持。
无合格完整Teacher成功轨迹，R2/ForceProxy与正式成功BC/DAgger准入继续关闭。
