# 原生预览离线诊断与清理修复

用户批准范围仅为离线代码、CPU验证和发布。本包未启动Isaac、未采集新RGB、未推进原生物理/PPO。基线077ad4122ef564cd870274e5cc2820a35c8bf415，上一原生结果仍为强制结束、测量0行、camera0、native blocked。

## 实际修改

1. R06生命周期新增默认关闭的stage_observer，不重写既有步骤。preview专用StageJournal逐阶段记录start/returned/error/skipped，JSONL逐条flush+fsync，原始acquire及清理异常在app.close之前持久化。日志写入失败仍保留原错误并继续清理，不形成虚假通过。
2. RenderBudget.close恢复原render方法并丢弃sim、原bound method和override引用；关闭后不能render/install。RGB bridge与owned reader关闭时清除sim/reader/snapshot/annotator别名，即使清理失败也保留失败状态而不重试native销毁。这仅证明Python引用和调用行为，不等同native views释放。
3. 预览专用ManagedPreviewApp明确拥有launcher；关闭前退订两个timeline订阅，移除仍属于本launcher的信号处理引用，尝试一次正常close，失败字符串保留，最后丢弃launcher/app引用。独立进程内将原生SIGSEGV/SIGABRT/SIGBUS设为默认致命处理，避免Python handler再次调用app.close；不发送人工致命信号，不将native错误改写为exit0。
4. 新预览专用GDB wrapper保留真实inferior exit/signal。只首次采集线程栈/动态库/映射等，payload上限8MiB，截断显式标记；重复致命故障时暂停debugger和已停inferior，等待处置，不无限dump、不自动kill、不伪造自然退出。旧共享GDB模板未改。未来运行必须显式使用该wrapper，不能继续沿用之前旧wrapper。
5. preview实际入口已使用managed launcher和durable stage journal；adapter/debugger policy进入冻结合同。旧指纹失效，需重新冻结和批准。

## 本机关闭来源审计

精确安装源SHA列于preview_cleanup_source_audit_20261116.json。IsaacLab确实为SIGSEGV/SIGABRT安装调用app.close的Python handler；SimulationApp.close会先关闭stage，在资源加载时还会update，最后触发Kit extension shutdown；WebRTC扩展on_shutdown调用native livestream shutdown。第三次原生首栈位于livestream库，但这些源码事实不能证明关闭崩溃唯一根因。未升级/修改SDK安装，也未凭猜测禁用streaming或跳过正常app.close。

## CPU验证

151 passed，命令使用Ubuntu DH116解释器和scripts/python.sh、CUDA_VISIBLE_DEVICES为空。包含原R06故障注入回归、关闭前错误落盘、fsync、弱引用释放、回调/信号失败时继续close、旧预览和RGB回归。GDB集成测试只启动纯CPU Python子进程：安装模拟递归handler后切换默认fatal，再触发真实SIGABRT，核验inferior signal6及非零退出、单份故障栈；禁用core dump。该CPU信号测试不是Isaac/native物理验证。

## 未解决与下一步

第三次原生最初acquire错误仍未知，无法回补已丢失的内存文本。SimulationContext是否完全释放、WebRTC native shutdown、相机实际首帧/同步和自然退出仍待验证；180秒仍是软截止。修复后的原生预算未在本包执行。应先申请一次有界诊断，使用新debugger并核验preview_stages.jsonl中原始错误，失败保留证据、不重复同版本。
