# DEXTRAH关闭顺序对照后最后一次复验：窗口已释放，解释器退出仍失败

## 实际执行

代码e31a694879e16726f6caf6ea5ed61025482e50c9（修复d91a86086ebab0030893ae70e6c10352bb43c83c）。
本批第二次且最后一次；冻结合同98cf39c201a4a99f0e5870e0e4f6ff171dfd1ba60e056c5f48b6d8614ca166d9。
共享GPU0、单实例、320×240、零PPO；130SDK步（2初始化+128测量）、1536测量solver子步、实际61render。
17对RGB/34张PNG经Pillow逐文件verify，17条帧身份同一标定SHA，166条physical JSONL无残行。
170.417319秒；无强制结束，无自动重试。完整原件/GDB日志留私有目录，不上传Git。

## 真正通过的边界

STOP/commit与manager保护无错误；关闭前后各16/16资源弱引用均失效。
已启动的唯一HotkeyTrigger hook真实执行，原SDK destroy返回、窗口字段确已清空、无记录错误。
正常SimulationApp.close返回，整个stage日志无error。没有据此授予进程退出或抓取资格。

## 仍然失败

外层255，子进程SIGSEGV，非自然exit0。
本次GDB主线程_extensions.cpython-310-x86_64-linux-gnu.so +0x1e445 →carb→Py_FinalizeEx，
故障call *0x18(%rax)。第一轮是_appwindow+0x15e55；不是同一故障模块。
窗口引用修正已实际生效，但不足以解决完整解释器退出；不能把它写成唯一C++根因。
也不把模块变化解释成全部SDK生命周期已经正确。
日志SHA 5d28688a2977297cf523198e83b4fff7a6a5fbf2e6f328c2f80c0086620a5ae4，854705字节，单次完整GDB dump。
原始left/right帧ID仍不同，底层raw simulationTime不冒充SDK物理钟；
17对输出不代表像素时刻/重投影或Stereo精度已经资格化。

## 与DEXTRAH的结论

[参考Teacher入口](https://github.com/NVlabs/DEXTRAH/blob/ebc08edafe5f7c5ad73dcb25ffa15e8d0353df50/dextrah_lab/rl_games/train.py#L201-L209)
为env.close后simulation_app.close，已按资源/窗口先释放再正常App.close核对。
参考项目的调用方式不能证明当前SDK解释器终结阶段可用。其入口未显式指定fast_shutdown=False，
我们继续保留该严格设置，不用快速退出、跳过插件卸载或信号伪造通过。
详细源码对照与CPU RED/GREEN记录见上一份2026-11-16-preview-dextrah-shutdown-window-owner.md。

## 回归与发布

代码307项CPU回归通过，原生结果单独标记blocked。
528保护文件、两个历史Checkpoint不改，SDK安装不改；旧Teacher/Student与物理合同不改。
本包复跑同27文件CPU回归，并由发布调度器再次校验保护与Checkpoint。
README、CHANGELOG、Registry、readiness及批次摘要随结果发布；经Windows推送并核对SHA，PR #2继续Draft。

## 预算与下一步

本批两次已用完：合计260SDK步、3072测量solver子步、122显式render、337.588944秒、零PPO。
保留恢复点与所有失败，不发起第三次。
下一离线工作应先盘点解释器终结前剩余扩展/Carb接口所有权与SDK卸载顺序，
对照最小headless无直播入口，避免逐对象猜测清理。确认新实验方案和预算后才可再原生执行。
当前自然退出、Restart、Stereo同步/精度、Teacher/Student训练准入均未通过。
