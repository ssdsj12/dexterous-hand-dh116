# 原生关闭复验与Timeline Python入口修正

基线66f18cb07e563f3319dd57cb04158d29fdfa1b87。原始资产、控制、物理、模型与成功标准不变。

## 本次实际原生

合同SHA 1ab1505542ee0b5fc90f212cfcbcc8d530ae506c8335460d1cacb0e3f570b35c；用户单次批准后执行一次，未重跑。
GPU0共享，1环境，320×240；17对RGB，130SDK步（128测量）、1536测量solver子步，零PPO。
新增关闭前durable计数实际记录61次显式render，而不是按预算推算。
151.440秒；外层255、子进程SIG11，无强制结束，预算余0。

## 原生验证到的改善与未通过项

before_close的17个资源弱引用中15项已失效：SimulationContext、左右Articulation与ArticulationView、
左右两种WrenchComposer、runtime、rope、episode、material reader与camera。
仅left_asset_sim_view/right_asset_sim_view存活；两个弱引用指向同一SimulationView，
各自唯一可见referrer为同一个含 `_physics_sim_view` 的dict。manager释放仍未通过。
这支持上包自身完成帧alias清理的实际效果，不代表全部native view已经释放。

STOP实际报错：`AttributeError: Timeline object has no attribute get_timeline`。
此前只核对底层ITimeline声明，漏了实际SDK Python包装层，是本次调用修复中的错误。
SDK `omni.timeline.__init__.get_timeline_interface()` 已调用底层ITimeline.get_timeline并返回Timeline；
脚本再次.get_timeline导致STOP与commit根本未执行。不能将其写成STOP回调已经通过。

直播disable/shutdown均通过调用返回检查，采集scope返回且camera资源已失效；
本次完整GDB栈不再是旧present thread：故障是主线程 `libomni.usd.so + 0xd9623`，
经 `omni::usd::UsdManager::destroyContext` 发生在扩展关闭。
此前即时更新中把本次归为旧直播故障不准确，以本报告和真实栈为准。
寄存器与指令已记录，故障指令为 `mov 0x8(%rax),%rax`；没有足够信息证明其具体对象或唯一上游C++原因。

## 实际最小修正

脚本直接使用 `omni.timeline.get_timeline_interface()` 作为commit_preview_stop参数。
新增回归编译执行安装版Python包装函数和实际脚本stop函数：原版1 failed，修正后通过，
不再用猜测的ITimeline对象替代真实Python入口。
独立API lock v2纳入该Python入口，7项实际文件指纹；旧v1保留，安装文件不改。
Launcher版本v4；停止、freeze、时钟、manager/ownership保护与预算保持不变。

全CPU回归296 passed，0 failed，0 skipped，14.05秒。26个文件由
`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q <files>` 执行。
日志SHA 0df96cba6e1118d11ba9d07f9db6ce71d04effd4034558eb6ce5eb228c6f0a4d。
只读复审已核对真实Python入口与7项源码/二进制指纹，未发现新增Critical/Important。
本修正版尚未原生执行，未把CPU回放写成自然退出通过。

## 证据与恢复点

私有目录：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/live-stereo-preview-20261116/owned-shutdown-native-20261116T074718Z/attempt-1`。
原生日志SHA b119a45bab158d2334d17a03e8e6bf3d59ab7f42053be2e14b93df4e553617e5。
保留完整JSONL、61次render计数、PNG与帧身份、GDB栈/映射/寄存器/指令、guard错误及外层退出码。
结果摘要：docs/contracts/preview_owned_shutdown_native_result.json。
源码保护528项和两个历史Checkpoint继续不改；原件、模型与日志不上传Git，PR #2继续Draft。

## 剩余

需要新预算验证修正后的实际STOP/commit和manager释放，然后验证USD context销毁及自然exit0。
目前没有证据把C++段错误归因于单一指针或宣称已解决。原生资格、Teacher/Student准入仍为false。
同版本失败不重跑，不扩旧预算；新修复发布同步、冻结后才申请下一次有界复验。
