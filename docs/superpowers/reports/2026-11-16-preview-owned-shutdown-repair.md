# 预览关闭故障：源码定位、资源所有权与修复

基线 c7716c09e9cd86c8ceb71db0a62bd7f41444a43b。仅修改独立预览路径；没有新Isaac运行、PPO更新或环境升级。

## 真实故障边界

上一批第二轮已经取得17对320×240 RGB、128个测量物理步、1536个测量solver子步，采集正常返回。
此后manager/ownership释放guard仍为false/unknown，在app.close发生SIGSEGV，外层255。
两次已消耗运行的GDB首次PC分别经真实映射归一化，均指向
`libomni.kit.livestream.plugin.so` 文件偏移 `0x11f26`；线程都是present thread。
这定位了稳定的直播关闭故障边界。当前二进制缺少源码调试信息，旧dump没有寄存器，
因此不能把它直接断言为已证明的null dereference或use-after-free，也不能认定Tensor View是唯一C++根因。

原件：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/live-stereo-preview-20261116/tiled-usd-batch2-20261116T063806Z`。
没有重跑旧配方或消费第三次原生预算。

## 已确认的软件问题

### 1. 完成帧仍强持有物理对象

真实before_close弱引用记录显示simulation被Tiled构造、from_native、create_preview_camera完成帧持有，
episode被run_preview/run_owned_scene完成帧持有，另有闭包与_sim等引用。
新测试只保留真实Python完成frame，不读取f_locals，使用弱引用检查；修正测试自身f_back引用后，
在原基线再次得到6 failed，新finally清理后6 passed。额外两项RenderBudget构造路径也先失败再修复。
即使外部诊断器或SDK持有完成frame，本路径也应撤销自己借入的sim/episode/reader/回调引用。
这里只证明并修复自身强引用；外部为什么保留完成frame仍未被认定为唯一上游原因。

修复：run_preview、run_owned_scene、create_preview_camera、TiledStereoCapture.from_native、
TiledNativeStereoReader.__init__、RenderBudget.__init__以及实际脚本make_sim/present/acquire的自身alias清理。
不清理他人frame，不手改native view，不增加gc.collect，不修改物理。
identity改从acquire层holder取得runtime/当前exchange，避免present finally撤销rod局部后破坏跨帧采集。

### 2. STOP回调处理与释放检查顺序

已安装SDK SimulationManager._initialize将STOP清理注册为pop回调；_on_stop才会invalidate并撤销manager view。
原脚本只调用timeline.stop；v3生命周期在允许app.update前又要求manager view已为空。
当前原生没有记录STOP回调是否已经执行，不能伪造此项证据；源码与CPU重放确认旧路径缺少显式派发边界，
会在仍有待处理STOP时走入guard失败且跳过update的路径。

修复：独立预览stop调用真实默认Timeline的stop()+commit()，后者的安装API说明为在主线程
提交pending changes并调用callbacks。调用前后检查physics freeze与弱引用SDK时钟。
不调用App.update/render/physics step，不直接写SimulationManager._physics_sim_view。
原有manager/ownership guard不降低、不绕过，实际native readback仍是验收条件。
CPU测试编译执行实际SDK的_on_stop函数，证明待处理回调执行后invalidate发生；不是原生通过证明。

### 3. 直播消费者与stage/纹理关闭顺序

安装版SimulationApp.close先close_stage，再shutdown App；原预览直接调用它，未显式停止直播消费者。
与present thread重复故障边界相符，但尚需新原生验证因果。
安装版ILivestream明确暴露enable_livestream(bool)->bool与shutdown()->bool。

修复：自己的预览先调用enable_livestream(False)、shutdown，再销毁camera/render product并关闭App。
ManagedPreviewApp.stop_presentation只尝试一次，失败sticky，false不是通过；partial startup也先停止已创建App的直播。
不强制卸载SDK插件、不修改安装文件、不杀进程。源码/API/实际.so共6项SHA进入独立lock JSON，启动与关闭均核验。
这些API返回并不自动授予native thread quiescence资格。

### 4. 异常与证据保留

复审确认journal在try外创建，构造失败会跳过清理；现已用部分资源None保护并把初始化纳入try。
新增真实acquire AST+Journal OSError回归验证sim/owners/trace不因完成frame被保留。
复审也发现采集和cleanup同时失败时，cleanup异常会隐藏原错误；现在原故障先flush/fsync到
preview_acquire_error.json，然后与cleanup错误以文本合并。双失败测试在停止直播前检查原错误已落盘。
正常采集的实际render计数也在关闭前fsync到preview_acquire_result.json，避免再次只有预算上限没有实测值。

## GDB补强

首次dump仍限8MiB，重复故障仍暂停、不自动强杀、不重试、不转换exit0。
新增info registers和16条当前指令，在大线程dump前采集；POLICY_VERSION为v2。
CPU子进程SIGABRT回归仍得到真实信号码6，外层非0，未重入Python关闭handler。

## 测试、保护与实际状态

完整26个测试文件：295 passed，0 failed，0 skipped，16.76秒。
命令为上一23文件回归加tests/test_preview_retained_frames.py、tests/test_preview_shutdown_order.py、tests/test_preview_shutdown_wiring.py。
`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -B -m pytest -q <26 files>`。
日志SHA f46e854a3f86c5efbef55cb49416c9ce54fdfacb847e702d7d6c41206d3c2765。
已检查528项保护文件与两个历史Checkpoint未改；没有触碰Teacher/Student/MPC、动作/观测、材料、关节范围与30mm/5s/0.5mm标准。
源码和CPU问题已修，原生SIG11、实际STOP回调派发、资源释放、Reset/重启仍未验证。

## 下一原生复验

旧两次预算剩余0。本包不运行Isaac，先Windows发布并同步原目录，再冻结新合同。
建议新的一次GPU0共享、单实例、320×240零PPO复验，上限仍为130SDK步/1536测量solver子步/162显式render/180秒软截止。
逐项检查17对RGB、采集计数、直播disable/shutdown、manager/ownership readback、app.close返回、外层自然exit0。
失败保存首栈/映射/寄存器/指令和所有durable错误，不自动重跑或强制结束。
仅在获得对应新预算授权后执行；不自动恢复旧分辨率批次，不启动训练。
