# DEXTRAH关闭对照与预览窗口所有权修复候选

基线cf7fac999ef82e3598f467cae2641f78411b8503。原模型、物理、动作、成功标准与SDK安装文件不变。

## 参考项目实际关闭方式

核对NVlabs/DEXTRAH main的ebc08edafe5f7c5ad73dcb25ffa15e8d0353df50：
[Teacher train.py 201–209](https://github.com/NVlabs/DEXTRAH/blob/ebc08edafe5f7c5ad73dcb25ffa15e8d0353df50/dextrah_lab/rl_games/train.py#L201-L209)
先env.close()，随后simulation_app.close()；play.py同样。
[Student run_distillation.py 201–205](https://github.com/NVlabs/DEXTRAH/blob/ebc08edafe5f7c5ad73dcb25ffa15e8d0353df50/dextrah_lab/distillation/run_distillation.py#L201-L205)
在main返回后调用simulation_app.close()，该入口未显式env.close()。
任务继承DirectRLEnv，无自定义close覆盖；官方IsaacLab v2.2.1 DirectRLEnv.close
先释放event manager/scene/viewport，再清SimulationContext，最后释放窗口。
参考入口没有本次AppWindow解释器退出问题的专门修补，也没有显式指定fast_shutdown=False。
其确切Isaac Sim runtime没有在依赖清单固定，不能据此断言它运行了强制退出；
也不能把相同app.close调用写成我们完整原生自然退出已经通过。
我们的私有预览坚持fast_shutdown=False、原生插件正常卸载、外层exit=0独立核验。

## 当前批次第一次实际原生：关闭改善但最终失败

用户批准最多两次；第一轮source cf7fac9，合同678b67a4f9b132bdf4f4b45b647a75b68f665d6ee79524336be92c06b2ac8909。
17对双RGB/34PNG，320×240，130SDK步、128测量步、1536测量solver子步、实际61次显式render、零PPO。
167.171625秒，166条完整physical JSONL，无阶段错误。
STOP/commit、manager与ownership保护均通过；before_close及after_app_close各16/16弱引用失效。
以实际JSON计数16项为准，不沿用旧阶段的17项口头计数。
app_close_returned.json确认标准app.close已返回，但外层255/子进程SIG11，故自然退出不通过。
GDB主线程_appwindow绑定→carb→Py_FinalizeEx，故障指令call *0x18(%rax)；
不是旧present线程或旧USD context故障，尚不能证明唯一C++指针根因。
原生日志SHA 71b6ff187ba220be7f4b703201b69fe7b6ace7e7600c37e1da6dd9074cfcb966。

## 唯一候选与失败测试

安装版HotkeyTrigger保存__app_window，destroy仅取消future和输入订阅，没有清窗口字段。
CPU直接编译安装版类destroy AST，确认destroy返回后窗口弱引用仍存活。
新增tabletop_preview_window_owner.py只包装私有App已经存在的一个trigger实例的destroy：
先执行原SDK destroy，再在插件仍加载时释放其窗口引用，不修改SDK class或安装文件。
只读匹配已启动extension registry与真实on_shutdown来源；未知布局拒绝。
12项SDK源码/二进制SHA锁v3，旧v1/v2保留。无新增物理步、render、额外gc或强制退出。

只读复审发现并修复两条异常路径：日志I/O不得打断SDK shutdown，错误以字符串收集，
app.close后外层校验再报；launcher部分初始化失败也先安装同一释放hook再正常close。
对应四条失败测试先RED，再GREEN；最终新增11项CPU测试通过。
正式回归27个测试文件307 passed/0 failed/0 skipped，日志SHA
42a7ad4508191a67845d5f1b1f2cc8941346d87270517ce0e5ef7cff0469c59e。
实际命令：
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -B -m pytest -q <27 files>
只读复审两项Important修复后无新增Critical/Important。此候选尚未原生验证，不能宣布解决SIG11。

## 恢复与剩余预算

私有批次/home/xk/.local/share/dh116-autonomous-3c80166/evidence/live-stereo-preview-20261116/timeline-wrapper-batch2-20261116T080011Z。
第一轮账本、PNG、GDB栈/映射/寄存器与退出码保留，不上传Git。
批准批次剩一次；仅测试通过的小型修订发布/同步/冻结新SHA后可用，不重复旧版本、不扩大预算。
保护528文件和两个历史Checkpoint；PR #2 Draft。Teacher/Student、抓取资格均保持false。
