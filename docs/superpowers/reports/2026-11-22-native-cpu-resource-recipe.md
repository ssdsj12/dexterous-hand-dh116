# Teacher有限CPU资源配方接通
基线c9c15aac75440347c7cd9916e680d015f588af73。解决已有调度器支持与Teacher严格schema之间的实际断层，不升级或替换环境。

## 修改
validate_teacher_recipe_v2现在只接受旧精确字段集合，或旧集合加可选environment。
environment必须精确只有DH116_NUM_THREADS，值为规范ASCII字符串1..64；0/65/非字符串/01/空/None/库路径等拒绝。
原预算、来源闭包、validator、动作/观测、奖励检查不变。原deepcopy返回保留，旧v1 fingerprint公式及无environment配方保持。
同一来源和输入只改线程1->2，现有scheduler确实生成不同code_config SHA；配置字段绑定原冻结身份，不是启动时隐藏注入。

## CPU和实机启动器
最初合法上下界/调度器fingerprint3个用例失败，6个非法环境用例拒绝。实现后中间142通过。
加01负例及真实已有scripts/python.sh集成后，worker、actual-launch、冻结v10、Registry联合144通过0失败，34.33s；不与中间计数累加。
冻结v10测试先因新package缺失失败，补真实配方后通过；沒有仅靠schema flag宣称运行。
实际CPU子进程故意继承OMP/OpenBLAS/MKL=17，DH116_NUM_THREADS=1，经原脚本读回Torch数值线程1、OMP/OpenBLAS/MKL/NUMBA均1、CUDA mask空。
Torch inter-op仍64，未修改；不能说所有系统线程都变为1。Python3.10.12、Torch2.7.0+cu128、NumPy1.26.4原样。
线程比较为历史数据CPU复放：几何8包1线程中位0.819s、64线程1.040s；所有输出相同。失败Checkpoint仅用于CPU推理性能探测，mean/value线程比较最大差0，不作恢复或专家。
这些测量不证明新原生完整wall可通过，不改全局CPU默认或正式训练并行配置。

## 冻结v10与执行边界
v10指纹01d238dc651cc69a1cc0f32ff9b4bdb76a23578398d4e7b4a904f93db4288fbc，565文件闭包。
仍GPU3、1实例/1环境、8控制、2+64SDK、64Rope/768内部子步、1更新、180s内层/240s外层。
相对v9组合了已测试的日志、精化粗算去重与线程资源设置；后续原生是组合修复诊断，不独立归因或长训练。
v9已消耗且失败，不重跑，不恢复其检查点。本包0次Isaac、0PPO；执行前必须重新核对发布SHA、实际资源与单实例锁。
528保护文件、3原Checkpoint、原Reset实现来源、双RGB和物理成功门槛均不改；R2/ForceProxy和正式BC/DAgger门控关闭。

独立只读规范与最终质量复核通过，01负例建议已落实。Antigravity仍无额度，没有冒称其新回复。
