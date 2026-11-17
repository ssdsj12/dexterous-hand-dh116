# PhysX 生命周期 GDB 全线程暂停取证

已消耗的 registry 与 explicit 配方在 `Py_FinalizeEx` 回调内请求全线程栈时均失败。旧脚本随后在仍运行的选中线程上执行 `continue`，无法给出进程退出状态。带 daemon 线程的 CPU 程序复现了相同的两条 GDB 错误。CPU fixture 不包含 Isaac、CUDA 或 PhysX。

新增独立 GDB 命令生成器。命令显式使用 all-stop。断点与信号回调只记录停止类型；在 `run` 或 `continue` 返回后验证选中线程和全部已知线程均暂停，再捕获栈、寄存器、指令及动态库映射。由 GDB 的退出变量输出实际 inferior 退出码或信号。未知停止或不完整捕获保留失败。旧脚本和已消耗配方指纹保持原样，新生成器尚未接入新原生配方。

Ubuntu GDB 12.1，现有 Python 3.10，屏蔽 GPU。三项带后台线程的真实 GDB 进程测试分别验证自然退出 0、自然退出 7、SIGSEGV，均重开原始取证文件确认栈、寄存器和映射。旧 3 个测试执行失败的日志 SHA256 `8bd9f2ff9669921fe5b6c04f4cba802fe5ed9c6ed5bfaf349d26c749f781f418`。首次通过日志 SHA256 `3a1c84cb64cf6f4891e1a0780dea324cc2244721498bcee57931c9037c229c9f`。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_physx_lifetime_debugger_stopped.py tests/test_dh116_physx_lifetime_probe.py \
  tests/test_dh116_physx_lifetime_validation.py tests/test_dh116_physx_lifetime_scheduler.py
~~~

新增 Isaac 步和 PPO 更新均为 0。此包修正观察器的 CPU 已复现错误。两个原生实例仍观察到 `build_native` frame 等保留资源，不能由本修正推出 Tensor View 释放或自然退出资格。后继原生配方应先准备独立构造作用域清理，再绑定新的捕获入口与审批。
