# C0–C2 原生相机生命周期对照

2026-11-12，`eb31c79-camera-once` 在 Ubuntu GPU3 串行运行 C0、C1、C2，各一次。每次只启动一个 Isaac 实例、显式调用一次应用 update，物理步和训练更新均为 0。每次均观察到共享实例锁。**三次均在 `app.close()` 返回后进入 Python finalization，并由 AppWindow 原生段错误终止。** 没有自然退出 0，生命周期资格继续阻塞。

## 冻结身份

冻结清单 HEAD 为 `0764fb95e1a2cb99c3a851385d0cd26c75139ffe`。调度器在每格后提交结果元数据，导致后续实际 HEAD 前进。已逐一用 `git diff --exit-code` 核对各格实际 HEAD 与冻结 HEAD 的探针、调度器、启动器、测试及三份运行配方，字节相同。

| 对照 | 实际运行 HEAD | command spec SHA256 | code/config SHA256 |
| --- | --- | --- | --- |
| C0 | `7a4a0a938e5e9489686bc634d009829a8c018e5b` | `ca1d3ee9257d05e219537646831a4dc9d595a22dcf12c7a6f3e606fa060f6d30` | `1387059c1e2d6a554047665f0743073238747b43cedb2918d149b7ac555cff5a` |
| C1 | `44d6ef56cd3ab9ca7a6fe261d65b8922cd9c1dcf` | `b9e2ae59cc1c96eeb25ecd999228083d191dd2dd84262bef9f5470d374da0b0a` | `d23fe888c6217dd22848d35f871d7c1cb32777cb0ae195e3942a8aa9b216568a` |
| C2 | `00c21d94705eeafc937599f469af103ed7316729` | `fc32fcb3bf668a812ed113b4c58612993823f28344c16b7744edda49fc92dd95` | `5719e930418f6f1c163061ed45c6472534dff16c5eb8f83a5b0a933058d354c3` |

共同探针 `scripts/probe_dh116_simulation_app_lifecycle.py` SHA256 `794cba172ffabf6b77d1372212bb699a2551757df005bbc3231f122b6dae146f`，调度器 SHA256 `9f4d8e9a2c6a58dad5334fdd00c15202edb981868f052785221a43f7d66a02ae`，启动器 SHA256 `3aadbf9026b9ccc9008a7b72eb5eb2cbacd2087fa5153e6717d32b6f991bf27e`。三次使用同一 GDB 二进制 SHA256 `a6222ff0f450ca040e3479c04fdb7d0f1e0cabb099688270e720b7115d933c3d`。

执行入口为 `bash scripts/python.sh scripts/probe_dh116_simulation_app_lifecycle.py --case C0 --output <独立目录> --execute-native`，C1、C2 对应切换 case；均由有界调度器在冻结配方及一次性预算下运行。

## 实际运行读回

| 对照 | 实际 experience | SDK create_new_stage / stage | 默认 viewport / USD RenderProduct | 启用扩展数 / 关前库映射数 | close 返回 | GDB 外层 / Python inferior | 耗时 |
| --- | --- | --- | --- | ---: | --- | --- | ---: |
| C0 | `isaacsim.exp.base.python.kit` | true / 存在 | 存在 / 1 | 300 / 232 | 是 | 255 / SIGSEGV 11 | 98.374 s |
| C1 | `isaaclab.python.headless.rendering.kit` | false / 无 | 存在 / 0 | 145 / 177 | 是 | 255 / SIGSEGV 11 | 54.370 s |
| C2 | `isaaclab.python.headless.rendering.kit` | true / 存在 | 存在 / 1 | 145 / 181 | 是 | 255 / SIGSEGV 11 | 62.683 s |

C0 SDK experience SHA256 为 `92e92727259dde994ab2e0dab143ac0a4b1157e3bd7694fc53198fb87773b766`。C1、C2 rendering experience SHA256 为 `978bd89e90f2ca00b32805a70dd0a15c5b8b48edeb42cd042f8eb679d33fa4ea`。C0、C2 的默认 product 路径均为 `/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0`。C1 的 viewport 对象存在，product 路径为 null。C1 与 C2 的 145 个启用扩展完全一致，C2 在关前多映射了 4 个 NVIDIA NGX 共享库。三格均启用 `omni.appwindow-1.1.9`、`omni.kit.viewport.window-107.0.8`、`omni.replicator.core-1.11.35`。C0 还启用 `isaacsim.core.nodes-2.1.4`。三格均为 headless，`fastShutdown=false`。

解释器为 `/mnt/mydisk/xk/envs/dh116/bin/python`，Python 3.10.12，Isaac Sim 与 Isaac Sim Kernel 包版本均为 4.5.0.0。SDK 版本文件为 `4.5.0-rc.36+release.19112.f59b3005.gl`，实际 Kit build 为 `106.5.0+release.162521.d02c707b.gl`，Kit kernel 为 `180.6+release.7436.c5558fdf.gl`，Kit 包 commit 为 `d02c707b20b56088d49fa5d47df9b4a2d2b9dc86`。Kit app 插件 SHA256 为 `cbadd48220baa507190414ce09300513c3de046e20afc573fc321078cbcd80b5`。

三格 `CUDA_VISIBLE_DEVICES=3`，`DISPLAY` 与 `WAYLAND_DISPLAY` 均未设置。读回的 `LD_LIBRARY_PATH` 为 DH116 Python 环境的 `lib`、已安装 `omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310/bin` 与 `isaacsim/kit`，顺序相同。三次实际映射均包含 SDK 的 `libomni.appwindow.plugin.so`、`_appwindow.cpython-310-x86_64-linux-gnu.so`、`_carb.cpython-310-x86_64-linux-gnu.so`、`libcarb.so`、`libomni.kit.app.plugin.so`，以及 `/usr/lib/x86_64-linux-gnu/libcuda.so.580.178.04` 和当前 Python 环境内 `nvidia/cuda_runtime/lib/libcudart.so.12`。应用关闭前的路径来自 `/proc/self/maps`，信号时映射来自同一次 GDB。

## 同次首次故障栈

三格首个 SIGSEGV 均由 Python 主线程在 `Py_FinalizeEx` 内进入相同的二进制位置。以信号时映射起始地址和映射文件偏移换算。

- 第 0 帧为 Isaac Sim SDK 的 `_appwindow.cpython-310-x86_64-linux-gnu.so`，三格文件偏移均为 `0x15e55`，安装文件 build ID `63c333ee1ae4878fe05ff9c6d583aa4fd940c976`。
- 第 1、2 帧为 Kit `_carb.cpython-310-x86_64-linux-gnu.so`，三格文件偏移均为 `0x4213d`、`0x425de`。
- C0、C2 的第 7、8 帧经过系统 `_asyncio`，文件偏移 `0xacea`、`0xad7d`，第 12 帧为 `Py_FinalizeEx`。C1 的第 8 帧为 `Py_FinalizeEx`，首栈未出现对应 `_asyncio` 帧。
- 三格第二次 SIGSEGV 停点在 `pthread_kill` / `raise`，信号处理器之下仍是同一首次故障栈。GDB 继续交付真实信号后，Python inferior 状态为 `exit_code=null, signal=11`，GDB 外层实际返回 255。两次停点均留下当前线程及全线程回溯、`info sharedlibrary`、`info proc mappings`，四条 GDB 命令均无采集错误。

| 对照 | 私有 attempt | 关闭前读回 SHA256 | 原生日志 SHA256 | 私有回执 SHA256 |
| --- | --- | --- | --- | --- |
| C0 | `attempt-c5wry8u0` | `5a51818cf5c4c3c226b8e58afce2d81a9d2440bedc629f2b2176d9116dd2e965` | `2a7054541a74cf77ba6875b294a85b16a0e5795c029aed77731124fdf12b9737` | `344d998de7ee9310106f5ff814e964b47f6a160a2cecc76fcd078f0d15f8d9e6` |
| C1 | `attempt-br72v91b` | `5114d891e41ed754dd2e9b07f197fcaeb2681f8f094049b0dd866a7175eb551a` | `c8d6f1f1c5c192c52f56dae05578107fe39a416610ff7a6654f86b378ecc867f` | `3f62c161845530fb672e5f6f62ec0171005d2b98d97bce456a656641932d310e` |
| C2 | `attempt-th_0qdti` | `26624cbd05f979cd8209a2903f1343cc49c0dee50bbd7e8b9485c7668b2e18c5` | `6298601bce35401a7048e4ce3d61ad62c63b59304fd4331fb518d0e9fb54a078` | `eba3f6f7685b8e1039371c8c527a00f0c1ce5616676af881328863ab32e95f72` |

C0 证明在当前安装和启动环境中，SDK 默认 experience 也能触发该退出故障。C1/C2 只改变默认 stage 设置，确认在无 stage 和无 USD render product 时也能触发同一故障，恢复默认 stage 仍会触发。共同故障范围定位到 AppWindow/Carb 在 Python finalization 的调用路径。现有栈缺少该原生帧的符号与对象归属，具体扩展回调、引用所有权和 GPU 可见性影响仍待隔离。C0、C2 的额外 `_asyncio` 帧也无法解释 C1。

C3 未执行。V1–V4、RGB、深度、Native ID、任务双目、柔性相机和后续采集均无本批资格。下一项生命周期对照可从已有 C3 单因素配方着手，待纳入对应原生预算；本次保留全部失败证据，不重复运行 C0–C2。

## CPU 与证据复核

`tests/test_dh116_simulation_app_lifecycle.py` 与 `tests/test_dh116_render_capture_boundary.py` 合计 19 项 CPU 回归通过。用三次私有回执及实际日志重新核对命令指纹、日志 SHA、全局锁、关闭回执、真实信号和首栈映射。528 个受保护文件与 2 个历史 Checkpoint SHA 验证通过。本提交仅新增此报告。
