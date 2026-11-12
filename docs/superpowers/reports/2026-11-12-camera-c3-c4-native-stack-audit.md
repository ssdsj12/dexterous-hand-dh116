# C3/C4 相机生命周期和原生故障栈核查

核查基线 `d7f6bb6147ed1a59f5e25787d9db5a4f0d0f733a`。原生批次 `continuation-c3-c4-once` 在 Ubuntu GPU 3 串行执行，每格各一次。两格均只有一次显式 App update，物理步、训练更新、采集帧和完整评估回合均为 0。本报告复核既有证据，没有启动新实例。

## 实际启动、窗口和退出

| 格 | 实际 experience | 默认 AppWindow | 默认 viewport | 启用扩展 / 关闭前过滤库路径 | 实际退出 | 墙钟 |
| --- | --- | --- | --- | ---: | --- | ---: |
| C1 历史对照 | rendering headless | 未读回 | 存在，无 render product | 145 / 177 | GDB 255，inferior SIGSEGV 11 | 54.370 s |
| C3 | no-camera headless | 未读回，AppWindow 扩展未启用 | 未读回 | 68 / 104 | GDB 0，inferior 0，无信号 | 26.843 s |
| C4 | rendering headless | 实测不存在，扩展仍启用 | 存在，无 render product | 145 / 177 | GDB 255，inferior SIGSEGV 11 | 54.807 s |

三格均为 headless、`fastShutdown=false`、`create_new_stage=false`、无 stage、无 USD render product，且观测到原生共享锁。C1/C4 使用相同 rendering experience SHA-256 `978bd89e90f2ca00b32805a70dd0a15c5b8b48edeb42cd042f8eb679d33fa4ea`。C3 使用 no-camera experience SHA-256 `b4a60880219827b4843f3ec3d6f9cbad8028c1c9df588a89c1796a60b8030a92`。三格 SDK 为 Isaac Sim 4.5.0，实际 Kit build 为 106.5.0。

C4 在显式 update 前和 App close 前都读回 `autocreateAppWindow=false`、`default_appwindow_present=false`、`appwindow_extension_enabled=true`。同一读回路径通过已加载模块查询默认窗口，未额外导入 AppWindow。C1 未记录该 getter 的值。C1/C4 的 145 个启用扩展以及 177 条过滤库路径集合逐项相同。C3 的两组列表都是其真子集，分别少 77 个扩展和 73 条库路径。C3 的启用列表不含 AppWindow、viewport、RTX renderer 和 Replicator Core。这些差异来自切换 experience，同时涉及多项组件。

三格的内层 App close 标记均已写入。C3 的 GDB 还独立记录 inferior 自然退出 0，资格范围为 `camera_c3_isolated_lifecycle`。C3 没有崩溃栈或信号时全量映射。它保留自然退出的 GDB 日志与 `before_app_close.json` 中 104 条从 `/proc/self/maps` 过滤的运行库路径。该过滤器按 basename 取相关库，无法据此判断未列出的任意模块是否曾映射。

## C4 的同次原生故障

C4 首个停点的 PC 为 `0x7ffff2a1e445`，位于 `omni/ext/_extensions.cpython-310-x86_64-linux-gnu.so`。信号时执行映射为 `0x7ffff2a00000–0x7ffff2a6c000`，映射文件偏移为 0。按 PC 减映射起点再加映射文件偏移，故障文件偏移为 **`0x1e445`**。独立保存的同进程 live proc maps 中，该映射起止与文件偏移一致。对应安装文件 build ID 为 `49e7f31a15a35129b06c776e0e79404c7944ca1a`，SHA-256 为 `695ea005b3a39a433c13a4f589ed137a51b604a9563b6bfc031b6216d9a075a6`。

| 首次故障帧 | C1 | C4 |
| --- | --- | --- |
| 0 | `_appwindow+0x15e55` | `_extensions+0x1e445` |
| 1 | `_carb+0x4213d` | `_carb+0x4213d` |
| 2 | `_carb+0x425de` | `_carb+0x425de` |
| Python 退出路径 | `Py_FinalizeEx` 在第 8 帧 | `Py_FinalizeEx` 在第 10 帧 |

第二次停点经 `pthread_kill` 和 `raise` 交付同一 SIGSEGV，信号处理器下方仍可见上述 C4 原始故障栈。GDB 记录实际退出 255 和 inferior 信号 11。全线程回溯、动态库列表和信号时映射均在该次日志中，GDB 四项采集命令无错误。安装 ELF 的动态符号范围没有覆盖 C4 三个故障地址。具体回调、所有者和释放错误仍未定位。

C4 证明当前复现可在默认窗口不存在时发生，并把首故障从 C1 的 AppWindow 绑定移到 Extension 绑定。两者共享 Carb 和 Python 结束路径。C3 的结果支持继续按 rendering experience 的实际扩展依赖做只读定位。C4 依然阻塞 RGB、深度与 Native ID 的正式双目采集，无相机物理和 Teacher 准备可按各自前置证据继续。本批次已用完 C3/C4 各一次额度。

## 冻结身份与私有证据

| 格 | 实际启动 HEAD | command spec SHA-256 | code/config SHA-256 |
| --- | --- | --- | --- |
| C3 | `dea157f2531c31569f880f19e6e5d93c604a7b9c` | `47fd6d1b18c2f2cb40ed9a6082d564f5e4cd0982d9d8113398fc0588586ac4e3` | `3218f2fa287f9b65d01cddc79b8e6ceeda2140c18aa7af7f3d0491c69d8ab5c2` |
| C4 | `27b881ef0b48d6845c67b2e7e49d12b665d34728` | `ef59621be11834327c73be02da5391b9e117c46a8851f863cab1b0af5ac55d49` | `a5d3eccdd4d0da813208b9ddafd7462ff11fe642e63a0c2ac7540d60009fcf0e` |

实际命令为 `bash scripts/python.sh scripts/probe_dh116_simulation_app_lifecycle.py --case C3 --output <C3 独立目录> --execute-native`，C4 使用对应 case 与独立目录。回执命令、批准 JSON 和外层退出逐项复核一致。原始日志及映射保留在 Ubuntu 私有证据目录。

| 格 | attempt | 关闭前读回 SHA-256 | 原生日志 SHA-256 | 调度回执 SHA-256 |
| --- | --- | --- | --- | --- |
| C3 | `attempt-b5nskwks` | `c0977b007c6b06c38b051928ce8956db1e590fd259cff78d33e0f54a7c107951` | `e43c4812d796f4c7bbcf4f8ad623edcede0b863176137c22b0a364a2d7a88730` | `c0c48203d13e7de5416fca42d9a8d40096c588d8ca9eef5aa858f733c4a24f42` |
| C4 | `attempt-iqt7vw5q` | `2004b2e0294bad971e89f95c8f251d6640bab44e22daafbea8d7395e52e3cf7d` | `53b96994d16e772be4d93996a0229607388f9ee3f0fc109f906463bf9ae25ea0` | `7ef892c4dc65ea85168e167b023ec5d07c496f7aed12f99911d45238de5c05d6` |

C4 live proc maps SHA-256 为 `4a1639447a7a3459b310a48bc1ca84a5c839f09c851bc5ba3c6110328bd0b0cd`。私有派生 `audit.json` 包含精简首次故障栈、PC 与映射换算、实际扩展差分及原始产物摘要，其 SHA-256 为 `f953921955af541d3849fc0bfd863f0123f4bfdb093d0f3538e823dad8053576`。

## CPU 回归

```sh
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_simulation_app_lifecycle.py tests/test_dh116_render_capture_boundary.py
```

CPU 30 项通过，用时 2.22 秒。528 个保护文件与两份历史 Checkpoint 通过哈希核对。本提交只新增报告。
