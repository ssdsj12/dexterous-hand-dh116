# F6 渲染退出的 AppWindow 原生回溯

在现有 Ubuntu Isaac Sim 4.5 上，保持入口 `e9f5af25668510f5a55af3ef1d739d5b8902824d` 和 wrapper SHA256 `03b58e47802d81891effeb0d28e894f3edd8da3ec46722bf7ddf858ae86fd083` 不变，仅增加 GDB 观察。运行仍是 drop variant 0、frame 0、一个相机。渲染采集完成且 App.close 返回，首次 SIGSEGV 出现在解释器回收时。

| 帧 | 原生回溯 |
|---|---|
| 0 | `omni.appwindow-1.1.9/.../_appwindow.cpython-310-x86_64-linux-gnu.so` |
| 1–2 | `omni/kernel/py/carb/_carb.cpython-310-x86_64-linux-gnu.so` |
| 8 | `Py_FinalizeEx` |

原生符号已裁剪。回溯将范围缩小到 AppWindow/Carb Python 对象析构路径，尚未识别具体 Python 持有者。三次无调试器运行的外层状态仍为 139。

GDB 在取得首个回溯后执行了 continue，信号进入 CPython faulthandler。faulthandler 再抛出的 SIGSEGV 被原调试命令第二次截获，GDB 外层状态 255 属于诊断。后续调试配方在取得首次回溯后先切换为 `handle SIGSEGV nostop print pass` 和 `handle SIGBUS nostop print pass`，再 continue。CPU faulthandler 夹具确认该配方打印真实的 `Program terminated with signal SIGSEGV`。另有 CPU 夹具验证 `--return-child-result` 传递普通退出码 23。

| 证据 | SHA256 |
|---|---|
| GDB launcher.log | `a33c38d46af7b28c6a44f0e309c6c2fc245b02681ee11a02985cf9c0de351303` |
| 原生合成 stage | `0eb1cd7e0843c29a826056d20cdd48b1196a3dac23b09a9f58b774654047e7e1` |
| 经 CPU 夹具验证的修订调试配方 | `b0b3ef79db8c1c38980e866d925cb0a8ca87c4ace0692acebbae68cb8b6cfb80` |

证据目录 `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/f6-gdb-drop-3IgyAO`。运行从 09:14:58 UTC 至 09:16:51 UTC。该轮未启动 hang，`native_render_verified=false`。可对比现有 IsaacLab headless rendering experience 的窗口依赖，保留相同相机和读回路径。
