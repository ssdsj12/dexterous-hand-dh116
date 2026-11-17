# AppWindow 持有者定位与同钟深度证据

基线为 `2ef7f90ed9af9039dd32fb197063040ae11996f3`。Ubuntu 集成、Windows 裸仓中开发分支的远端跟踪引用和 GitHub PR #2 head 一致。PR #2 为 open、Draft。

## 从本轮栈定位对象类型

复用 [C0–C2 原生报告](2026-11-12-camera-c0-c2-current-native.md) 对应的本轮同次 GDB 栈与 mappings。三个进程均在 `app.close()` 返回后，由 Python 主线程在 `Py_FinalizeEx` 路径触发 `SIGSEGV 11`。三格首次故障在同一 `_appwindow` 映射文件偏移 `0x15e55`，紧邻的 `_carb` 两帧也一致。C1 无 SDK stage 和 USD render product，仍存在默认 viewport 对象。

对实际映射的 Isaac Sim AppWindow 1.1.9 绑定进行静态核验。其 `_appwindow.cpython-310-x86_64-linux-gnu.so` SHA256 为 `cb563891e1f1a921d108fef3cd19fc7fe5f9e34b606945b888722446f672d3b8`，build ID 为 `63c333ee1ae4878fe05ff9c6d583aa4fd940c976`。

- 从 `objdump -d --start-address=0x12360 --stop-address=0x12490` 可见，类型注册在 `0x12374` 引用 `.rodata` 的 `0x417a4`，该字符串为 `IAppWindow`。同一注册记录在 `0x123e5` 安装 `0x15e10` 的回调，并在 `0x123f4` 引用 `0x43508` 的 `Application window carb object` 文档字符串。
- 从 `objdump -d --start-address=0x15e10 --stop-address=0x15e90` 可见，该回调保存 Python 异常状态，读取并清空持有的 C++ 对象指针槽。它通过对象 vtable 的 `+0x18` 槽间接调用，首次故障落在此指令 `0x15e55`。据此可把故障缩小到 Python `IAppWindow` 包装对象释放路径。当前二进制缺少该目标方法的符号，三次旧 GDB 记录没有当时的寄存器值，尚无法确认具体窗口实例、虚方法目标和保留包装对象的扩展。
- 安装的 AppWindow `_appwindow.pyi` 把 `get_default_app_window()` 描述为默认窗口指针读取，将手动创建和销毁窗口列为不同接口。安装的 viewport 扩展会持有 UI/viewport 事件处理器；其中输入事件代码可获取默认 AppWindow。其 `on_shutdown()` 销毁所拥有的 viewport 条目并清除 scene 注册和事件代理。当前源码可以界定所有权路径，尚无法把某条 viewport 路径与此次残留包装对象绑定。

## 待授权的生命周期对照

C3 保留原配方。相对 C1，只有安装的 headless no-camera experience 变化。由于保留相同的显式启动参数，C3 是 experience 对照，不能以 F3 先前的物理测试结果代替。

C4 沿用 C1 的渲染 experience、无 SDK stage、headless、普通 `SimulationApp.close()` 和同一 GDB。唯一启动设置变化是 `--/exts/omni.appwindow/autocreateAppWindow=false`。安装的 `omni.appwindow/docs/SETTINGS.md` 明确该设置控制自动创建窗口，其默认值为 true。安装的 `docs/USAGE_PYTHON.md` 将 `/app/window/enabled=false` 用于 virtual window，C4 采用自动创建控制来检验默认窗口对象这一假设。

C4 启动后和关闭前均记录实际设置值、AppWindow 扩展启用与默认指针状态。探针仅通过 SDK 已加载模块的 getter 做瞬时读回。回执须同时证明扩展仍启用、autocreate 为 false 且默认对象确实不存在。原关闭路径、信号处理和 GDB 退出校验继续生效。若仍在同一地址崩溃，则默认窗口对象的自动创建不足以解释故障；若退出自然通过，仅取得该隔离生命周期的证据。

| 配方 | Command spec SHA256 | 本工作树 CPU 预检 code/config SHA256 |
| --- | --- | --- |
| C3 | `47fd6d1b18c2f2cb40ed9a6082d564f5e4cd0982d9d8113398fc0588586ac4e3` | `369198f95959a46dddbdc6f415d520e872f43fefcd0af572e6c99533d1a9919b` |
| C4 | `ef59621be11834327c73be02da5391b9e117c46a8851f863cab1b0af5ac55d49` | `41b48a5af713ab8b453621882188b531cddee7f05bfa1fa7db77e3b0a0aa3a7d` |

每格独立申请至多一次，GPU3 单实例，180 秒墙钟预算，至少 4096 MiB 空闲显存和 1 GiB 磁盘。一次显式 App update，0 PhysX 步，0 PPO 更新，无相机采集。先由最终集成调度器重新冻结 SHA。C0–C2 保留原证据。C3、C4、V1–V4 本工作包均未运行，未新增原生生命周期、相机或数据资格。

## 一次渲染采样的产物验证

- `_read_products()` 要求 RGB 为已知 `H×W×3/4` uint8。深度和原生 ID 接受 `H×W` 或 `H×W×1`，显式拒绝同元素数量的换轴图像及扁平数组。按已安装 Replicator 声明保存 float32 径向距离、float32 光轴 Z、uint32 ID 和两种独立有效掩码。原始无效数值保留。
- `capture_at_boundary()` 执行一次显式渲染，保存每个相机产品读取前后的实际 SWH/ReferenceTime/IsaacReadTimes 时钟，以及渲染前、读取前、读取后的物理状态。V4 仅在首个边界允许既有 render-only 预热，后续已就绪边界直接同钟采样一次。V4 仍为现有刚性代理实验。
- `audit_record()` 从同一已校验 SHA 的 NPZ 字节重新加载图像，核对分辨率、数据类型、两种掩码、相机内参及其 SHA。独立复算像素中心针孔投影和径向到光轴 Z 的 P95，并保留原 2 mm 门槛及 V3 已知 1.98 m 平面核验。独立核对左右同钟、跨读取物理状态和 V4 标定身份。该审计是 CPU 产物复核入口，原生调度器的退出及资格规则保持原样。未得到实际原生 annotator 输出前，无法确认原生语义、采集时钟或动态任务数据。

## CPU 复核

```sh
export DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_simulation_app_lifecycle.py tests/test_dh116_render_capture_boundary.py
```

先运行新增回归，9 项按预期失败；实现后 30 项通过。合成反例覆盖换轴、深度误标、渲染时钟变化和物理状态变化。C3/C4 使用现有调度器解析并生成 CPU-only 预案，528 个保护文件和 2 个历史 Checkpoint 已核验。
