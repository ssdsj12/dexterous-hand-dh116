# Isaac Sim 4.5 相机生命周期 C0–C3 对照

基线为 `eb31c793d60cb50eb8d6dd2e06bae2f15c5bd8d5`。本次完成入口、固定输入配方和 CPU 核验。C0–C3 均未启动 Isaac，也未获得新的关闭或相机资格。

## 已知失败

原生无 stage V0 使用 `aaaa97e8528538225cd09e3975cf0e4e98158dac`。实际读取确认 `create_new_stage=false` 且 SDK 无 stage。应用关闭回执已落盘，外层仍以 `-11` 退出，耗时 32.227 秒，调度器观察到全局锁。日志 SHA256 为 `dd10fd1ae95f36c071e66ef934fbbae6f430a71534da2208751ef1033ecc2cfe`，该次 Python 记录了段错误。此前 drop 资产入口的 GDB 回溯涉及 AppWindow/Carb 和 Python finalize，资产入口的回溯不能用来认定此 V0 的原生栈。

旧 V0 日志显示 Kit 针对 `CUDA_VISIBLE_DEVICES` 发出设备编号和稳定性警告，并跳过七块 GPU。该证据支持保留设备可见性这一独立待查因素。C0–C3 在同一 GPU3、同一调度器设备可见性下执行。调整设备可见性需要独立实验，以先确定 Vulkan 和 CUDA 的实际物理设备映射。

## 已安装的启动路径

NVIDIA 4.5 standalone 示例在 `SimulationApp` 构造后导入 Omniverse API，显式调用一次 `update()` 和 `close()`。新探针在启动前只加载标准库，记录并拒绝预加载的 Kit、USD、Torch 或项目模块。[Isaac Sim 4.5 Python Environment](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/python_scripting/manual_standalone_python.html)

安装读回为 Isaac Sim `4.5.0-rc.36+release.19112.f59b3005.gl`，Kit `106.5.0+release.162521.d02c707b.gl.linux-x86_64.release`，Kit commit `d02c707b20b56088d49fa5d47df9b4a2d2b9dc86`。配方绑定安装的 `SimulationApp` 源码及工具文件、SDK 版本文件、Kit 包元数据、Kit app 插件二进制和所用 experience。探针记录当前进程实际映射的 AppWindow、Carb、Kit、PhysX、Omniverse、CUDA 与 NVIDIA 共享库路径，并记录 `LD_LIBRARY_PATH`。映射记录和包元数据各自保留来源。

安装的 `SimulationApp` 默认创建空 stage。关闭实现会尝试停止 Replicator、按默认参数等待后台任务、关闭其 stage，再调用 Kit shutdown。构造时还包含 SDK 自身的 UI/viewport 准备与内部 update。测试中所称“一次 update”是探针的显式调用。读取 active viewport 只使用 SDK 已加载的模块，记录该视口的 render product 路径，并另从 stage 读取 USD RenderProduct。原生对象引用只存在于读取函数的局部作用域。

## 有界对照和预期证据

| 实验 | Experience | SDK stage | 对照 |
| --- | --- | --- | --- |
| C0 | 4.5 SDK 默认 Python experience | 默认创建 | 官方最小生命周期控制 |
| C1 | 固定 Isaac Lab headless rendering | 禁止创建 | 独立最小渲染生命周期 |
| C2 | 与 C1 相同 | 恢复创建 | 相对 C1 仅改 `create_new_stage` |
| C3 | F3 使用的固定 headless no-camera | 禁止创建 | 相对 C1 仅改 experience |

C1 保留旧 V0 的 rendering experience、stage 配置及 Orchestrator/persistence 启动参数。入口改为最小 SDK 调用，去掉显式 core.nodes 和项目模块导入，改用与 C0、C2、C3 一致的 SDK 默认 `close()`。因此 C1 与旧 V0 的差异含多个启动和退出条件，二者结果不足以归因到某一项。C2、C3 的变量控制基于本表 C1。C3 只验证关闭生命周期。

每格独立目录，GPU3，最多 180 秒预算，0 物理步、0 更新、1 个进程，无项目场景、DeformX、相机和额外 render product。调度前检查至少 4096 MiB 空闲显存、1 GiB 磁盘和单实例占用。Unix 文件锁与现有项目一致，探针保留文件描述符直到 Python 进程退出。

- `plan.json` 保存解释器、Python/Isaac 包版本、环境、前置导入和请求配置。该文件在调用构造器前写入。
- `before_explicit_update.json` 和 `before_app_close.json` 保存实际 experience 及 SHA、Kit 包身份、启用扩展、stage、viewport、render product 和运行时共享库。
- `app_close_returned.json` 只标记应用关闭函数返回。自然退出、真实信号和进程日志由外层调度器独立记录。
- 四份配方均使用相同的 `gdb_on_segv` 及固定 GDB SHA。以同次 `native.log` 的当前线程、全线程回溯、共享库和映射，以及 `outer_exit.json` 的 inferior 证据为准。GDB 自身退出码按实保存。

配方为 `configs/dh116_execution/deformx_camera_c0.json` 至 `deformx_camera_c3.json`。每格通过真实外层退出 0、全局锁观察和产物校验才获得相应生命周期证据。C0–C3 没有相机采样。只有合格的空渲染生命周期后才继续原有 V1–V4。仍需原生复验同一渲染状态的 RGB、ID、深度及不变的物理时钟。

## CPU 回归

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
  bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_simulation_app_lifecycle.py tests/test_dh116_render_capture_boundary.py
```

新增测试先因入口缺失而无法导入。实现后 19 项通过，含 9 项新测试及 10 项既有渲染边界测试。覆盖四份配方的共同 GDB 版本、C2/C3 的单变量差异、有效运行时库路径、SDK 导入边界、无 display 准入、CPU 预案、Kit commit 读回和关闭失败时的产物保留。下一步由单实例调度器在已批准顺序内绑定实际集成 SHA 后执行。
