# TiledCamera 两次预算：首轮失败与时钟注册修复

基线：d1469d29fd288c2b52d45ced096a95ee95c3ac2d。仅 opt-in 预览；旧控制、物理、PPO 和模型不变。

## 实际原生证据

本批首轮合同 SHA d1cafbd09d639a5ea0332394b92d2de259c063d81dd1742ead963f07b3ad4686，GPU0，320×240，1环境，0 PPO。
2个初始化步后 acquire 失败，0测量步、0有效双目帧，6行物理账本完整。
74.427秒；外层255、子进程SIGSEGV(11)，没有强制终止。本批尚余1次，旧版本不重跑。

首次错误为 `AnnotatorRegistryError: No annotator of name IsaacReadTimes registered`。
此前相机 Fabric 首次读位姿错误本轮未再触发，但这只说明执行越过该边界，不授予完整相机资格。
实际 SDK 的 `isaacsim.core.nodes` 扩展在 `on_startup -> register_nodes` 中注册 IsaacReadTimes；
直接导入 binding 不执行注册。旧 CameraView 路径启用 sensors.camera 时由依赖间接启用它，
新 TiledCamera 路径没有该前置。没有伪造时钟或删除时间检查。

关闭的独立错误：GDB `present thread` 栈 #0–#2 为 `libomni.kit.livestream.plugin.so`，
随后为 renderer/carb events。该栈不能证明全部 Tensor View 问题已消失。
manager/ownership readback guard仍报告unknown/false；正常关闭、资源释放均不合格。
完整栈、动态库映射和原始日志保存在服务器私有目录，不上传Git。

证据目录：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/live-stereo-preview-20261116/tiled-usd-batch2-20261116T063806Z/attempt-1`

native.log SHA：ca8a0d27a4363d0e63ea2d9700ea5eeda40655632e78f8e3196ec55733edff08。

## 最小修改

`tabletop_tiled_stereo_native.ensure_clock_annotators()` 在创建私有rig及传感器前启用已安装的
`isaacsim.core.nodes` 扩展，并确认公开 IsaacReadTimes 已注册；ReferenceTime 是 hidden，
保留真实 `get_annotator()` 获取作为权威校验，不能用公开列表错误拒绝它。
已启用时不重启；缺失立即报错。没有更换/安装SDK、Python、CUDA或依赖，
没有禁用共享扩展，没有修改物理、渲染计数或预算。扩展由App管理，私有相机不擅自卸载。

实际扩展注册源码：
`.../isaacsim/exts/isaacsim.core.nodes/isaacsim/core/nodes/impl/extension.py`
SHA 0e26204134e2648b88e93175b9deacefa95fd15cbe84bfc51ef0e3c5cdf837f1。

只读复审发现 hidden 注册表陷阱后，新增同 SDK 语义测试先确认3 failed/2 passed，修复后重跑完整回归。
CPU fixture不代表原生渲染成功。未运行额外Isaac。

## CPU与准入

新增5项回归；23个相关测试文件合计271 passed，0 failed，0 skipped。
CPU日志SHA：7c893cdf2e52439164b806600802726601c29598a873b2a273ff44c2f52fc83a。
命令为上一兼容包22项回归加 `tests/test_tiled_clock_startup.py`，经现有
`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q` 运行。

修正版尚未原生执行。下一步发布/同步后重新冻结源指纹，在同批剩余1次范围执行。
不恢复旧四档批次，不扩大预算；即使读出图像，关闭和时钟审计仍须分别通过。
Teacher/Student准入保持false，未训练、未获得抓取成功。
