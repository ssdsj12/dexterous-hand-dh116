# TiledCamera 批次复验：真实双RGB已读出，关闭仍阻断

本报告记录实际执行，不是CPU fixture或离线复放。源基线 f8e321d8f2318a762d27742fd33f843134cc7d53；PR #2继续Draft。

## 两次运行与预算

| 项目 | 首轮 | 修正版第二轮 |
| --- | --- | --- |
| Source HEAD | d1469d29fd288c2b52d45ced096a95ee95c3ac2d | f8e321d8f2318a762d27742fd33f843134cc7d53 |
| 合同SHA | d1cafbd09d639a5ea0332394b92d2de259c063d81dd1742ead963f07b3ad4686 | 0768a1fad59ef0b4d1459adc20abf807a8783862262462df7f6fc932dc2999a3 |
| GPU / 环境 / RGB | 共享GPU0 / 1 / 320×240 | 同左 |
| SDK步（含2初始化） | 2 | 130 |
| 测量物理步 / solver子步 | 0 / 0 | 128 / 1536 |
| 左右RGB对 | 0 | 17（34 PNG） |
| 物理JSONL行 / 不完整行 | 6 / 0 | 166 / 0 |
| 墙钟 | 74.427s | 150.401s |
| 外层 / 子进程 | 255 / SIGSEGV11 | 255 / SIGSEGV11 |
| PPO更新 | 0 | 0 |

合计132个SDK步、1536个测量solver子步、224.828秒，低于本批260步/3072子步/360秒软截止总上限。
每轮显式render仍受源码162次硬计数限制；崩溃前没有持久化实际总render计数，不能把上限写成实测值。
两次已用完、剩余0；没有同版本重跑、扩预算或强制终止。

## 实际修复与RGB结果

首轮失败是已安装的 `isaacsim.core.nodes` 没有启用，IsaacReadTimes未注册。
最小修复 `ensure_clock_annotators()` 已在 930ab04 发布，f8e321d为发布核验提交。
修正ReferenceTime为hidden的检查陷阱后，全CPU回归271 passed；没有安装/替换SDK、修改物理或动作。
第二轮 acquire 正常返回：初始化帧1对，16个60Hz控制间隔各1对，共17对。
实际图像检查可见两只DH116和由真实DeformX节点驱动的橙色杆Mesh；左右图不同。
本轮动作始终零24D，属于预览，不是Teacher或Student抓取评估。

标定SHA aab774c7b2a40d13d9f72b77249523fba838856a7068170f884ffb6f71888587；
基线80mm，fx=fy=208.5160596545929 px，主点(160,120)，按320×240重新设置相机内参。
传感器内参、USD实际位姿与冻结标定的运行时校验通过；没有缩放640×480图片冒充原分辨率。

## 时间与身份

17个采集时刻的SDK step为2,10,18,…,130；rod tick为0,96,192,…,1536。
128个测量物理记录连续为3…130，16个测量帧逐项匹配原始rod账本的
run_uuid/env_id/episode_uid/reset_id/rod_uid/rod_tick和物理步。
双视图是在冻结同一物理状态时顺序渲染，保留不同原始render frame ID。
每对左右render simulationTime相等；SDK时间与render simulationTime稳定相差
0.004166666883975267秒，即本轮2个初始化步。原始时钟未被重写、补齐或伪装成相同frame ID。
这表明软件捕获边界和账本关联有效；像素级运动时间、极线/深度精度、Reset重启和正式同步资格仍未授予。

## 关闭失败与资源所有权

第二轮没有acquire错误；四项关闭guard仍为false/unknown：manager_readback、ownership_readback、
manager_before_close、ownership_before_close。before_close的弱引用审计仍看到SimulationContext、
左右Articulation/ArticulationView/SimulationView、WrenchComposer、runtime/rope和preview_rgb。
simulation的referrer还包含Tiled reader构造帧、from_native/create_preview_camera完成帧、闭包及_sim字典。
这些是具体的后续所有权调查入口，不是已证明唯一崩溃根因；Python对象存活、native view释放和GPU完成必须分别验证。

GDB首次故障为present thread，#0–#2位于 `libomni.kit.livestream.plugin.so`，随后是renderer/carb事件。
正常app.close未返回，外层255/子进程SIG11；没有os._exit、SIGTERM或SIGKILL。
本轮进程已退出，GPU0用量回到本次启动前13865MiB，但不能据此把native资源释放guard改成通过。
自然关闭和完整生命周期继续blocked，不能因为已经出图而授予Teacher/Student准入。

## 证据、测试与保护

私有原件目录：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/live-stereo-preview-20261116/tiled-usd-batch2-20261116T063806Z`。
每轮保留start、冻结合同、完整physical/stage JSONL、GDB首栈和动态库映射、outer退出码和summary；
第二轮另外有左右PNG、逐帧JSON、capture_calibration、audit.json和file-manifest.json。
原始模型、日志和图片不上传Git。Git中结果摘要为 `docs/contracts/tiled_step_native_result.json`。

第二轮native.log SHA 74f85dfc079c00625258e6643b5ae416aaa6ae71a06e2659f49a224f771967a9。
包含34张PNG及帧记录/标定/账本/日志/summary的清单SHA：
e79f8c79b0f57c91031b1930293732458bc735134ac33ab0439b10d6e541ab00。

23个相关CPU文件271项通过，0失败、0跳过；发布器会再次执行同组回归并保留实际日志SHA。
原生与CPU结果分别报告；保护检查覆盖528文件与两个历史Checkpoint，原资产、控制器、模型合同和成功标准不变。

执行命令：
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python bash scripts/python.sh -B scripts/preview_dh116_physical_webrtc.py --freeze <private>/attempt-2.contract.json --camera-backend tiled_step --resolution 320x240 --physical-gpu 0
# 冻结后经批准的run-one.py仅调用一次 --execute-native；GDB wrapper仅包装此子进程。
python3 <private>/run-one.py
```

## 下一工作包与门槛

离线追踪剩余构造帧/闭包/手和solver所有者，定位具体释放责任；同时审计WebRTC present回调与正常关闭顺序。
以失败测试覆盖已确认问题后再修复；本批不再启动Isaac。
其他分辨率尚未用本修正版原生验证，不恢复旧四档预算；后续原生需要新的冻结配方和预算。
尚无Depth MAE/完整率或抓取成功数据；640×480仍为主要精度验证目标，并未固定320×240为上限。
Teacher、Student、BC/DAgger训练准入保持关闭；30mm/5s/0.5mm标准不变。
