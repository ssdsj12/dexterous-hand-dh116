# 双手 Pinch：真实 v4 零动作账本后的单个 8 步读回 profile

本包给出一个可执行的、带条件的最小动作建议：全新 owner 先重复原 8 步零动作区间，再执行唯一一个 8 步非零区间。只作小闭合目标与实际读回验证，不宣称建立接触、有效对向接触、无滑移、抬升或抓取。源码没有启动原生，没有修改任何旧 worker/config/material/关节/速度阈值。

独立 worktree 基线 d31b88c6f71e168a7b8f5f7ba6afd616ccdb3535；integration 不改。新增脚本、测试、报告三文件。原生新执行步数 0，training/optimizer updates 0。

## 冻结 profile 与实际父证据

最终 profile：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-pinch-readback-profile-20261113-dxcb62it/run02/profile.json
SHA256：67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498

其绑定结果：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-pinch-readback-profile-20261113-dxcb62it/run02/result.json
SHA256：75ec5cee02eca0676d15db8eb07d00273c2b50b40b7dc8d7d16d09d18e17f803

父账本是已通过 scoped native gate 的 dual Pinch v4：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-dual-pinch-zero-v4-20261113-utbAE0/native-runs/flex.f7-dual-pinch-zero-interval-v4-native/attempt-fz50kogx/native/physical.jsonl
SHA256：ead0dbf2c15b83864a1c440848ed9cf159b86c88a6ade0e4275208efebcf778c

原 receipt：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-dual-pinch-zero-interval-v4-native/9e95d33e8b0565133c05d8fae5eb5c5eb68de269806ae561e4c5bd022bd9e202/attempt-fz50kogx/receipt.json
SHA256：aa1a6877c29b460e8f12589674d4ba09381faf8aea200400a7cd0e71d1a5a1ec
原 receipt native_qualification=true、exit_code=outer_exit_code=0；本脚本仍重放原 inspect_interval，确认 2 setup + 8 physical，SDK 3..10、rod 12..96、一完整非 terminal control interval。父账本内部 Teacher/grasp 资格仍 false，该次通过没有给未来非零动作授予资格。

配对 reset 原件：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/paired-pinch-reset-BaP0hx/result.json
pair SHA：996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37
reset seed SHA：647247fe31219f04337b6f6e29ddefaa2c81bcee27dbed8dd3d6f962e406230d
原 action contract SHA：498cebedfe520cc2f6593b32776581c3da826ba95e4db441b7938d29176c55d9

future worker 只应按字节读取 profile，并严格核 SHA；**勿在 AppLauncher 前 import 本几何审计脚本**。profile 字节和 run02 原始结果不可修改，亦不应复用 run01 的中间 profile。

## 唯一动作及预算

| action24 index | 原命名通道 | 归一化动作 | 名义目标速度 |
|---|---|---:|---:|
| 1 | left.finger12（thumb） | +0.1 | +0.030 rad/s |
| 2 | left.finger21（index） | +0.1 | +0.030 rad/s |
| 7 | right.finger12（thumb） | +0.05 | +0.015 rad/s |
| 8 | right.finger21（index） | +0.1 | +0.030 rad/s |

其余 20 分量全为零，包括 thumb-spread、其余手指、世界平移 indices12..17 和 intrinsic rotation18..23。左右坐标的正值均增加原命名关节角；这里分别用两侧真实几何验证方向，并非镜像左手。右拇指保留原独立配对路径相对食指的一半速度比例。幅度为原配对 CPU 速度的 0.4 倍。

~~~json
[
  {
    "name": "repeat_original_zero_interval",
    "sdk_steps": 8,
    "action24": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  },
  {
    "name": "one_nonzero_readback_interval",
    "sdk_steps": 8,
    "action24": [
      0.0,
      0.1,
      0.1,
      0.0,
      0.0,
      0.0,
      0.0,
      0.05,
      0.1,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  }
]
~~~

8 个 480Hz 步 = 1 个 60Hz control interval。名义总增量为左 thumb/index +0.0005/+0.0005 rad，右 +0.00025/+0.0005 rad。执行原 finger_pre_guard、surface_final_guard 后，每次 submit 的 float32 写回作为下一步 backend target 输入；最终实算增量约为 +0.0004999638/+0.0005002022、+0.0002500117/+0.0005002022 rad，精确九行在 result.target_rows_rad。原 .3 rad/s 限制、.1 rad lead、affine coupled limits 均未改。这里冻结 measured q 计算 target arithmetic，实际几何则评估 q 跟踪目标的条件轨迹，两者都不是动力学保证。

已退出的旧 owner 不可恢复为本次控制 owner。因此新进程需 **2 setup + 8 zero + 8 nonzero = 18 SDK steps**，测量16、两个完整interval。profile 仅建议一个非零interval，不自动重复。通用预算校验拒绝非整数、非8倍数或超出8..240的测量步数，不能用238个物理样本拼成完整控制interval。原静态1.9mm与原runtime2.1mm分开，实际本包采用2.1mm。

## 真实读回存在什么，缺什么

父账本每一步有真实 17DOF q/qd/backend targets、实际 mount anchor、同钟杆节点及 pairs；23 个实际 link transforms 只在 initial_geometry 记录。逐步 exchange 没有落盘 post pad/link poses，physics agent 也独立核对了这一字段缺口。故本包的 post pad pose 是 **实际初始 link pose校准 + 实际 post q/anchor 的 FK 重建**，绝不标为直接实测。

按原 mount authoring，从实际 anchor quaternion、三局部 prismatic q、intrinsic XYZ q 重建 base_link，再复用原 USD FK。初始 actual-to-FK 最大位置残差左80.003nm、右50.372nm；旋转残差最大约7.397e-7/6.253e-7 rad。8行实际 mimic最大残差约4.983e-8/1.964e-8 rad。它们说明当前静态零动作锚定一致，**不能界定未来非零动力学或柔性杆误差**。

最后一帧真实41节点相对初始杆的最大偏移约89.485nm，当前节点仍共线（记录精度内 chord deviation=0）；历史最大节点速度3.833e-5m/s。未来杆是否仍如此未知，预测固定这份真实节点位置，不合成未来杆状态。

每个 pad 1mm SDF NPZ 的路径/SHA/cooked capture SHA 在 result.sides.*.sdf_binding；source USD/shape SHA 与实际 initial_geometry 独立核对。姿态校准后的原17 source与5 locked cooked形状分别检查；没有声称12个非pad live cooked已重新读取。

## 9 个条件状态的几何与 node-SDF

每侧用实测 post q 起点及8个真实控制核/FP32目标状态，检查全部22形状。每相邻段按原 per-link 点位移上界 B 扣 B/2；原 finite-axis 支撑平面下界另扣实际节点到端点chord的最大偏移，覆盖这份固定节点折线。桌面与杆保护均有2μm数值allowance。**这个allowance只是CPU数值buffer，不是未知future tracking的物理误差界**。

| 条件预测 | 左 | 右 |
|---|---:|---:|
| 起点最小桌面间隙 | 2.110844617 mm | 2.116352433 mm |
| 整8步路径桌面下界 | 2.104922121 mm | 2.112601726 mm |
| 相对原2.1mm门余量 | 4.922121 μm | 12.601726 μm |
| 非目标杆表面间隙下界 | 2.315695771 mm | 0.729100362 mm |

原非目标保护仍含 finger13 等近端。如此小的左侧余量不能作为长路径或真实无接触保证，必须用新实际逐步几何检查。

| pad | 实际post q重建的node-SDF起点 | 8步目标末态 |
|---|---:|---:|
| left thumb | +1.613433473 mm | +1.584897283 mm |
| left index | +2.054444049 mm | +2.014470752 mm |
| right thumb | +1.867071260 mm | +1.851300709 mm |
| right index | +2.454303671 mm | +2.410255373 mm |

四条9点曲线逐点下降，所有gap仍为正。本动作旨在验证命令方向、目标和实际追踪/几何读回，不会依据这些CPU值宣称接触建立。采样逐点保留实际float32三线性phi-R、梯度与域内node IDs；单个node在grid外是正常的不可用值，不填零。必需pad完全没有有效域内minimum才是读回失败。

对原配对长路径末态只做一次节点查询，左thumb/index仍+0.161451753/+0.251743477mm，右+1.113573555/+0.252952799mm。因此原连续capsule/凸包近接触并不保证原12.5mm节点与1mm SDF满足phi-R<0，尤其右thumb。没有重新扫161点长路径或生成238步闭合推荐。

## 后续 worker 必须保留的停止边界

profile.required_fail_safe 是可审查要求，不是本脚本已经安装的原生回调。主线程单独实现新worker。本包没有变更旧接口。

- 先预留完整8步区间；same run/env/episode/reset/rod身份，SDK每步+1、rod_tick每步+12；部分推进后失败必须记录并关闭，禁止重试伪造完整区间。
- 原 joint/mimic/nonfinite 和每pad force>30N终止逻辑不变，不能添加补偿力或放宽物理材料。
- 新落盘必须包括真实post named link transforms/pad poses、q/qd/targets、同钟rod节点。源17和locked5的实际每手桌面间隙必须>=.0021m；非目标几何不得为负。
- 腕动作全部零，但原 safety projection仍可能主动抬手。必须保留该原保护；若它介入抬升/旋转，则停止本次“无抬升读回”假设，不能静默抑制投影或把动作零当物理不动证据。
- source-invalid/SDF必需pad无有效域内minimum、异常力或任何保护失败均停止，保存partial实际步数，不自动加步。
- static GPU hand-table原生contact仍NA，不能把几何正间隙或未报告contact称为0接触。Teacher、完整371D、R0和grasp资格仍不由profile获得。

新的epoch可重复相同reset seed，但不冒用历史run UUID。每次新owner的零动作后必须重新确认实际初态/clock/源数据，不能只核历史parent SHA。

## 验证与原件

先写测试得到预期缺失模块错误。新增FP32往返负例先复现理想double累加与实际写回差异（单步最大2.527e-8rad），修正为每次submit后cast float32再读double。run01原件和当时执行源码保留，run02为最终冻结结果。一次相关测试选择拼错文件名，0 tests ran；纠正路径后无测试失败。

最终 **19 新 + 56 相关 = 75 CPU tests passed**：40项（新profile+SDF helper）2.74s，35项（原wrist projection+bimanual step）2.19s。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q tests/test_dh116_dual_pinch_readback_profile.py tests/test_dh116_pinch_closure_margin.py
# 同环境
bash scripts/python.sh -m pytest -q tests/test_dh116_flexible_wrist_projection.py tests/test_dh116_deformable_bimanual_step.py
~~~

最终run02脚本计时4.111141370s、外层wrapper子进程6.832618620s、exit0。内层50s软限从导入之后开始，外层90s；实际为complete。每侧9个路径状态、一个旧长路径端点查询，0原生步。run01外层51.38s主要为导入/冷启动，其脚本段7.997s；保留为历史中间结果，不能替代最终profile。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh scripts/prepare_dh116_dual_pinch_readback_profile.py --output <new-exclusive-directory>
~~~

证据目录：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/dual-pinch-readback-profile-20261113-dxcb62it

| 原件 | SHA256 |
|---|---|
| run02/result.json | 75ec5cee02eca0676d15db8eb07d00273c2b50b40b7dc8d7d16d09d18e17f803 |
| run02/profile.json | 67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498 |
| run02-outer.json | 44bd52f191b9220273f6dad5e31bdc3ef5d16fa936fcb19cba3f0a430648ad4c |
| offline-verify.py | b82c008c0477ae0b294dab233bf4e3d684b883bb205a417309baccdfbc9d6417 |
| offline-proof.json | 1f4a7aa508721dbe24858d0246529206e415aaa6cd776210b87f78a92100220a |

离线proof核11源码SHA、10SDF SHA、原native账本重放、原控制核FP32九行目标、18个保存的几何状态及其分段界限代数。它没有新增FK/QP/node-SDF求值，也不代替独立原生测量。执行脚本SHA：ebaaf8f60b581da6b1113c0c9354c4d5803d57d8004e4e9f9438b494b9ae810e。
