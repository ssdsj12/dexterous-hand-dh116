# 右手 Pinch：独立来源的有界闭合筛查

基线 059d8e309b7044709f0dc32778a57c15376a734b。本包只新增独立 CPU 脚本、测试和本报告；未修改任何此前工作包、配置、物理参数、限位、资产或数据。未启动 Isaac、GPU、solver 或策略，真实新增物理步与 optimizer updates 均为 0。凸 QP 仅用于既有连续凸体距离计算，不训练或更新任何模型。

结果：**450/450 候选完成，0 数值错误、0 未完成候选；10 个通过 0.5 mm 粗闭合门，只有 1 个同时满足 0.05 mm 端点间隙。按本轮上限只检验该最佳候选的一条完整路径，未通过。** 不生成 registered/native candidate，不授予 Teacher/reset/grasp 资格。其余 9 个粗门候选未做完整路径验证；本轮不能据此断言右手不可达。

## 来源、固定网格与原门槛

两个来源均为独立右侧，未镜像左手：

- original_right：原 pose_reference.json 的 resolved_poses.right / pose_contract.right，与原右 USD/F4 cooked capture SHA 相绑。
- derived_right：right_pinch_pregrasp_v1.json 的右半 active_angles 和右 wrist_pose，配置 SHA256 13ee6dfb0f5bfe1a349667f138f155fe55c3c41e8b0abac4a22fc822a6851888。该配置自身 geometry_only=true、qualified=false，明确 stable physical lift 未通过。虽然引用旧 case35/frame4799，本轮未找到或打开原始 unrestricted_grip_lift_20261030_c archive，不将它称为真实成功 seed。

复用原 load_geometry/world_vertices、clearance_corrected_pose、measure_closure_grid 和 verify_candidate_path。每个右 seed 固定 225 个候选：roll [0,-10,10]° × pitch [0,-10,10,-20,20]° × thumb/index 初始角度 bias [0,.025,.05] rad × world X [0,-.002,.002,-.004,.004] m。两个来源交错枚举，总计 450。每个姿态按全部 17 source collider 与 5 locked cooked pad 的最低点竖直平移到 2 mm；不放宽既有 1.9 mm 路径净空门。

原闭合 helper 在两个 active channels 1/2 各固定采样 9 个角度；沿用 160/480 s 和 0.3 rad/s 上限，最大角增量 0.1 rad，不优化姿态或更改限位。粗门为两 pad |gap|≤0.5 mm 和原方向 dot<−0.2 等条件，**不等于**旧 left candidate 的 0.05 mm 严格端点门。最佳完整路径按原 161 个样本与 lever-arm 采样间隔界检查 source/cooked 表面及非目标 collider 对杆的连续包络。

全部 17 live cooked hull 尚不可用，12 个 non-pad 仍为 source USD convex envelope。没有 self/inter-hand native 接触或真实动力学跟踪证明；geometric direction 不是实测 contact normal。

## 实际观察

原右基线复现：source/cooked 最小桌面净空 −0.983093349 mm，竖直修正 +2.983093349 mm 后为 +2 mm；thumb/index 初始 settling-envelope gap 为 **1.310181061 / 3.764855511 mm**。

| 来源 | 完成 | 0.5 mm 粗门通过 | 同时满足 0.05 mm 端点间隙 | 最佳端点 thumb / index gap |
| --- | ---: | ---: | ---: | --- |
| original_right | 225 | 2 | 0 | −0.001344 / −0.112286 mm |
| derived_right | 225 | 8 | 1 | −0.018361 / +0.046951 mm |

original_right 最佳为 r0/p−10/bias0/x0，active delta=[0,.0375,.0625,0,0,0] rad，几何方向 dot=−0.363684，端点全形状桌面净空 4.310146 mm；它只有粗闭合证据，未通过 0.05 mm 端点门。

唯一严格端点候选为 pinch-right-derived_right-r10-p10-b0.025-x0，active delta=[0,.0125,.025,0,0,0] rad，方向 dot=−0.834368，端点全形状桌面净空 2.950197 mm。它的初始 source/cooked rod gap 均非负、reset geometry 层通过，连续桌面最低界为 1.995623 mm；但非目标 finger13_link 的 source rod 间隙下界为 **−0.036688362 mm**，原路径 gate 给出 non_required_collider_rod_swept_envelope_overlap。因此不能证明整条路径排除该近端 collider 的杆包络接触。负下界不是新的 native 碰撞因果证据；reset_is_native_verified=false、effective_opposed_contact=null、native_ready=false、teacher_admitted=false。

## 预算、错误与可重放证据

第一次直接调用 Python 因缺 pxr 在加载前退出，0 候选，exit=1、0.412844 s；保留目录 right-pinch-closure-cpu-20261113-UtWy3Z，不删除或伪报完成。随后核对并使用原 scripts/python.sh 的离线 USD 路径，它只配置库与 CPU 环境，未 launch/import app 或 solver。

正式运行采用 2 线程、空 CUDA_VISIBLE_DEVICES，脚本软期限 45 s，外层 timeout 60 s（TERM 后额外 5 s kill grace）。**实际脚本 37.404986 s，外层 38.366529 s，退出码 0**。451 条 journal 为 450 个完整 coarse row 和 1 个完整 path check。运行完成后不再进行候选筛查或路径计算。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
timeout --signal=TERM --kill-after=5s 60s bash scripts/python.sh \
  scripts/screen_dh116_right_pinch_closure.py \
  --output /absolute/new/private/run01 --max-candidates 450 --wall-seconds 45
~~~

实际证据目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/right-pinch-closure-cpu-20261113-oIQiFC

- run01/result.json：1a271578961192096d3d4aa5ec03785807c3940908bd9444fd0c285be86dff1f
- run01/progress.jsonl：d42d390f7d9d89688d94e7be7d9646d49d9922d65b33f9dde557c077dd5d4b8b
- outer.json：582de2baea797fc3f7130b912785dfbe1b1a4bdbea1d95f68e81144d28c9a083
- run.log：9f6e36f16d3d548c62234de1056b184b758addb920d9222ed9fefdeac257a62b
- offline-identity-proof.json：b7271772de568bf2c39365a9f74cc0854fc5660339e20e1b520856c950198bb9

正式运行后新增超时测试发现并修复一处记账问题：若在 progress callback 触发 deadline，已完成 row 原先可能仍进入 remaining list。最终实现用短 SIGALRM 屏蔽区一起提交 row/count/identity/journal，离开区后再处理 pending deadline；path 完整结果也同样提交。partial 状态保留完整已完成 rows、错误和未完成 candidate IDs，未完成 path 不记为通过。进程外层若被强制终止则仍必须按外层非零状态记录，不能靠零散 journal 认定成功。

原执行脚本已独占保存为 executed_screen_source.py，SHA256 **4a4863f3743d3c450448b52bc2ec27c6c997ddbafd573ccc7a1631e3c7b8de70**。最终脚本 SHA256 **06e39f8c801d399eef2adc1e60238e8f9e78701e8ca7291670368af8e97b08c7**。原 result/journal 未修改，也不把原执行重新标记成最终源码。独立离线 proof 精确核对 450 个唯一 row 与 journal、全部数值 helper SHA，以及 load_seeds/_specs/candidates AST 未变；它没有重新计算几何。

## CPU 回归与下一边界

**11 项新增、相关 38 passed / 44.67s，退出码 0**。先观察缺模块失败；另观察 interrupt-after-row 导致 remaining 重复的真实失败，再修复。测试覆盖 450 固定交错网格、仅右侧输入、seed 不变、严格候选数预算、partial 完成身份、数值异常、不完整 path、最多一次完整 gate、真实 SIGALRM 延迟至 journal/identity 一起提交。git diff --check 通过。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_right_pinch_closure_screen.py \
  tests/test_dh116_flexible_style_pose_preparation.py \
  tests/test_dh116_pinch_initial_table_clearance.py \
  tests/test_dh116_flexible_power_pose_preflight.py
~~~

目前只有有界几何闭合线索，没有通过整条路径的候选。要进一步推进，应先处理/核实 finger13 的近端包络限制及剩余候选的完整路径，再谈 native reset/contact；本包不自动扩大搜索、不修改早期配置、不请求原生或训练预算。
