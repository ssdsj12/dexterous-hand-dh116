# 独立 approach/control 数值候选：现有 Pinch 账本重放

本包提供可直接调用的纯数值 composer，解决已有真实 approach/joint 与控制代价缺少同窗、同单位和唯一计分所有者的问题。没有安装训练 reward、创建模型/optimizer，原 R0 十项 null、task/config、collector、原生配方不变。

基线 8a7b5f38dee9de5970ea3ad84358d49c37492344；树 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approach-control-composer-20261114。

## 接口与数值

ApproachControlComposer(candidate_spec(approach_points_per_m)) 必须明确提供一个正有限 points/m 候选系数，没有默认。compose(window) 消费原 physical event、原 live R0 row/end 和原 readonly-control-cost 窗口，核对 actual-event SHA、R0重复保存行、episode/reset、SDK/rod逐行时钟、action24/上一动作、完整17DOF qd、终止位置与原始行和。成本再次调用既有 formula_components，不复制算法。

候选 = 显式系数 × signed approach difference + 原 time/action-rate/velocity-excess + 一次 joint-limit terminal event。距离差不再乘 dt；成本已乘 1/480 s，控制窗直接求和、不除8。joint-limit 只能在真实 left:joint_limit 或 right:joint_limit 最后一行支付原 -25 points failure 事件，无重复failure。其他 terminal 保留分项，但总候选NA，原因是其奖励语义未注册。

原数字来源由 cost 合同 SHA 5f77d8ad37d1d3edbf98b6e2401b8b95a595647cbea69d16a93713400316bbd7 锁定，其中：
- TensorTask SHA b8e73877eef14c47c7440560bc4333e68349d46d8b0a579202937b3e0ddbda8e。
- TensorEnv SHA 93ece5a2abc6adb7c1e90a230c0283cc5c93880ddbbaded3451661ef6e6ca645。
- surface exploration SHA 8f7b84e16d2fe102bc18412bfa9b024e8319dabbb4145c985049547c9a816bb4。

velocity最大绝对数值仍混合17DOF的 m/s 与 rad/s，不能标为纯 rad/s 或物理无量纲归一化。

本次唯一正演明确输入 **100 points/m**，即每1cm signed progress计1 point。新数字仅为算术候选，不是 rigid 迁移、调优结果或正式训练推荐。spec SHA af89da850200f06894a283b549548dd5c0afff7ac248e6182ee1fcdbed92e7a9。rigid approach 注册但未支付；rigid dex_reach 的圆柱几何、3/1 rate 和 .008m scale 不是 node-SDF progress 系数来源。

缺失不能填0。首窗上一动作未知，action-rate继续NA；第一窗及全回合完整总和均NA。实例失败后毒化；重复窗、跨epoch、terminal后再次结算拒绝。compose是算术接口，本身不将任意调用方数据认证为原生证据；历史适配器另行固定原文件及完整审计证明SHA。

## 历史正演

原件目录 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-approach-r0-20261114-jfd3i7wj/native-runs/flex.f7-approach-r0-native/attempt-wy27jh8g/native：
- physical.jsonl SHA 4e9b42b1fc96119888cce34d4977d2fd8720b4385409dda45a10f59fcbb8b83b。
- live_r0.jsonl SHA d2615f42439583fc90c10f9d937aeba652209828b77c94d14fd2510c511034a7。
- director_reset.json SHA c524f8c8a142da5e3909464146d991230ead8474aad428bb67f12a94552631ab。

复用 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/r0-three-source-replay-20261114-sE6xRE/result-final.json，SHA 70cb68f5382fdb015595e30d3a58191034ee751dee883c2900bdad0fd6ce0b31。该证明已逐字重算232行/29窗。验证过的同一bytes直接解析，无FK/SDF重跑或solver。完整physical先经现有cost evaluator及原_Sequence校验；与live R0按原auditor只对齐reset/begin/physical/end，setup/initial_geometry仍由原完整parser处理。

实际输出 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/approach-control-composer-20261114/candidate-100.json：
SHA f0349729d5e4cf7413d698fc8e47dd52694becb97a9fe8913527886416d9a66f；CLI exit0，native_imports=[]。

| 项目 | 已知数值 |
|---|---:|
| 原物理行 / 控制窗 | 232 / 29 |
| approach，232行 | +0.028162961825728416 points |
| time，232行 | -0.04833333333333334 points |
| action-rate，224行已知部分 | -0.00009621746570074074 points |
| velocity / joint event | 0 / 0 |
| 完整可算窗口 | 28 |
| 这28窗候选总和 | -0.01859693043286804 points |
| 全29窗完整候选 | NA，首窗action-rate未知 |

负分不是不能学习的证据，正分也不是抓取成功。历史mesh Reset同步风险、last_error缺失、contact/hold/slip缺口保留；training_reward始终None，full R0/native/Teacher/health/mesh-reset资格全false。本包不支付contact/lift/hold/slip/support/success。持续无接触时approach可产生进展信号，但不代表完整R0就绪。后继可复用已有applicability/necessary-hold/health separation：已知无必要接触时贡献可不适用、raw仍NA；接触未知不能如此处理；正常桌面approach支撑也不能全期当外援惩罚。

## CPU验证

Ubuntu原Python，两线程，CUDA不可见。首次27项缺模块全部失败；CLI/spec不可变性负例3失败/29通过。首次历史执行因physical独有setup/initial_geometry操作而拒绝；独立历史测试先复现1失败，再按原auditor范围修复。所有失败日志保留。

最终回归 **99 passed in 37.09s**，33新增、66相关原测试：

    cd /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approach-control-composer-20261114
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.:tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q tests/test_approach_control_composer.py tests/test_dh116_readonly_control_cost.py tests/test_dh116_deformable_control_step.py

实际CLI（使用全新输出路径）：

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' timeout 60s /mnt/mydisk/xk/envs/dh116/bin/python scripts/evaluate_dh116_approach_control_candidate.py --approach-points-per-m 100 --output /absolute/new/exclusive-output.json

CLI先拒绝已存在输出，以exclusive-create写文件。first-NA、逐行/终止错位、SHA、单位、额外failure组件、重复joint扣分、跨窗spec漂移、重复结算有负例。仅CPU夹具及旧账本算术；新native=0，optimizer=0，模型构造=0。
