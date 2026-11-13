# 右 Pinch 定向微调：严格静态路径通过，reset 接入仍未完成

基线 9bfeee711a0bfa3f02532774862c0ab424ab1bbd。仅新增脚本、CPU 测试、本报告，不改此前工作包、原阈值、保护配置、资产、旧 reset 或物理参数。新 native physics steps=0、optimizer updates=0，没有 Isaac/GPU/solver/policy 执行或 push。

本轮 **120/120 局部候选完成、0 数值错误、78 个严格端点门通过，只做 1 条完整路径检查即通过**。选中 right-targeted-092。此为原 source/cooked 几何和连续包络门下的静态 proposal，native_ready=false、teacher_admitted=false、effective_opposed_contact=null。全部 17 live cooked 尚未知（12 non-pad 只有 source USD envelope），不把 finger13 的 source 几何界说成实测 native contact。

## 不扩大搜索与输入绑定

唯一 parent 为上一轮 pinch-right-derived_right-r10-p10-b0.025-x0，既有 result SHA256：
1a271578961192096d3d4aa5ec03785807c3940908bd9444fd0c285be86dff1f。
它的 finger13 连续 source 间隙下界为 −36.688362 µm。读取时强制该 SHA、唯一 parent/path 身份、原严格端点结果、原失败边界和所有 right geometry source hashes；没有重扫 450，也没有镜像左侧。该 parent 的 derived-right 配置仍非已验证成功的原生 archive。

120 个变体固定为：

- root X：[0,25,50,75,100] µm；
- root Z：[0,−25,25] µm；
- thumb seed：[0,−.00025,.00025,.0005] rad；
- index seed：[0,.00025] rad。

root Y 和 WXYZ 完全不变。闭合仍复用原 9-angle measure_closure_grid；严格追加原端点 |gap|<0.05 mm，方向 dot<−0.2；完整路径复用 verify_candidate_path 的 161 点、原 lever-arm 间隔界与 2 µm QP 余量，桌面下界仍≥1.9 mm，所有非目标 source/cooked rod 下界仍≥0。最多检查 3 条完整路径，首次通过即停止，没有降低门槛。

## 选中 proposal 与余量

相对 parent，right-targeted-092 只改变 root X **+75 µm**、root Z **+25 µm**、thumb seed **+0.00025 rad**；index seed 不变。原 active order 为 finger11/finger12/finger21/finger31/finger41/finger51。闭合 delta 仍为 [0,.0125,.025,0,0,0] rad，160/480 s，速度 [0,.0375,.075,0,0,0] rad/s，低于原 .3 rad/s 上限；原 expand() 检查主动/被动 mimic 限位。

| 检查 | 实际值 | 原门槛 |
| --- | ---: | ---: |
| thumb cooked 端点 gap | +46.480444 µm | 绝对值 <50 µm |
| index cooked 端点 gap | −6.306120 µm | 绝对值 <50 µm |
| thumb/index 几何方向 dot | −0.829692 | <−0.2 |
| 初始全部 source/cooked 桌面净空 | 2.043852 mm | ≥1.9 mm |
| 连续 source/cooked 桌面最低界 | 2.039475 mm | ≥1.9 mm |
| finger13 source 端点 gap | +45.083308 µm | 排序参考，完整路径另验 |
| finger13 连续 source 最低界 | **+38.706223 µm** | ≥0 |
| 其它非目标 source/cooked 连续界 | 全部≥0 | ≥0 |

初始所有 source/cooked 对杆间隙非负。thumb 端点仅有约 3.52 µm 门槛余量；这是特定原几何与标定假设下的细小静态余量，不代表动力学鲁棒性。source 指原真实 USD collision envelope，五个 pad 使用锁定 F4 live cooked 几何；缺失的其它 live cooked 不补零、不外推为已证实安全。以上几何通过不证明 force、实际 opposed contact、真实 table support、R0 reward 或 grasp success。

## 杆的初始支撑与 R0/reset 一致性

已明确检查：旧 pose_reference 杆轴 z=.308 m，候选半径 .005 m，桌面 .300 m，底面有 **3 mm 空隙**。原 FreeRodRuntime 把构造 nodes 交给 RopeBatch.from_positions；同一 rod 的 reset_envs() 恢复构造 seed，双手 reset transaction 不会自动将杆落桌。physics agent 独立只读核对确认这一结论。

本轮单独提出 supported seed：保留原节点 X/Y、数量和顺序，全部轴高显式设为 **.305 m**，nominal bottom gap=0。整个 capsule 的 XY 范围在原有限桌面内；pinned_node_ids=[]、fix_root_orientation=false，gravity 沿用显式 candidate [0,0,−9.81]。float32 constructor roundtrip 最大偏差 7.152558 nm，未运行或读取 native 质量/接触，因此 native_table_support_verified=false。

该 proposal **不匹配现有构造/reset 的 .308 m seed**，current_registered_reset_seed_matches=false。原 R0/task 中 radius/gravity/material 等 14 个 runtime 字段、Pinch native_reset_source_sha256 和 R0 coefficient_source 仍为 null；没有填默认、冒充批准或修改配置。离线复核还调用原 load_flexible_grasp_task 验证整个当前 task/source contract。未来外层必须显式采用相同 proposal nodes、绑定新 reset reference 和 R0/source 身份，再做独立原生验证；当前不能声称 reset/R0 已对齐。

## 预算、证据与可重放性

固定最多 120 候选、最多 3 条完整路径，2 CPU 线程，空 CUDA_VISIBLE_DEVICES。脚本软期限 50 s、外层 timeout 60 s（TERM 后 5 s kill grace）。实际 **19.819346 s 脚本、20.579489 s 外层、退出码 0**，只检查 1 条完整路径；未再扩大。progress 121 条完整记录为 120 rows + 1 path；remaining list 为空。已有原子 deadline/journal 提交逻辑复用，超时保留完整结果和未完成身份，不把 partial 当通过。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
timeout --signal=TERM --kill-after=5s 60s bash scripts/python.sh \
  scripts/refine_dh116_right_pinch_geometry.py \
  --output /absolute/new/private/run01 --wall-seconds 50
~~~

实际目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/right-pinch-targeted-cpu-20261113-TfEOGC

- run01/result.json：3590402b7405fed02b49c9db2ffcf7f47934c34543dd8b2065525f986d8c7bc7
- run01/progress.jsonl：954f75ca723df5d720881d93221a4c9142cc401f6c48320b6b68ff8ff8260e72
- outer.json：1601f572f298769dcca35226d711e34575b8cb1c69af4856bc8bbdcf2ca5c2cd
- selected-proposal.json：02b79aaa83f2c21f58a16e377cf67bd273511aae4b26714ab46fc30df1fa4e43
- offline-proof-v2.json：d50a77b0757501c69d56c9c5f6b8e15506bee4e96b42d1f604f8e481e50144fc
- executed/final script SHA：bc80278127ebbc8c69a0bd70a7f011e1dd1307b29b032fb7eee612c49de72326

proposal 含精确 root WXYZ、q、delta、path metrics、完整原/新/float32 rod nodes 和 source SHA，保持原生资格 false。offline proof 只重建 120 个候选参数、核对 journal/result/source/原 task loader 以及原阈值算术，没有重算几何。首次 proof 写入遇 NumPy bool_ JSON 类型错误，残件 offline-proof.json 保留，不作有效证明；v2 proof 独占写入且完整复核。

## CPU 回归与交付限制

**14 项新增、相关 25 passed / 0.79s，退出码 0**，git diff --check 通过。先观察缺模块失败；测试覆盖固定 120、独立右侧、root/q 小范围、parent SHA 拒绝、seed 不改、supported proposal 显式不匹配旧 reset、pinning/table 错配拒绝、0.5 mm 粗门不能充当 0.05 mm 严格门、完整路径失败/缺失不准入、最多 3 条路径及首次通过停止。

~~~sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_right_pinch_targeted.py tests/test_dh116_right_pinch_closure_screen.py
~~~

交付的是一个通过上述严格静态 gate 的右手 proposal，加一个明确尚未接入的 free/supported rod seed。没有提交原生 recipe、没有授予 R0/reset/Teacher 资格，也没有申请或消耗新的原生/训练预算。
