# Tripod proximal blockers：120 个独立 CPU 定向候选

基线 f21b428ffa6684323479bd97771a411867e63540。本包仅新增独立脚本、CPU 测试和本报告。原配置、关节限位、USD、材料、solver、R0/task、原始证据和冻结入口未修改；没有 Isaac、GPU、原生 physics tick、policy 或 optimizer 更新，没有生成 reset proposal，也没有 native/Teacher 资格。

## 结果与准确范围

实际完成 **120/120（左 60、右 60）**，0 errors、0 remaining；脚本 10.045138 s，外层 13.348017 s，exit 0。候选不重复上一轮 450 搜索；只保留各侧零变化父候选作为数值复核。两者原 active q、delta、root 和端点误差复现到 1e-12。这里的失败只约束固定直杆参考下的这 120 个局部变体，不能推出自由杆物理不可达。

| 指标 | left | right |
|---|---:|---:|
| 原 Tripod 三 pad / 0.5 mm / 相向 / 端点桌面粗门通过 | 2/60 | 19/60 |
| 指定 proximal 形状端点不相交 | 12/60 | 0/60 |
| 两项同时通过 | 0/60 | 0/60 |
| 完整 161 状态路径检查 | 0 | 0 |
| 可形成静态双手配对 | false | false |

所有 21 个粗门通过候选的指定 proximal 端点已相交，足以否定其整条连续净空路径。其余候选本身未通过原粗门。因此没有消耗完整路径检查；连续路径余量保留 unknown，没有用端点净空替代连续门。脚本仅对粗门和指定 proximal 端点都通过者最多检查 2 条/侧完整路径；只有双侧完整路径通过、world-y 分离下界正值且任务完整完成时，才可生成独立静态 proposal。

最佳仍满足原粗门的候选：

| side / ID | 指定 source proximal gap (mm) | required 最大绝对 gap (mm) | **端点** table gap (mm) | 对 2.1 mm 的**端点**余量 (mm) |
|---|---:|---:|---:|---:|
| left / tripod-proximal-left-08 | finger22 −2.121312 | 0.212412 | 2.489238 | +0.389238 |
| right / tripod-proximal-right-22 | finger13 −0.281315 | 0.170509 | 2.108633 | +0.008633 |

left-08 相对自身父候选只增 world-roll 4°、原 thumb spread 初态 +0.04 rad，root 沿退出方向位移 0；三 digit 正向闭合 delta 为 [0,0.025,0.075,0.0625,0,0] rad。thumb 对 index/middle 的几何方向 dot 为 −0.277392 / −0.828360，方向问题改善，但 finger22 仍相交。左侧最大的 finger22 端点正间隙为 +0.516602 mm（left-52），却使 required 最大 gap 增至 3.570872 mm，两个方向 dot 都为正。此范围内“退出近端形状”与“到达、相向”没有同时满足。

right-22 相对自身父候选沿退出方向 +0.25 mm、world-roll +4°、thumb spread 初态 −0.02 rad；delta 为 [0,0.025,0.0625,0.0625,0,0] rad。方向 dot −0.862464 / −0.829753，但 finger13 仍有 0.281315 mm 缺口。其他右候选的指定 proximal gap 更差。

**1.9 mm 和 2.1 mm 未混用。** 原静态路径 helper 的 1.9 mm 门完全保留；2.1 mm 从现有 tabletop_deformable_wrist_projection.CLEARANCE_M 读取并 SHA 绑定。以上正余量仅属端点，不代表 reset、全路径或 runtime 入口通过。另一个已提交右 Pinch targeted-092 的源连续净空为 2.039475 mm，因此仍低于既有 2.1 mm runtime floor 约 0.060525 mm；本包未重扫该 Pinch、未提高或降低任何门槛。

## 真值来源和定向范围

父证据是原 Tripod/Power 实测 final-preparation.json，SHA e6335c291c48ff40d1f19eb114a1f074594b2b7981e749b0f86335929babe4a6。输入锁定原独立 Tripod seed c36e8a06c9ff227f7cfb32477dd83de3c4ab425092e1397894445d47a5f1053b、原 helper SHA、每侧 USD/FK/cooked source map；任何不符在搜索前拒绝。没有使用 Pinch seed、镜像或 rigid success 标签。

父候选 left tripod-left-r20-p30-b0-x0.002 的连续 source blockers 不止 finger22 −3.013530 mm，还含 finger13 −1.043589 mm、finger32 −0.413690 mm；右 tripod-right-r0-p15-b0-x-0.002 主要为 finger13 −0.563260 mm。对固定父端点只读测量原 source convex 最近点，退出方向约为左 (−0.631878,0,+0.775068)、右 (+1,0,0)。去除数值噪声的 world-y 分量后归一化，沿它平移 root；并在既有杆参考 pivot 周围局部 world-roll、改变各侧原 thumb spread 初态。其余五个 active 初态不变。

固定网格：

- left：退出距离 [0,1,2,3,4] mm × roll [0,−4,+4,+8]° × thumb spread [0,−0.04,+0.04] rad。
- right：退出距离 [0,0.25,0.5,0.75,1] mm × roll [0,−2,+2,+4]° × thumb spread [0,−0.02,+0.02] rad。
- 每侧 60，总量严格 120，左右交错执行以均匀消费时间预算；不是 120/侧。q 使用原 expand/active_limits 检查。
- 三个必要 digit 仍各测原 9 闭合角，最多 +0.1 rad，160/480 s，0.3 rad/s；没有放宽原 0.5 mm endpoint 或 dot < −0.2。没有套用 Pinch 两 digit 或其 0.05 mm 独立门。
- CPU convex-QP 只算给定形状距离，不优化候选姿态/关节/模型。17 source envelopes + 5 F4 locked cooked pads；12 非 pad 的实际 live cooked 仍 unknown。没有真实 contact normal、reaction、材料/驱动/自碰撞/动态成功声明。

原 rod 节点 axis z=0.308 m，radius=0.005 m，table top=0.300 m，原 reset 底部间隙 3 mm。沿用旧几何审计的 z∈[0.305,0.308] 包络，闭合端点针对 z=0.305 m 测距离；这是几何参考，不是 native settling、桌面支撑或 R0 reset 已对齐。原 FreeRodRuntime 构造/registered reset seed 未改，本包没有把支撑姿态偷偷装入运行时。自由杆实际响应需要独立来源，当前不发原生申请。

## 测试、预算和可重放证据

先新增测试观察模块缺失失败，再实现。当前新增 **16** 项；相关回归 **48 passed，0 failed，0 skipped，5.46 s，exit 0**：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q tests/test_dh116_tripod_proximal_screen.py tests/test_dh116_right_pinch_targeted.py tests/test_dh116_right_pinch_closure_screen.py tests/test_dh116_flexible_style_pose_preparation.py
~~~

覆盖 120 总量与各侧独立 lineage、tampered receipt、原三 digit 门、缺侧/失败/重叠不得配对、1.9/2.1 区别、负端点不误进路径、独占输出、预算拒绝，以及超时保留已完成 2 行与精确 118 remaining IDs。实际 120 个候选只运行一次：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=5s 60s bash scripts/python.sh scripts/screen_dh116_tripod_proximal.py --wall-seconds 50 --output /absolute/new/private/run01
~~~

50 s 是从脚本 setup 开始的软截止；支持完整 partial 落盘。60 s 是外层 TERM 限制，另有 5 s 清理宽限；未声称硬性 60 s 正常结束保证。本次实际仅 13.35 s，完整落盘且 exit 0。所有源和原始结果未覆盖。输出保存完整 120 个候选参数/曲线/端点、父最接近方向、SHA、完成/未完成身份和逐行 ledger。

私有目录 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-proximal-cpu-20261113-5t2r_0kq。

| 证据 | SHA256 |
|---|---|
| executed / final source | cce50c1ee2271698f0515ab4f919cd380e51bb334643ed69ef7469b3eb130f58 |
| run01/result.json | 29c4ff4fb3a333f7e06af04df7edf8598a691fd8d27d7c8fb9270d01d575c58e |
| run01/progress.jsonl | 31950de0099d3a656b339f053eb118b483fda237ccf36feb1fbf5f552b67d858 |
| outer.json | 28bee36e79076ea150b0ec160adadb5976118766354cea74d130dfcec4bd620a |
| offline-proof.json | 945223784386a9eb41d07436a5c963ae4914e6fad0e547918f614d35fad04689 |
| offline-verify.py | 0aa116ad9ea7e48cf0336b72663cd631f4b6bdadc666ae3d2ffb71cc198774ca |

离线 proof 未调用新的 FK/convex 距离，只验证运行源码 SHA、原父 SHA、重建全部参数、120 ledger 行精确等于结果、两侧零变体复现及统计/外层退出码。它保留原结果不改。若继续 Tripod，需针对多个 proximal 形状与三 pad 方向/到达的耦合约束提出新的有据方案；不能直接追加此网格或由粗门成功启动 native/Teacher。本包由 root 集成发布，不自行 push。
