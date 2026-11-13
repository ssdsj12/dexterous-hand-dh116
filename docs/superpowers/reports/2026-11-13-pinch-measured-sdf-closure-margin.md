# Pinch：实测末帧 SDF 与固定底座闭合余量

基线 c4d8d1c6dea95862ae51cda1f26f8ac5b7c854d0。新增独立 CPU 脚本、测试和报告；原 USD、SDF、材料、杆 seed、配置、原生入口、模型、旧账本均未修改。**0 native physics、0 GPU、0 训练更新；只得到固定底座左单手的离线接触发现候选，没有新原生配方或准入。**

## 结果

找到一个原 base 完全固定、原 48 quiet + 160 move + 30 hold 步数不变的独立初始 q / 闭合增量。两指速度分别 0.055666172460325514 / 0.07021898390297807 rad/s，低于原 0.3 rad/s。沿候选整条几何路径：

- 全 17 source + 5 locked cooked 桌面连续下界 **1.9564872089946053 mm**，比原单手静态 1.9 mm 门高 **56.487209 µm**。
- 所有非目标 source/cooked 对原杆垂向包络的连续下界最小 **+22.96778267935626 µm**，来自 source finger13。这是 CPU 几何预测下界，不是 live cooked 或实测接触证明。
- 新初态全部 22 个 shape 对杆包络最小下界 **+581.8036811421408 µm**。
- 用原 48 帧杆位置及实测手指帧作新初态反事实，5 个原 SDF 均无负 gap 且非零梯度的接触谓词；在域内最小 gap **+646.0323929786682 µm**。这不是新 reset/quiet 实测。
- 末态按原 208/238 实测姿态锚定的 FK 相对变化，两个必要 pad 的 SDF gap 均低于 −100 µm，方向相向。真实力、跟踪、接触、材料、自然退出和抓取仍未验证。

**1.9 mm 是本次固定底座单手的原预检；2.1 mm 是六轴策略 wrist projection 的独立原门。** 本候选仍低于 2.1 mm，policy_projection_qualified=false；不能接入该入口。1.95 mm 仅用于从已计算路径中选择保留 50 µm 静态余量的起点，没有替换或放宽任何门。

## 原生失败的具体距离区别

原始 v3 完成 238 测量行、0 手指 pair，所有 pair 为 table，原外层退出 1 的失败保持。仅重放 SHA 锁定的 plan、physics/pairs/table ledger、五个原 NPZ 和原实测 link/rod 数据。

| 历史帧 | pad | 连续杆段到 cooked hull gap | 最近杆节点球面到 hull 可行距离 gap | 原 SDF 节点球面 gap |
|---|---|---:|---:|---:|
| 208 | thumb | +94.782778 µm | +223.715683 µm | +230.889767 µm |
| 208 | index | +100.008716 µm | +300.965741 µm | +307.107810 µm |
| 238 | thumb | +4.707814 µm | +141.215942 µm | +146.933366 µm |
| 238 | index | +5.367791 µm | +206.135010 µm | +210.299157 µm |

因此“末帧只有约 5 µm 正 gap”并不是原生 solver 的接触阈值距离。杆有 41 节点，名义相邻间距 12.5 mm；这里最近节点为 thumb 35 / index 34。连续 capsule 可以在两个节点之间更接近手指，原 contact.cuh 则在每个节点球面判定 SDF。节点离散与连续段的差别在这两帧明显大于网格插值差别。

锁定五个 pad 的 spacing=1 mm、radius=5 mm、bake margin=12 mm、round=0、thickness=0，未重烘焙。margin 是网格覆盖域，非接触放宽。原 contact.cuh（SHA 8d90d9d18c853fbc3931a69435000726f49d36c887e62debe10225e8aa77d2bb）57–75 为 float32 三线性 phi 及插值梯度；99–106 在 object frame 取样，只有 phi−radius<0 且梯度范数>0 才接触，域外返回 false，没有额外距离 margin。1 mm 是采样间距，不能把它当 ±1 mm 的确定误差或直接当碰撞深度。

本包 CPU sampler 对照相同运算顺序、节点布局、域边界和梯度公式；不导入 ropesim/CUDA，不声称 CPU 与 GPU bitwise 一致。float64 对照在最终四个预测点的 gap 与 float32 相差最多约 0.0124 µm，远小于选定 100 µm SDF 余量，但这只是当前点的数值对照，不是未来 GPU 跟踪误差上界。原 SDF 与选定节点 hull 距离差为数 µm量级；后者为原 SLSQP 可行上界，未伪称精确最优值。两种几何均不能代替实际 solver pair 和力。

## 可执行的离线候选常量

原固定 base world WXYZ 完全不变：

~~~text
[0.49742945748902095, 0.24306903145211278, 0.4171107819860638,
 -0.5364008139784637, -0.3846972717135587, 0.7507884234984583, 0.024472822478934547]
~~~

active 顺序为 [finger11,finger12,finger21,finger31,finger41,finger51]。

~~~text
initial q =
[0.27005018256823987, 0.4638128961765604, 0.6709579060315968, 1.15, 1.15, 1.15]
end target q =
[0.27005018256823987, 0.48236828699666895, 0.6943642339992562, 1.15, 1.15, 1.15]
thumb/index closure delta = [0.018555390820108503, 0.023406327967659353] rad
thumb/index velocity = [0.055666172460325514, 0.07021898390297807] rad/s
canonical action24 indices 1,2 = [0.18555390820108503, 0.23406327967659352]
all other action24 channels = 0
physics_hz = 480; quiet_steps = 48; move_steps = 160; hold_steps = 30
~~~

所有 initial/end q 均经过原 active_limits/expand 的原 source/manual/mimic 限制，没有 clamp 或增大范围。初始 thumb spread、thumb、index q 与旧 seed 不同，ring/little/middle 和 base 未改变。独立 export/candidate.json 保存完整 q、base WXYZ、原 joint bounds、action24、速度、时序、原始路径 SHA、源/NPZ SHA 和全部预测 pose/gap 字段。它明确 old_v2_preflight_compatible=false、native_recipe_created=false。

| 锚定历史帧 | pad | 预测 SDF gap | 连续 capsule 几何 gap | SDF 梯度范数 |
|---|---|---:|---:|---:|
| 208 | thumb | −116.438139 µm | −136.046898 µm | 1.020623 |
| 208 | index | −149.749685 µm | −318.309276 µm | 0.999998 |
| 238 | thumb | −203.771982 µm | −225.179927 µm | 1.020771 |
| 238 | index | −245.839823 µm | −400.225703 µm | 0.999998 |

两帧的 SDF world normal dot 为 −0.774890 / −0.776233，仅为几何相向诊断，不是 effective opposed contact。负 gap 表示该几何距离定义下的径向重叠；连续 capsule gap 不是完整三维最小分离“穿透深度”。实际碰撞深度和力为未知，不能用这些值宣称力限已验证。本任务明确允许接触建立末态跨过旧 |gap|<0.05 mm 的静态 reach 条件；没有改旧 validator，也没有放宽非目标或初始碰撞保护。

预测变换为 T_actual(old frame) × inverse(T_sourceFK(old measured active q)) × T_sourceFK(candidate q)。原 208/238 的实测与原 FK pad 位置残差为 0.022–0.061 µm。这一锚定保留实测 frame，新的 q/初态/速度仍是离线假设；尤其不能假定新动作复现旧 tracking lag。最终缩短闭合增量、速度更低，但不据此声称动态跟踪保证。

## 有界过程及失败保留

1. 第一轮两变量 thumb/index closure，两尺度 0.001/0.0005 rad 中央差分，基线 1 + 差分 8 + 三个实测 CPU 端点，共 12。Jacobian 相对差 0.000355435；求得合法追加闭合，但完整 161 点路径未获非目标净空。
2. 单独 1 次原 QP 确认 finger13 对杆垂向包络的可行 gap=−370.298374 µm，故不是仅负下界造成的误判；该候选拒绝，run01 和独立拒绝文件保留。
3. 第二轮仅以失败候选为中心，联合初始 finger11 spread 与 thumb closure；index 增量沿用。两尺度、12 端点，信赖盒为 spread±0.05 rad、thumb 修正±0.02 rad，同时受原 limits/0.3 rad/s。Jacobian 相对差 0.000164931。选择回溯 0.75 后非目标路径净空 +22.968 µm，但初始 finger13 table 仅 1.568973 mm；run02 拒绝，未冒充合格。
4. 同一 run02 已计算的 161 点路径，关节直线 q(f)=q0+fΔ，固定 base。选择 index42 起的安全后缀，新的 q0=q(42/160)，保持相同末态，以原 160 步走更短的路径。几何轨迹是原轨迹精确子集；原每段 B/320 更保守的连续覆盖继续有效，不需重算或重新搜索端点。重新核新初态全部 22 shape 和 48 quiet 的反事实。先保存 index36 的 1.900907 mm 初步结果，再纯选择 index42 的 1.95 mm buffer；最终 run04 为本报告候选，run03 不覆盖。

共 24 个定向端点、322 个完整 FK 路径样本，两次小型线性求解；不是盲网格。两次后缀选择没有新端点，各重核 22 个初始 shape 和 quiet。四次主 CPU 命令 outer 分别 5.512058 / 5.561684 / 4.623540 / 5.268736 s，均 exit 0，总 20.966018 s；另有单独 finger13 QP 和离线算术/方向复核。所有 CPU 2 线程。没有自动继续优化。

保护数学复用原 path_displacement_bound，161 样本间任意点距最近样本的运动上界为 B/320。桌面最低顶点减此界；非目标对原 z∈[0.305,0.308] m 的完整垂向直杆包络，用已审查 XZ 支撑平面下界，扣 2 µm 后再扣 B/320。这里没有将 SLSQP 的可行上界用作净空下界。杆初始 axis z=0.308 m、radius 5 mm、table top=0.300 m 原样保留，未伪称 reset 时已桌面支撑。

## 与旧入口的边界

原 tabletop_pinch_initial_geometry_v2 validator 不硬锁某一 q seed，但要求真实 SDK 同钟完整 body names/paths/poses、source 与 stored-cooking 元数据和实际阈值，再做磁盘重算。CPU FK 不能代替该真实采集；原生 acquisition 原初始 non-overlap 检查仍须独立通过。12 个非 pad 只有原 source hull，实际 live cooked 仍未补齐。

旧 audit_dh116_pinch_initial_pose.evaluate_candidate 还要求两指相等的单 velocity_rad_s、delta≤0.025 rad、旧 |gap|<0.05 mm 及几何相向；旧 v2/v3 preflight 绑定旧 pose/action/source 身份。本候选虽最终两增量均小于 0.025，仍是不等速、新初态并有意进入 SDF 接触侧，不符合旧格式。后续若采用必须准备独立版本、冻结该候选及正常原生审批，不能拿它替换旧配置或绕过旧 gate。本包没有做那一步。

## 测试、审查和证据

新增 21 项 CPU 测试。导出功能加入前相关回归 **99 passed，0 failed，0 skipped，78.10 s**；加入导出拒绝负例后专用 **21 passed，0 failed，0 skipped，2.78 s**，合计覆盖 100 个不同测试。先后记录模块缺失及新方法缺失的失败，然后实现。覆盖三线性梯度/xyz顺序/域上界、零梯度拒绝、实测锚定、quiet/hold/速度、原 limits、margin 双指要求、独占输出、partial、真实父结果 SHA 拒绝、后缀连续记录/端点不变和门不能放宽。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q tests/test_dh116_pinch_closure_margin.py tests/test_dh116_tripod_joint_root_direction.py tests/test_dh116_power_closure_constraints.py tests/test_deformable_grasp_runtime.py tests/test_dh116_pinch_v3_evidence_replay.py tests/test_pinch_initial_geometry_v2.py
~~~

主 CLI 每次 --wall-seconds 50，外层 timeout --signal=TERM --kill-after=5s 60s；50 s 软计时不含 import/parser 和最终保存，不称硬 60 s 正常退出保证。默认为首方向；--refine-from、--crop-from、--export-from 严格接受本轮已冻结的相应父结果 SHA，非通用无限续搜接口。所有输出独占创建。

独立 camera agent 只读审查 sampler/contact、measured anchors、161 点覆盖、支撑平面及安全后缀；未见数学阻断。该审查不包含后增纯 JSON 导出，也没有代跑测试/native。

证据目录 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-closure-margin-20261113-_dmnkcda：

| 文件 | SHA256 |
|---|---|
| final script / export producer | 4671b391c0169ee24ae9df6a03c94396a7f22e6b4aa11d39fb13f4b9d9a78503 |
| run01/result.json（原 q 追加闭合拒绝） | c861cfeaccabd242d11af1e4179290ef1082fc87f93abd8bb71670d65f6b4aea |
| finger13-rejection.json | 8404a7af36da7246b124061a7d68c7f594fd48e79975e82d7621fe5c1fa490f6 |
| run02/result.json（原始完整路径） | 6778504d7485b8ba99f3629ad134f8f79840e2524612f9cc6659b0e6c2ae515f |
| run03/result.json（初步后缀） | 675ad8ab29f1a0163a42c9de3749d547ff2c7017749d607a311b0c49739d2b2f |
| run04/result.json（最终后缀） | f96b29a82e6b532e153f3b63e91a34a6303c7d3be9bff7b686eb3508b65ed6d0 |
| export/candidate.json | 3d04ccca63263b4480e1831693b8d7c70f1336170f2006b5c6a3a1f414553fe0 |
| selected-direction-audit.json | 397a3c96f1de219df4b9ba647ba0984f49bb88c65b5bb91cc0fcc6950f8897ed |
| offline-proof.json | 448630ca4f54c3b4b35dbeb4246201445aff1fc55753c7631412062b80389e1b |
| offline-verify.py | ac0f82108babd285ea04565f0a4a4ccfeb7e9297d6162cbd67d6d06ab0e77c7f |

offline proof 无新几何查询：核每次 executed-run 源快照、原 native/NPZ SHA、24 端点、322 path rows、4 个 Jacobian、最终119样本后缀及同末态、门值与全部 outer exit。run01/02 旧字段 unchanged_initial_q 是当时传入 path 函数的 q，不表示与历史原 seed 不变；最终候选已明确分出 original_initial_q / candidate_initial_q。原运行源快照与最终新增导出实现的 SHA 分开，不回填旧执行身份。根代理负责集成发布，本工作树不 push。
