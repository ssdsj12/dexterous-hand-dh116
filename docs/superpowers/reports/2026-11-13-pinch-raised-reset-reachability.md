# Pinch 初始掌位Z与两指闭合的定向可达性诊断

本轮没有找到同时保留原保护且达到双pad node-SDF接触的候选。左侧局部模型仍被thumb近端余量约束；右侧抬高2mm能增加几何净空，但候选node-SDF依旧为正。结果只拒绝本轮两条固定根线性q路径，不证明全局不可达，也不把支撑下界失败说成已证明碰撞。

基线 a4a5af8b2328c1290ef8511adf3146504f73ccbc；新隔离worktree/branch独立提交。只新增脚本、测试、报告，integration、原paired proposal、冻结profile和native候选不改，未push。0 native/SDK步，0训练更新。

## 来源与实际约束

- 原左右独立paired结果：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/paired-pinch-reset-BaP0hx/result.json，SHA 996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37。
- 原reset seed SHA：647247fe31219f04337b6f6e29ddefaa2c81bcee27dbed8dd3d6f962e406230d。
- 原已通过dual v4零动作账本SHA：ead0dbf2c15b83864a1c440848ed9cf159b86c88a6ade0e4275208efebcf778c；通过原inspect_interval重放。
- 固定杆节点直接取该账本SDK10的41个实际节点；未替换成新高度或绑杆。原半径.005m、自由节点、无固定root orientation及candidate重力[0,0,-9.81]保持。
- 父native结果SHA：0c2d2458b133d704b0ef2a3a86651606e152bf21636b54e4fae32cf8caff268c；receipt artifact摘要匹配后读取实际SDK gravity=[0,0,-9.8100004196167]。SDK读回与DeformX candidate配置重力分列，不把旧training task未知物理参数变成已测量值。
- paired文件的rod_seed仍保留历史source=explicit_supported_seed_proposal_not_installed_reset等未知/未安装标记。本包不改该原件；当前使用的是之后v4实际消费该proposal所得的节点，不宣称完成长期task/R0 reset绑定。
- 每侧独立原USD、17 source+5 locked cooked和五个1mm SDF的SHA保存在result.sources。没有镜像左手，也没有改材料/关节/速度常量。
- 冻结readback profile SHA 67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498 核对未改。

这仍是固定实际节点的CPU几何问题：未模拟自由杆后续受重力、接触和变形的响应，不能据此承诺未来支撑、稳定或有效对向接触。

## 明确的小范围与预算

每侧从原paired近接触末态做三变量局部诊断：静态初始掌位Z增加0..2mm；thumb/index末态各独立改变，初态q不变。thumb-spread、其余手指、根XY和旋转不变。总正闭合增量仍<=.1rad、时长160/480s、速度<=.3rad/s，原coupled limits继续检查。

一次baseline加两尺度导数，共13个状态：rootZ用正向二阶差分，避免发出降低初态的候选；两指用中心差分。步长为归一化.01/.005，scales=[.002m,.05rad,.05rad]。每侧只解一次有界LP，再对唯一结果检查17点路径。共 **60个完整几何状态**，未动用剩余4个额度。内层50s软限、外层60s硬限；实际脚本4.524333344s、外层7.580199756s、exit0。

LP拟合的是**末态**约束，诊断设计目标为桌面2.4mm、非目标下界0.2mm、目标node-SDF -0.1..-0.3mm，并最小化共同归一化缺口加小坐标惩罚。这些是寻找余量的目标，未替换原2.1mm/非目标>=0物理保护。初态和整条路径仍独立用原门检查，LP不是全轨迹优化器。末态负SDF只能表示CPU插值接触谓词，不能表示原生力/有效抓持。

17点连续下界采用原每link最大点位移B扣B/32。非目标支撑面同时覆盖全部41实测节点及其折线。有限采样/支撑方向可能使下界过保守，未满足下界不等于出现真实碰撞。

## 左侧：近端余量先成为局部约束

两尺度Jacobian相对差0.000991316，LP选择Z增量0，不代表全局抬高无益。候选：
- 原root不变；
- thumb总增量0.0264430802rad，速度0.0793292405rad/s；
- index总增量0.0275889365rad，速度0.0827668095rad/s。

末态node-SDF thumb/index分别 **+76.965429/+75.899530μm**，两者仍不接触。source finger13的末态支撑下界仅 **+23.755966μm**，诊断0.2mm余量目标未满足。

| 路径量 | 数值 |
|---|---:|
| 实际采样最小桌面间隙 | 2.110896810mm（起点source finger22） |
| 连续桌面下界 | 2.002723064mm |
| 实际采样最小非目标支撑下界 | +23.755966μm（末态source finger13） |
| 连续非目标下界 | -68.838850μm |
| 初态所有22形状的杆支撑下界最小值 | +1.442380000mm |

因此当前有限样本证明不足，**没有负的实际可行距离上界碰撞证据**。早期table连续界在fraction0..1/16及1/16..2/16不通过，主要因初态只有10.9μm余量，而B/32扣减更大。不能据此声称真实桌面穿透。

baseline局部SDF导数（m/m或m/rad）：
- thumb: dgap/dZ=+0.168360，dgap/dq_thumb=-0.0589611；
- index: dgap/dZ=+0.423589，dgap/dq_index=-0.0677677；
- source finger13支撑gap: dgap/dZ约0，dgap/dq_thumb=-0.0790037。

这说明在当前局部姿态里，抬高会让目标节点更远，同时没有改善选中finger13支撑方向的近端余量。用baseline导数对候选末态作条件线性外推：thumb到SDF零约还需0.00130536rad，而23.76μm近端支撑余量约在0.000300694rad时消耗。此为局部诊断，未新增几何查询，也不是严格未来碰撞时间证明。仅改变approach起点或跳过早期路径并不能改变这个末态tradeoff。

## 右侧：抬高改善几何，但SDF接触仍未到达

两尺度Jacobian相对差0.000197343，LP取允许上界Z+2mm：
- 新rootZ=0.3921748744972687m，其余root分量保持原右手值；
- thumb总增量0.0258083904rad，速度0.0774251711rad/s；
- index总增量0.0326607826rad，速度0.0979823478rad/s。

| 路径量 | 数值 |
|---|---:|
| 初始/采样最小桌面间隙 | 4.116394609mm |
| 连续桌面下界 | 4.026022194mm |
| 连续非目标下界 | +111.807138μm |
| 末态source finger13支撑下界 | +202.179553μm |
| 末态thumb/index node-SDF | +1.401092391 / +0.504780561mm |

原baseline长路径末态thumb/index为+1.113605686/+0.252943952mm；这条有限方向未改善到接触，thumb甚至更远。局部导数稳定不等于较大步长预测有效。baseline thumb dgap/dZ=+0.0581611、dgap/dq=-0.0554258；index对应+0.461168/-0.0891527。最小gap节点仍是thumb4/index3，不能将非线性误差唯一归因于最近节点切换。

source finger13支撑gap导数对Z为+0.0980171、对thumb q为-0.0464399。当前末态余量的条件线性外推只容许约0.00435357rad进一步thumb闭合，而到thumb节点SDF零约需0.0252787rad。右侧几何清只是本固定节点模型条件下的静态结果，不是实际接触或全局不可达证明。

## 具体下一假设与停止点

本轮到此停止，不继续调Z/闭合量或扩大网格。两个候选都没有生成reset proposal。

下一次若获准继续CPU设计，应把**初态table余量**直接放入求解，并增加能改变proximal/distal相对杆关系的受限变量，例如独立thumb-spread或掌位轴向站位；保留端点实际node-SDF、全非目标保护及原2.1mm。仅从当前近端余量继续加闭合或只更换路径起点，没有本轮证据支持。是否选这些额外变量需要下一独立小预算，尚未执行。

## 验证和不可覆盖原件

先写测试，得到预期缺失模块错误。最终 **15新+40相关=55 CPU tests passed，3.08s**。两条SciPy warning表示threads=2被转发给HiGHS；没有失败。测试覆盖原速度/关节/静态Z边界、64状态cap、一次两尺度模型、正向root探针、失稳拒绝、17点完整性、timeout保存partial、身份变化拒绝及exclusive output。

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q tests/test_dh116_pinch_raised_reset_direction.py tests/test_dh116_pinch_closure_margin.py tests/test_dh116_dual_pinch_readback_profile.py
~~~

实际geometry只运行一次。offline-verify只重放60条已存状态的Jacobian/LP及下界代数、核七实现SHA/十NPZ SHA/父native重力读回，不新增FK/SDF状态。未把离线复算称独立物理验证。本包未获得新的独立代码短审，不计入已审。

证据目录：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-raised-reset-direction-20261113-fjmj4jye

| 文件 | SHA256 |
|---|---|
| run01/result.json | 59ddbd86359758e6102da7e550d397b1e35f11f9b9aaa66bacf71a418a332634 |
| outer.json | b01d5980317ac4d954bfc830f8e707a3510d7e7f2aca3751a13685d7649c7886 |
| tests.log | 97c5ebe37c70bcfc77ed970b4840ec8ea22bc53c5493c5d8b52271a285f63904 |
| offline-verify.py | de6e35e08c72db6a6ef7a589a162c6c6b08533defbee299cfe69c5d9dd15791d |
| offline-proof.json | 4f1982ec2494d75b12e47af838033d66cf759b9ac53ac26b60e14034efbd80f4 |

执行脚本SHA：d16964b2e0718f3467d1433c6732655ed23e1b7df58ab4699519ca4dcd1a72de。完整左右pose/q/delta、44个约束、两尺度Jacobian、17点路径、来源摘要均在result中，没有只保存“最好值”或丢弃失败候选。
