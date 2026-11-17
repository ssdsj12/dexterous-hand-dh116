# Pinch 独立静态偏航：真实几何有界筛选

基线 2421f1e1ad0e08f0c89b5676923af49949c9810f。只在Ubuntu独立工作树新增CPU设计脚本和测试，不修改USD、物理、材料、旧动作路径、旧模型或原生配方。0物理步、0PPO。

## 实际实现与限制

- audit_dh116_pinch_static_yaw.py 在已完成trust-region候选上，左右手分别执行9个固定World-Z偏航，范围±0.03rad。绕每手原base原点前乘旋转，保持平移、主动q、原两指delta和被动联动不变，不是修改腕部动作幅度。WXYZ/XYZW转换有独立测试。
- 读取左右各自USD源碰撞几何、实际cooked指腹和锁定SDF，以及原完整原生账本节点；每个候选分别核对初态/端点，选择后保留33点完整路径及连续位移下界。共102个完整几何状态，硬上限102，墙钟上限45秒；无自适应追加候选或原生运行。
- 目标仍为两指各自-0.1至-0.3mm SDF设计接近、原桌面runtime floor、非目标形体安全。SDF预测不是实际接触或载荷。全局不可行未被证明。

## 实测离线结果

|手|选择偏航rad|拇指SDF间隙mm|食指SDF间隙mm|连续桌面下界mm|连续非目标下界mm|合格新候选|
|---|---:|---:|---:|---:|---:|---|
|left|0.0000|0.269220676|-0.106180552|2.437380308|0.155903199|否|
|right|0.0000|0.438481569|-0.287448056|2.746973141|0.197699707|否|

左右都保留0偏航。所有非零候选至少违反一个非目标设计约束，不能替换原候选。

左手负偏航主要受source finger13，正偏航受source finger22约束；右手相反。这里只能说保守分离下界不足，**不能把负下界当作已证实穿透**；下一步需对固定候选做可行最近点上界/碰撞见证，再决定联动方向。没有放宽设计阈值来选中候选。

World-Z前乘对所有形体初始桌面高度的最大数值差左右均约1.11e-16m，符合旋转几何；没有靠降低桌面或移动杆来增加净空。

### 选中候选每个指腹初态桌面间隙（毫米）

|手/指腹|source mesh|cooked hull|
|---|---:|---:|
|left/finger14_link|3.716206170|3.782167445|
|left/finger23_link|3.162695380|3.578143701|
|left/finger33_link|39.238124569|39.377999045|
|left/finger43_link|56.729158813|56.874298525|
|left/finger53_link|74.194224328|74.327650926|
|right/finger14_link|4.260852890|4.696966440|
|right/finger23_link|4.082831451|4.082831451|
|right/finger33_link|21.805028472|21.842661501|
|right/finger43_link|25.878584779|25.878584779|
|right/finger53_link|55.492197368|55.725384140|

## CPU测试和失败证据

先写测试，缺失模块的失败保存在 pinch-yaw-red-7hQMMZ。17项新单元测试先通过，再执行一次真实几何离线筛选；补充102状态硬预算/异常保留测试后，新旧回归58项通过、0失败。4条既有HiGHS threads选项透传提示不计为失败。未重复执行102状态筛选。

运行命令：DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh scripts/audit_dh116_pinch_static_yaw.py --output /home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-static-yaw-20261113-sqwO6J/run01
结果：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-static-yaw-20261113-sqwO6J/run01/result.json；SHA256 d434fbd069325dbaf71195a0452717667a43be9c043e7e57601b77ffc1407890；耗时 4.540109秒。
回归：tests/test_dh116_pinch_static_yaw.py tests/test_dh116_pinch_trust_region.py tests/test_dh116_pinch_spread_axial_direction.py；日志 /home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-yaw-regression-ih4sSz/pytest.log；SHA256 875f72b2c93540d0a4358d855ba70d5cc7fc56318ed35a0a9203e06a8e69ec1f。

## 准入与下一工作包

离线筛选完成；新初态/原生接触/Teacher晋级仍blocked。该失败只阻断此候选原生执行，不影响已通过的参数/生命周期源审计或其他抓法准备。继续针对候选的非目标shape真实距离见证，并联动初态与闭合量。未经几何/完整物理/Reset/R0资格，不启动Teacher PPO。PR继续Draft。

## 源文件SHA

- scripts/audit_dh116_pinch_static_yaw.py: 22ef838b07fa1b480505bbff8c9589cf7abf79dc0ea20ad47603ff7253b9d0af
- tests/test_dh116_pinch_static_yaw.py: 2c36a5027d3dd4b133aa01beeff40655610674d7e546b92d2743395f01162ee9
