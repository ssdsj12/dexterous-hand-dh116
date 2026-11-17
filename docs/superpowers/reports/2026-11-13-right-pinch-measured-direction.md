# 实测右 Pinch 有界闭合方向：非目标源凸包阻断（2026-11-13）

基线68983aacee09b80b10f87085f7ac1034000866fb。复用SDK234实测右手，不重新运行Isaac。新物理步0，PPO0，Teacher准入0/3。

## 实际代码

新增 scripts/audit_dh116_right_pinch_measured_direction.py：load_anchor、direction、refine_direction、predicted_poses、failed_path_choice、gates、run。
- 同时锁定原failed receipt与独立成功复审SHA，逐artifact核验，读取实际右手17DOF q/qd、23link位姿及同钟41节点。左右身份不混用。
- 仅用右USD/FK/SDF。实际位姿锚定 T_actual × inverse(FK(q_actual)) × FK(q_candidate)。保留全部23link记录，源碰撞路径只使用其17个实际对应link；额外wrist_anchor不能当指关节FK。
- 原0.3rad/s、160/480秒上限、主动/被动限制不变，不clip越界方向。两尺度有限差分核验后解一次2维方向，三个非线性回溯点全部保存。
- 一个明确SHA的失败可单独refine一次；累计位移仍<=0.1rad，不重置原动作预算。另一个明确SHA失败可仅执行161点路径审核，不再次优化。
- 所有17source+5cooked形状检查原2.1mm桌面门、非目标杆距离；用原B/320覆盖样本间点运动。旧XZ负界才使用全3D逐段证明，半径和2µm裕量不变。
- 指腹距离由实际锁定1mm SDF三线性读取；负gap和法向仅几何预测。没有新原生配方。

## 实际原生来源（历史，只读）

source SDK234，env0，episode dual-pinch-a3c7e980097149789f15b74830bccac0:one，reset1，rod_tick2784，timestamp0.4875。
原physical SHA170b9bc8090eeb3f4291ce4102335eabbcb1d49892e333d1aa5b4ac4d9585fdf。
独立复审SHA57cbc1b6b97eb594368e4702746241c3be3a3b0c562d99293ab2b470b30706cf。
原receipt SHA97f83e1e001ea28cfa08ebcf4017d717f7adcc11aad38402280e780f48311746仍failed。

右手动作全零、实际q与SDK目标近似相等；指腹无实际杆接触。
q主动顺序finger11,finger12,finger21,finger31,finger41,finger51：
[0.4800281823,0.0376150049,0.6821601987,0.2752428651,0.5636945367,0.2510532439]。
拇指/食指实际帧SDF gap1.871493/2.451136mm，最低桌面source finger13为2.116345mm。只比原门多16.345µm。

## 三个独立CPU结果

1. 首方向9个差分点+3个非线性点，总12。Jacobian相对两尺度差0.000255764。
预测delta=[0.032052242,0.029396052]rad，完整FK后gap=[+0.392651,-0.143053]mm；拇指没有接触。
结果 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-pinch-measured-v2-20261113-ff59_aym/run01/result.json
SHA de54cdf504215ac568950a938408e58233c05f02408fb48460734fac5b5d06b6，5.407103s，exit1。

2. 对上述实际计算点重新局部线性化，另12点；Jacobian相对差0.000502127。
累计delta=[0.056519153,0.029473603]rad，完整FK gap=[+0.702374,-0.149995]mm。拇指继续闭合反而变远；同一初始线性Jacobian不能跨SDF/几何曲率区间。
结果 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-pinch-refine-20261113-4_e48g_c/run01/result.json
SHA ca22f02d999a520b640ed6c261c8456c20f1a7b14348ae8a517d8a946f5d39cb，5.616475s，exit1。
没有进行第三轮方向搜索。

3. 从第二轮三个已计算回溯点中选择残差最低者，仅检查完整路径，新endpoint优化0、路径161。
delta=[0.042389365,0.022105202]rad，速度[0.127168095,0.066315606]rad/s，24D索引7/8，其余0。不是已执行动作。
结果 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-pinch-failed-path-20261113-0z8_jdc8/run01/result.json
SHA2894befba70c78b506178c3a0b3bd95086d40eaf1d6fe951b77c4b5f38e48735，7.251553s，程序exit0表示诊断完成，static_path_passed=false。

| 原约束 | 结果 |
|---|---|
| 连续桌面最低界 | 2.101502mm，通过2.1mm |
| 非目标杆距离最低下界 | -0.834005mm，source finger13，index157 |
| 首个非目标无法证明净空点 | index34，-0.004401mm |
| 指腹全路径最低SDF gap | +0.338897mm，未形成接触 |
| 末态拇指/食指gap | +0.342255/+0.507449mm |
| 末态几何法向dot | -0.373106，仅预测相向 |
| 原生可运行/Teacher资格 | false/false |

负下界本身不证明穿透。因此对index157、实际节点4另做**一次**原有convex_patch_closest可行QP检查：source finger13凸包内可行点距节点4中心4.748622mm，小于5mm半径，gap上界-0.251378mm。
它说明这组源几何存在重叠，不仅是支撑下界不够紧；不是实际native非指腹cooked碰撞或穿透深度测量。
节点[0.4300003648,-0.1999822259,0.3049999177]m，凸包点[0.4340294707,-0.2012272332,0.3071829589]m。
one-source-hull-feasible-witness.json SHA4087c0e88975e51188eb661fca5116a701413254194ee169e85f057f8a607c0b。

## 失败和测试

首次脚本把实际wrist_anchor当成源指关节FK，KeyError即停止，原run保留在right-pinch-measured-20261113-_2ft1ppc。增加独立挂载link/碰撞link测试后修复；没有忽略缺失指关节。
测试先模块缺失11失败、再新映射/有限refine/失败路径选择测试各RED，最终14新增+63相关=**77 passed，0 failed，7.00s**。
日志 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-pinch-measured-full.log
SHA e348d6ae2e8720fd36ba0b8c422a2dd5cbcf53a462073de07e7d0315f13cb006。

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_right_pinch_measured_direction.py tests/test_dh116_polyline_support_certificate.py tests/test_dh116_pinch_closure_margin.py tests/test_dh116_pinch_rod_axis_clearance.py tests/test_dh116_pinch_segment_geometry.py
```
CLI：python -m scripts.audit_dh116_right_pinch_measured_direction --output <独立目录>；
--refine-from仅接受上述首方向SHA，--path-from仅接受上述refine SHA。每次90秒有界，输出exclusive+flush/fsync。上述CPU批次已完成，不原样重复。

## 结论和下一任务

当前固定右掌姿态下，单纯增加拇指闭合不能获得所需接触，且候选途中存在非目标source finger13重叠。需改变独立右Reset的展指/空间关系后重新作有界几何验证，不允许提高动作幅度、取消碰撞或改关节硬限位。

全部预测冻结SDK234杆位置，不假装实际杆响应、实际追踪或连续模型误差已验证。实际关节状态、23link和右侧原始来源保留；旧失败不覆盖。
Pinch左短接触通过限定执行复审；右/Tripod/Power及连续材料滑移/R0仍阻断。WireSeg本包新增训练步0，既有128seg/128depth未增加。相机不阻断无相机准备。
528保护文件和两个Checkpoint由发布器复核，PR2继续Draft。
