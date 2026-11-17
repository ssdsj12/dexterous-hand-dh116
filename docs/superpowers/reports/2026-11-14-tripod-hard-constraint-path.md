# Tripod 硬约束端态与连续 approach 参考路径

基线2c31e9600a3a3992c66c304ad3f253a9fb61cbb0。只复用独立左手Tripod lineage、原17source+5locked cooked形状、原1mm SDF、原41杆节点、原耦合关节限位和原2.1mm桌面门。没有复用Pinch姿态，没有改USD、碰撞、材料、控制、371D/24D、30mm/5s/0.5mm。

**本包找到通过原几何门的近接触端态，以及从原独立起点到该端态的连续参考路径。尚未找到三指都进入solver接触条件且满足全部几何门的端态，未运行原生，未授予Teacher资格。**

## 1. 硬约束求解，不再用残差改善掩盖越界

此前node-aware least-squares的最佳探针仍有finger13相交、桌面不足和几何对向失败。本次复用同一原CPU变量域：root xyz各±5mm、world rotvec各±.1rad、spread±.1rad、三闭合角各[0,.1]rad并与原限位相交。SLSQP将桌面、非目标杆间隙、三pad连续hull区间、节点SDF、梯度与两种对向指标作为逐项不等式。

求解内部目标比原门更紧：桌面2.105mm、非目标gap下界5μm、pad区间±45μm、dot≤−.205；**这是数值搜索余量，不修改原物理门**。目标函数仅包含节点到−25μm的距离。每种模式最多20次SQP迭代、500次真实几何查询、55秒CPU软限，缓存重复查询不计新增几何。每个探针/迭代全量保存。最终原门另由旧endpoint_gate复核。

near-contact模式实际452次查询、5.224010246秒；solver因迭代上限退出，不能称最优或收敛。原几何门通过的查询284条，最终返回点也通过全部原门和更紧的搜索约束：

| 指标 | 最终返回点 |
|---|---:|
| 最低桌面间隙 | 2.198028920mm |
| 非目标杆gap保守下界 | 5.550681382μm |
| thumb/index/middle node-SDF | +22.689876 / +5.889104 / +42.849706μm |
| 三pad连续hull区间 | [6.691,11.550] / [−44.840,−31.328] / [−44.965,1.026]μm |
| 几何thumb-index / thumb-middle dot | −0.459307 / −0.856384 |
| SDF梯度对应dot | −0.561475 / −0.877194 |

这些SDF都是正值，因此只代表近接触，不是原生接触/载荷。程序保存的旧字段accepted_iterates实际是SLSQP callback迭代记录，不表示物理接受、模型训练或每点可行；最终代码已明确命名solver_callback_iterates。原始结果不改，保留两个执行源码副本及SHA。

## 2. 强制三节点接触候选未通过

独立require-node-contact模式从上述实际最终点启动，把节点搜索上界收紧为−5μm，下界仍−45μm，其余原限制不变。实际442次查询、4.885749350秒，0条同时满足全部硬约束。95条查询仅通过旧近接触门，不能被称为三指接触候选。

最终thumb gap+257.742805μm、index−3.793662μm、middle+11.205090μm；原端态门false。该运行也到达20迭代上限，不构成全局不可行证明。保留所有失败查询，不提高速度、位移、关节界限或接触容差，不启动原生“碰运气”。

## 3. 固定新掌位张开三指方案被拒绝

从near-contact端态将thumb/index/middle各回退.03rad，固定掌位后按原.09rad/s、160/480秒闭合。初态source finger13桌面间隙0.570077991mm，低于2.1mm；thumb、中指等也有不足。所有点仍在桌面上方，因此这是**安全净空不足，不是已发生穿透**。路径在0和0.5处有实际查询违约，直接拒绝，不用连续下界替它开绿灯。

实际2.262443735秒，3个路径样本即拒绝。没有把这个新张开姿态保存为运行Reset配置。

## 4. 原起点→新端态，掌位与关节联合参考路径通过

保留上一包已验证的原Tripod独立起点，关节线性插值，root世界平移线性插值，root旋转使用最短SO(3)旋转路径。它尚未映射成实际intrinsic-XYZ SDK轨迹。

连续证明对每个原形状使用：
L = ||root_translation|| + ||world_rotvec|| × (初态点到root最大半径 + joint_point_bound) + joint_point_bound。
joint_point_bound复用原完整主动/被动耦合角度×杠杆界限，没有漏掉root旋转或平移。每个区间取两端和中点，扣L×区间长度/4；证明不了的区间继续二分，最多深度12、1025样本。负实测点、预算耗尽或身份改变均不授予通过。

实际119个样本、59个不重叠连续覆盖区间、0未决、0违约，2.192560304秒。原起点桌面/落杆几何和连续节点SDF也重新通过。全路径保守最小安全余量 **0.910426838μm**，非常小；它不涵盖实际控制跟踪偏差和自由杆运动，不替代运行时保护。

逐区间保留全部table/non-target margin、全局点位移上界、源SHA。quiet参考只表示固定pad与规定3mm垂直节点线段，不是实际quiet步沉降证据。

## 5. 实现、测试和证据

新增实际函数：
- solve_dh116_tripod_hard_constraints.contract / Study / solve：三pad与两对方向完整性、硬约束、原界限内绝对步长差分、真实查询缓存、受限SLSQP。
- audit_dh116_tripod_fixed_root_path.candidate / certify_path：原速度范围内张开闭合假设、逐区间保守证明/拒绝、完整样本保留。
- audit_dh116_tripod_pose_path.pose_at / combine_bounds / run：proper WXYZ姿态插值、完整root+joint点位移界、同独立起点的联合路径检查。

先分别记录7、3、6、5项缺失实现失败；额外5项缺pad/方向约束测试先失败后修复，避免zip截断漏约束。最终新增30项和旧相关46项共 **76 passed、0 failed、0 skipped，3.83秒**。实际私有证据测试核SHA、最终点身份、近接触与接触区别、失败Reset、59区间无缝覆盖以及资格false。CPU测试不计原生成功。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_tripod_hard_constraints.py tests/test_tripod_contact_constraints.py tests/test_tripod_constraint_shapes.py tests/test_tripod_fixed_root_path.py tests/test_tripod_pose_path.py tests/test_tripod_hard_constraint_actual.py tests/test_tripod_node_closure.py tests/test_dh116_tripod_two_phase_sdf.py tests/test_sdf_segment_extrema.py
# 各模式输出使用独占新目录；不得覆盖下列实际结果。
bash scripts/python.sh scripts/solve_dh116_tripod_hard_constraints.py --output <new-directory> --wall-seconds 55
bash scripts/python.sh scripts/solve_dh116_tripod_hard_constraints.py --require-node-contact --output <new-directory> --wall-seconds 55
bash scripts/python.sh scripts/audit_dh116_tripod_fixed_root_path.py --output <new-directory> --wall-seconds 55
bash scripts/python.sh scripts/audit_dh116_tripod_pose_path.py --output <new-directory> --wall-seconds 55
```

实际执行均使用上方同一Ubuntu Python环境和CUDA_VISIBLE_DEVICES=''。0 Isaac、0 GPU物理、0新physics step、0PPO/BC/其他训练更新，无新原生预算消费。

| 实际结果 | SHA256 |
|---|---|
| tripod-hard-constraints-20261114-run01/result.json | a484586dd4e08cbdfedd740fba3df53b4bcf9d636b16a7162e624422665fb1df |
| tripod-hard-constraints-20261114-run02/result.json | 858c72d463cda94f3416087dc6c6e7892a2e6f3b6450f28c288ec93558b6670f |
| tripod-fixed-root-path-20261114-run01/result.json | 5a95240e0b75e5dc5edc651489ba05ef104767db52d4b0bd99b282bdfe3c5535 |
| tripod-pose-path-20261114-run01/result.json | a0aa3d147095585c63c03922b1e94278452530b2cca5671b0b1b75b6527035a0 |
| near-contact-source.py | 2cfcf9d60faf61cf08a03b0278663f9b5728b884fbf47f91017e96d5c110e971 |
| contact-source.py | d3db01d40f6296600aefa381e2374a0e9f326bb8603bac84e27ecf2cd603e092 |
| regression.log | 5f67304e4ff13183db745cf7bba8deb96bab3088d64a0648260274603a693822 |

结果目录前缀/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/；日志及执行源码副本在/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-hard-constraints-20261114/。不上传原始大记录。

## 后继和阻断

Tripod不再被“没有安全近接触端态/没有任何连续参考路径”阻断，但尚未满足实际SDK动作路径、真实接触、独立Reset/R0或Teacher门。下一项需按原24D worldXYZ/intrinsicXYZ合同离散到60Hz/480Hz，核对速度和预算，并重新核验实际分段Euler目标路径；不能直接把SO(3)参考路径等同执行目标。右手Tripod继续独立，不复制左手结果。

Pinch仍需稳定性和连续材料滑移合格来源；Power、相机和数据路径独立推进。PR保持Draft，原生标准不变。
