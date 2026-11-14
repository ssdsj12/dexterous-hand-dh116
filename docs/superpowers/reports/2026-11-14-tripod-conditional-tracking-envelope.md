# Tripod 条件跟踪范围与独立几何证明核验

基线 `2b72102f8fa3d04efba7d5f48491c8736e0633f1`。CPU工作包，无新增SDK步、PPO更新或原生配方。
原物理、动作率、碰撞网格、关节/桌面保护和历史Checkpoint不变。

## 候选变化与实际结果

沿用固定腕姿参考，在独立候选的起点和终点同时增加World X=0.05mm、Z=0.15mm，
World Y不变。不是附加力或运行时物体位移；没有把候选写入旧原生配置。
左右身份不互换，右手未修改/重新资格化。初始q0与终点q1仍来自原冻结Tripod参考。
掌根从Z=0.3861294128948734m下降到0.3846294128948734m，X=0.5337631957393896m、
Y=0.1384745111335029m；固定wxyz四元数
[-0.4081791671129427,-0.4692063685461108,0.5879675397388606,0.5172323689254057]。
相对上一CPU候选只变初始/终点共同位置；24D动作序列不变、腕旋转动作0，原60/480Hz不变。

旧候选928个滞后状态中24项失败；新候选 **928/928采样状态通过**。
最低采样裕量0.034369823mm，限制项为cooked finger33相对原参考允许间距。
该采样结果本身没有授予原生资格。

## 连续条件证明

显式域为motion/observe连续时刻t∈[0,224]tick；
finger11/12/21独立延迟∈[0,11]，finger31∈[0,16]，
平移延迟∈[0,6]。其余两指静止；腕姿固定。
这些延迟范围来自已消耗原生账本的测量加保守筛选余量，**不是新运行的执行器性能保证**。

使用原USD关节帧/耦合及source17+cooked5完整点集，按每个关节下游杆臂构造点运动上界。
每个子域中心的有效距离下界减去最大点运动半径，证明整个子域满足约束。
桌面仍2.1mm；非目标杆间距仍非负；必要指腹沿用较严格CPU参考阈值−0.05mm。
额外保留10um规划余量，不降低原保护。

最初使用未收紧的全局位移界，在4096次查询预算内有9个未决域，不能判通过。
记录run01原件。随后根据clip阶段的精确区间计算点运动半径，
消除已经完成运动部分的虚假位移上界，**保持4096次预算不变**。

最终3215个查询、1608个子域覆盖全部六维域，无违规、无未决域。
独立核验器重新计算1608个中心，校验边界、非重叠、总体积28621824、
源距离与每个下界；拒绝缺块、重叠、篡改样本或虚增裕量。
扣除10um规划余量后的最小证明裕量1.5231068113e-8m。
这是条件运动学集合的证明，不是实测未来跟踪、实际接触或Teacher训练资格。

## 终点还没有接触

锁定原SDF、原float32采样规则与SDK199真实杆节点参考下：

| 左指腹 | 终点SDF最小间距(mm) | 原生接触资格 |
|---|---:|---|
| 拇指finger14 | +0.101327896 | 无 |
| 食指finger23 | +0.086228829 | 无 |
| 中指finger33 | +0.053838827 | 无 |

三个距离均为正。该路径是更可靠的**接触前参考**，不得称三指已承载。
下一步需要在原安全范围内建立真实闭合目标，并重新验证路径；
不会仅为了证明安全就消耗一次必然无接触的原生运行。
实际新实例的静置、节点位移、主动/被动跟踪、SDF接触与自然退出仍需分别验证。

## 实际源码、配置和测试

新增：
- `scripts/certify_dh116_tripod_tracking_envelope.py`：
  shifted_plan、configuration_at、setup_query、certify_box、phase_radii、
  verify_box_certificate、endpoint_sdf；复用原控制编译器、几何与SDF接口。
- `configs/dh116_tasks/flexible_tripod_tracking_margin_cpu_v1.json`：
  独立CPU候选，native_execution_enabled=false。
- `tests/test_tripod_tracking_envelope.py`：14项。
- 本报告。

含旧固定腕姿/SDK路径/固定根路径/原生跟踪审计回归共 **58 passed,0 failed**。
保留缺模块、clip半径、独立覆盖校验的RED日志；
覆盖校验测试初稿使用单域已可证明的线性函数，后来改为有效保守界下的常量函数，
以实际覆盖多个子域并测试缺块/重叠；没有为通过测试放宽实现。
源码尚不改变旧默认控制路径或启动器。

Ubuntu命令：
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_tripod_tracking_envelope.py tests/test_tripod_fixed_orientation.py tests/test_tripod_sdk_path.py tests/test_tripod_fixed_root_path.py tests/test_tripod_native_tracking_audit.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.certify_dh116_tripod_tracking_envelope --output /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-tracking-margin-20261114-run03 --max-seconds 240
```

完整结果30.652806s，SHA
`85c84a028f36a094719d2c5f27967233fef5784f5ee5cbc63456597059ff30a9`。
源receipt `9bb31008160da84e31c1e089eed925ba658d40c82a830173a958e21491ecf0eb`，
源physical `fc30831476a2247df49c875e185f13106168ffa117b696491ba0b25d94748f33`。
失败/回归日志在
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-tracking-margin-20261114/`。
原始数据和大证明文件留在Ubuntu，不上传模型、日志或私有原始数据。

## 晋级限制

conditional_kinematic_envelope_proved=true；actual_tracking_qualified/native_ready/
teacher_admission/contact_established均false。不将CPU域证明写成原生物理通过。
未重跑已消耗配方。Pinch的稳定/材料滑移/R0与Power独立闭合继续分别blocked；
WireSeg本包新增训练0步；PR2保持Draft。成功标准30mm/5s/.5mm完全不变。
