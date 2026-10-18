# DH116 USD 轴与派生限位

Baseline/Candidate：b5666c2 + dirty worktree，未提交。阶段：T500/DH116线性仿真近似；用户要求轴检查、派生限位和空载动作。

## 根因与修改

源 portable 与 linear_approx_verified_v2 左finger11的physics:axis是显式空字符串。旧验证器用 `'XYZ'.index(axis_name)`，空字符串结果为0，错误地隐式解释为X。新增严格axis_vector，拒绝空/多字符/小写/None，验证器改用此函数。

用源URDF局部轴和USD localRot1计算各标准轴在子连杆坐标中的方向。左finger11的X误差4.050723e-9，支持只修复此空轴。其他已命名轴回读有效，四指存在源导入的轴主轴化近似，最大单位向量误差0.000650711233，约0.0373度；未宣称精确复现URDF轴。未改变localRot0/localRot1，也未任意镜像左右手。

prepare_dh116_limited_usd.py固定输入v2左右SHA，先验证两侧再导出新目录，不覆盖旧资产。将active_limits()经正比例仿射展开后的11维角域写入USD，保持原mimic和禁用drive，回读检查轴与限位（float32角度容差1e-5度）。报告保存源/目标USD SHA、源URDF SHA、每关节轴误差及完整限位。

## 验证与产物

新增测试先RED 2 failed（缺模块）；实现后实际Excel/线性/新USD边界工具共47 passed in 0.22s。生成器在core0/OMP/MKL/OPENBLAS=1条件下运行，退出0。通过本机Isaac pxr离线读取，没有启动物理。

产物位于 `/home/xk/coding/M1/assets/dh116_v2/linear_limits_verified_v3/`：DH116_left.usd、DH116_right.usd、verification.json。完整11维上下界在报告内；主动范围侧摆0–80度、拇指屈伸0.453359858–59.014652898度、四指0–70度。源v2未保存写入。

Key files：[生成器](../../Go2Pvcnn/scripts/prepare_dh116_limited_usd.py)、[轴/限位工具](../../Go2Pvcnn/go2_pvcnn/control/dh116/usd_contract.py)、[验证器](../../Go2Pvcnn/scripts/verify_dh116_usd_kinematics.py)、[测试](../../Go2Pvcnn/tests/test_dh116_usd_contract.py)。

## 未完成与阻塞

空载动作NOT_RUN：stiffness/damping/maxForce保持0；尚无明确DH116仿真驱动参数。需用户同意配置仅仿真的增益/力矩/速度，不能继承O6或importer大默认值。该产物不是Play成功证据，尚无PhysX耦合误差、真实关节Jacobian或碰撞验证。后续先显式单指小幅慢速配置，检查跟踪/从动误差和异常接触，再扩至双手。

父节点：[T500](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。
