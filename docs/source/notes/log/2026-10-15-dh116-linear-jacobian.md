# DH116 USD 线性耦合与运动学 Jacobian

Baseline / Candidate: b5666c2 + dirty worktree。用户明确批准 URDF 线性耦合作为仿真近似，真实驱动力矩/速度另核实，不继承 O6。

新增独立 `go2_pvcnn/control/dh116/linear_coupling.py`，6 主动角展开到 11 关节角，支持按实际关节名重排 Jacobian 列并执行 J_active=J_full*C。单位为 rad，拒绝超限和非有限状态，不静默 clamp。拇指主动下限取 0.00697725445/0.88178789，避免线性 offset 导致从动角负值。

新增 `scripts/verify_dh116_usd_kinematics.py`，固定左右 portable USD SHA，直接读取 USD joint localPos/localRot/axis/body0/body1 构造 FK。每侧三个主动状态（各维 0.1、0.3、0.7 rad）及五个 distal link 原点，检查解析点 Jacobian 与中心差分 epsilon=1e-6 rad。

结果：左右各15个点姿态探针，最大绝对误差均 1.1437151573034932e-11 m/rad，小于 1e-7 门限。仅 USD 运动链内部一致性，未交叉验证 PhysX 动态或厂家 FK。末端连杆原点不等于已标定接触点。

发现：原导入 USD 没有 PhysxMimicJointAPI，尽管转换配置请求保留 mimic。原文件 drive maxForce/maxJointVelocity 是很大的 importer 默认值，而非有效厂家上限。新派生 USD 显式写入五个 PhysX rotX mimic，gearing=-multiplier，offset=-degrees(URDF offset)，回读校验 reference/系数；所有关节 stiffness、damping、maxForce=0，未启用运动。

产物：`/home/xk/coding/M1/assets/dh116_v2/linear_approx_verified_v2/DH116_left.usd`、`DH116_right.usd`、`verification.json`。初次 linear_approx 保留作诊断。中途 verified 检查错误地要求所有关节轴为 X，实际非耦合主动关节可为其他轴，故拒绝；已修正为读取各轴、只验证 mimic 配对 X 轴，v2 完整复跑通过。驱动禁用，不应直接作为已能抓取的 Play 资产。

验证：新测试首次因缺模块失败，随后7项通过。完整固定双臂相关回归 **680 passed in 6.34s**。真实 USD 检查使用 go2/headless GPU0，core0/数值单线程，无 physics step。旧 O6、Panda、wire 运行代码未改。

仍 OPEN：双 Panda 安装及统一动作入口、PhysX 实测 Jacobian、指腹标定、驱动参数、动态开合和抓握。不得将本轮称为完整仿真替换已完成。

独立 CPU 代码复审未发现阻塞问题，文档 rows/columns 描述已修正；全轴 X 的过严检查由父级真实 USD 运行发现并修正，不能用代码复审替代真实资产检查。
