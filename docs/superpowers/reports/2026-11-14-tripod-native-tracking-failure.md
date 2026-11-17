# Tripod 原生终止前缀与跟踪误差审计

基线 `94d1f2ca9b64534b0c81302f9f55033aeb627767`。本工作包只读复放已消耗原生账本，
新增原生步数0，PPO更新0。未修改控制器、关节/碰撞保护、物理材料、旧Checkpoint或原始账本。

## 原生实际结果

被审计执行HEAD `7c8c7b2186b5b61e1f07b0acebc25c9d3023bc8f`，
配方 `9a226954121beff9c2d459c523dbafe24531ceb6913c730b5b7f761a7d833990`。
实际199个SDK步=2初始化+197测量；2364个杆子步，25个控制窗口，末窗口5个有效物理样本。
原234步预算剩35步未执行，不将剩余额度用于重试。

同一身份：env0，episode `dual-pinch-41fcc62085e8494aa26f76c14c83942f:one`，
reset1，policy24/substep4，SDK199，rod2364，timestamp0.4145833333333333s。
八步静置准入在SDK3–10通过，杆最大速度0.038329238mm/s；全部后续接近都基于本实例准入。

首个终止：`left:non_target_not_proven_clear:source:finger22_link`。
非目标碰撞体到真实杆折线的保守间距在SDK197/198/199分别为
+0.015050264/+0.000078302/−0.014889359mm。负下界仅表示不能证明安全分离，
**不能写成实测穿透或指腹接触**。左/右全手最低桌面净空为2.438264/2.581601mm，
高于原2.1mm保护。末帧左拇/食/中指SDF间距为1.141156/0.064448/0.712262mm，均为正。
全程接触pair计数仅table96924，手指0；不得计为Tripod抓取或Teacher成功。

末帧左锁定cooked指腹桌面净空(mm)：finger14=3.672064、finger23=8.013630、
finger33=3.285175、finger43=11.267058、finger53=23.968744。没有这些指腹的桌面穿透证据。
未修改本轮初始姿态；后续候选必须重新检查初始及全过程几何。

## 实测与同帧运动学归因

逐行导出两手17个q/qd/SDK目标、24D动作、身份、几何与终止。
主动位置目标可比较；**被动关节SDK零目标不是主动控制命令**，只按真实耦合计算残差。

| 指标 | 左手 | 右手 |
|---|---:|---:|
| 最大主动跟踪误差(rad) | 0.002086937 | 2.794e-9 |
| 最大被动耦合残差(rad) | 2.657e-7 | 3.832e-8 |
| 最大腕部旋转跟踪误差(rad) | 0.021604292 | 1.058e-9 |
| 最大腕部平移分量误差(mm) | 0.095300376 | 0.000016006 |

原USD FK使用**实际11关节**及六轴安装链重建，末帧所有手link最大位置差
8.4203e-8m、旋转差1.0432e-6rad；因此后续离线几何替换有实测对齐依据。

| 同一末帧替换（杆、源网格不变） | finger22保守间距(mm) |
|---|---:|
| 实测link | −0.014889 |
| 实际全关节FK | −0.014946 |
| 只替换理想被动耦合 | −0.014948 |
| 只替换目标腕旋转 | +0.441293 |
| 只替换目标腕平移 | −0.052935 |
| 目标手腕、实测手指 | +0.401403 |
| 实测手腕、目标手指 | −0.163774 |
| 全部目标+理想耦合 | +0.237769 |

证据支持：腕部旋转跟踪滞后导致实际联合路径偏离此前假设精确跟踪的安全目标路径。
被动联动残差对该帧几何影响很小；仅让手指跟踪更快反而可能恶化间距。
这是**同帧运动学归因，不是新的动态因果实验**，不能据此提高执行器增益、取消保护或直接重跑。

附带CPU筛选：将腕姿直接固定到终点、从原q0开始，会令源finger13桌面净空仅0.810968mm，
仍不合格。不能用这个未经全路径检查的简化方案生成原生配方。

## 生命周期与失败保存

原生墙钟109.344887s；app.close返回，GDB inferior自然exit1，outer exit1，signal=null。
本次无SIGSEGV；失败来自原几何保护后正常异常传播。**没有exit0资格**。
释放账本保留subscription unsubscribe及SimulationContext清理返回；
native_release_verified=false，不能把Python引用清除等同所有native view已释放。
正常finalize C++栈、动态库映射均保留在原目录，本包未添加gc、强退或信号退出。

## 实际代码与回归

新增 `scripts/audit_dh116_tripod_native_tracking.py`：
Sequence校验终止前缀及60/480Hz身份；validate_control_end保留部分末窗口；
joint_tracking区分主动目标与被动耦合；geometry_comparison核对实际FK并逐项替换，
命令行只读SHA锁定原始文件，私有目录保存tracking.jsonl/result.json。
新增 `tests/test_tripod_native_tracking_audit.py` 16项，
加旧bilateral证据、quiet、projection、exchange回归共 **59 passed, 0 failed**。
保留初始缺模块RED及末窗口校验RED（1 failed/15 passed），修正后完整通过。

Ubuntu命令：
```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_tripod_native_tracking_audit.py tests/test_dh116_bilateral_native_evidence.py tests/test_tripod_quiet_gate.py tests/test_tripod_projection_scope.py tests/test_tripod_exchange_admission.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.audit_dh116_tripod_native_tracking --output /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-native-tracking-20261114-run02
```

账本：`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-tripod-owned-20261114-2xl75l3a/native-runs/flex.f8-tripod-owned-contact-native/attempt-r_cnsg84/native`。
回归日志：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/tripod-native-tracking-20261114/regression.log`。
只读分析7.269958s；结果SHA `8225701de3b8c447146dd52374fcb74b3247388c59d701946f804e082ba552e4`。
receipt SHA `9bb31008160da84e31c1e089eed925ba658d40c82a830173a958e21491ecf0eb`。
physical SHA `fc30831476a2247df49c875e185f13106168ffa117b696491ba0b25d94748f33`。
其余原始SHA在脚本LOCKS和私有结果内；大账本、模型、栈不上传GitHub。

## 晋级与下一工作包

本**分析工具**CPU通过；原生任务仍blocked，非Teacher资格。
下一步用实际跟踪误差筛选分阶段/固定腕姿接近路径，保留原始边界和独立右手；
先全路径CPU证明，再冻结新配方；本次失败配方不得重跑。
Pinch完整状态/R0、Power独立接触路径继续分别推进；三者都尚未达到完整Teacher PPO准入。
WireSeg本包新增训练0步（历史seg128/depth128诊断不重算），相机不阻断无相机Teacher准备。
成功标准30mm/5s/.5mm不变，PR2仍Draft。
