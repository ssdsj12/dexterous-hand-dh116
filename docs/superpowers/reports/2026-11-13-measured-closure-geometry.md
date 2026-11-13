# 实际逐 SDK 步 closure geometry helper：CPU 实现与验证

在 profile commit 68b646753ec2c1afc93d78116d8fc806df09672d 上新增只读 helper、测试和本报告。冻结 profile SHA 67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498 保持原字节；原worker、阈值、配置、integration不改。0新原生/SDK步，0训练更新，不启动实验、不push。

源码：src/go2_pvcnn/control/dh116/tabletop_measured_closure_geometry.py
SHA256：3e90dca3dfc2448ad2a1e82f545c29e7229fbda7510b90fba45bf88a5a7132de

## API 与保存顺序

~~~python
# Native运行时须在AppLauncher以后构造；只保存CPU原几何/SDF数组，不保存SDK handles。
geometry = MeasuredClosureGeometry(project)

# 由新的派生exchange在super()._step完成SDK+rod整步后调用。
row = super()._step(action, substep)
row["original_guard_failure"] = row["guard_failure"]
try:
    packet = geometry.capture(
        row, sim=self._sim, bindings=self._bindings, read_clock=self._rod.clock)
    row["closure_geometry"] = packet
    row["guard_failure"] = row["original_guard_failure"] or packet["terminal_reason"]
except Exception as error:
    # 外层新worker负责把错误变成原已推进row上的明确失败；不能丢失该样本。
    row["closure_geometry"] = None
    row["closure_geometry_error"] = str(error)
    row["guard_failure"] = row["original_guard_failure"] or "closure_geometry_source_invalid"
return row
~~~

上面是接入说明，helper自身不修改或定义root的worker/subclass。原control会先observe(row)落盘，再读取guard_failure结束循环。不能仅从record回调return reason，因为DualHandPhysicalEpisode内部observe不转发该返回值。读回异常保存、partial计数与停止后清理由新worker处理；不允许重试这一已推进SDK步。

disk复核：

~~~python
geometry.validate(packet, exchange_row)  # 不接SDK对象、不推进、无FK替代实际pose
# 若还需调用冻结旧validator，root用浅拷贝恢复guard_failure=original_guard_failure，
# 然后分别复核新geometry packet；不修改旧validator本身。
~~~

## 真实读回与同钟边界

capture要求exchange.rod、exchange.physics_step、action_readback.post.identity为同一完整已完成边界；run/env/episode/reset/rod/SDK/rod_tick/timestamp/solver_time/clock_valid全部相符。_clock复用原检查。读之前、原_capture期间和读取全link之后再次核当前SDK step/time与rod clock，没有模拟推进或隐式reset。

复用原_capture重新读取实际17DOF q、qd、backend targets、DOF类型、instance/path/cache/root anchor；其完整owned结果必须逐字等于该行已经记录的post状态，不能用command cache替代backend。

额外读取每侧真实get_link_transforms[1,23,7] XYZW与get_link_velocities[1,23,6]，完整拷贝出SDK。后者是PhysX world COM线/角速度，腕速度检查没有把COM速度称为link-origin速度。全部23 body names/paths必须唯一、匹配同一独立container，五pad的side/body/index/path/SDF/capture SHA逐一核对。两个mount articulation path与hand container依然独立，不做suffix猜测。

raw保存这些数组、原_capture结果、实际41杆节点和同钟身份；raw_sha256绑定包括未参加碰撞计算的腕部link。capture finally清除借用sim、read_clock、bindings、hand/view局部引用；owner只保留CPU几何/SDF/函数，不能延长live SDK handle生命周期。原依赖的异常帧仍依赖既有native owner cleanup清理，未声称代替整个生命周期框架。

## 门槛与负gap语义

每次读取原17 source与5 locked F4 cooked顶点，经该次**实际**link pose变换后计算桌面最小Z-.300。每个形状均必须>=原0.0021m，未改为1.9mm；无调用旧capture_initial的“目标pad初始不能接触”门。

非目标（除thumb finger14和index finger23）对全部41实际节点计算支撑面距离下界：
max(0, min_hull n·v - max_nodes n·p) - .005 - .000002。
64个XZ单位方向均为dual-feasible。节点凸包包含连接节点的整条折线，因此下界覆盖弯曲节点centerline，而不是错误假设当前杆一直共线。有限方向可能过保守；负下界标为non_target_not_proven_clear并停止，**不是声称已经证明碰撞**。缺少非pad live cooked资格保持原状。

required thumb/index复用锁定1mm SDF的原float32插值、梯度、phi-R与域有效性。只要求至少一个有效域内minimum且梯度非零、值有限，**允许目标负phi-R**以便接触建立；不会把目标接触当初始几何失败。单个节点在grid外保留None，不填零；整个required pad无有效读回则失败。

复用原FlexiblePhysicalExchange._guard_failure检查原joint/mimic及>30N每pad力门。另检查实际base_link位置相对本次projected target误差<=原3mm、world COM线速度<=原0.06m/s，实际intrinsic角位于原±pi/2，所有q/qd/targets/pose/velocity有限。任何非零安全投影palm offset、角度或safe_z_clipped会停止本profile的无抬升假设，**不抑制原安全投影**。actual measured qd被保存和核有限性，不把command速度上限误说成动力学实测速度的保证。

几何失败仍返回完整packet，passed=false、failure_reasons逐项列出、terminal_reason=closure_geometry_guard。非法身份/源/不可读数组抛ValueError，由worker按上述顺序保存已推进row。无分支会设置native/Teacher/grasp资格，static GPU hand-table contact仍NA。

## 无损独立复算与source锁

validate重新计算原raw内容、joint/force/wrist/table/non-target/SDF门以及摘要，整个编码后packet必须逐字一致。它不相信保存的passed/minimum/terminal标志。源和implementation SHA必须等于新构造owner实际读取的来源。primitive非有限值通过既有owned codec处理；有效读回数组自身必须有限。

.sources含：
- 每侧原source_sha256和五个NPZ路径/SHA/cooked capture SHA；
- 锁定SDF summary SHA；
- implementation七文件SHA：新helper；audit_dh116_pinch_initial_pose.py；prepare_dh116_pinch_closure_margin.py；solve_dh116_tripod_sdf_constrained_direction.py；tabletop_deformable_bimanual_step.py；tabletop_deformable_physical_exchange.py；tabletop_palm_mount.py。

其余原依赖继续由父native recipe原完整source_files锁定，包括linear_coupling、wrist_projection、R0 codec、builder、geometry/SDF loader和原task。packet的摘要不是硬件签名；原native外层receipt、完整ledger摘要和冻结源码仍不可省略。

## CPU fixtures 与验证

SDK doubles用父v4只读账本的真实q/metadata及真实初始link pose初始化，并把getter显式替换为numpy/torch数组。profile九状态测试再用冻结result中的q和原USD FK构造**CPU fixture pose**，不是本轮采集的新实际post pose。源码生产路径调用实际SDK getters；测试只验证该适配链和数学/序列语义。旧账本仍缺逐步post pad pose，未补写或冒充。

先写测试，预期缺失模块错误。两项adversarial负例曾实际失败：
1. 未参与碰撞的link pose被改，单纯指标重算未发现；新增整个raw_sha256后拒绝。
2. disk row的pad SDF binding被改，capture会拒绝但validate曾漏检；validate现在复核相同SDF/capture/path/index身份。

最终 **20新测试 + 75相关 = 95 unique CPU tests passed**。其中94项组合回归19.14s、外层20.013s，另外1项fresh-process pre-App禁导入测试4.25s。禁导入child明确拒绝pxr、PhysxSchema、omni、isaaclab、isaacsim、ropesim、warp；import模块通过，但constructor仍只允许在未来native App后调用（离线CPUvalidator例外）。

覆盖同钟漂移、乱序/外来path、NaN pose、post q不同、桌面低于2.1、非目标未证明清、SDF全域外、目标负SDF合法、腕部速度超限、raw/summary/qualification/rod/SDF绑定篡改、deep ownership、strict JSON失败行复算，以及冻结8步profile九状态全通过。

独立physics agent全文件只读审查未见新增same-clock/支撑平面/负SDF/lifetime参数清理阻断；未运行本包测试或独立重算数值，不能扩大为native审查通过。

证据目录：/home/xk/.local/share/dh116-autonomous-3c80166/evidence/measured-closure-geometry-cpu-20261113-hwy8keyr

| 文件 | SHA256 |
|---|---|
| tests.log（94项组合回归） | e468057bd7d37289610bd9528486848825e011ca0e6323309ef04bbdcdebcdef |
| tests.xml | b3dc8323c3814222bee6085cc63eb21592fe41ca093e3b37d75663a137eab669 |
| command.json | 5817a369e79b7d9f11796adf89e63bbefa81f6dc60e6dd5414cb159c1906e202 |
| verification.json | 1aef085f6a7680512de935015f08182d3a8968df22e455f4f8c6dd78bff94e96 |
| test_dh116_measured_closure_geometry.py | 1e322d289bef2e6331d9c3a3682eccce8a2c10f5e26bc0cc5daf2b153f21f6f1 |

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m pytest -q tests/test_dh116_measured_closure_geometry.py tests/test_dh116_dual_pinch_readback_profile.py tests/test_dh116_pinch_closure_margin.py tests/test_dh116_flexible_wrist_projection.py tests/test_dh116_deformable_bimanual_step.py
~~~

主线程仍需完成新的worker组合、持久化失败row的集成测试、源锁和批准门槛。此helper完成不构成实验启动许可，也不改变冻结profile字节。
