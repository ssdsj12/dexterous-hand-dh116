# Tripod 全几何闭合搜索与失败证据（2026-11-14）

## 本包结果

基线 `b21ea3976e6dc01d94abe1b3a8f0dc15c2902cf7`。沿用上一包已经证明的条件跟踪候选，新增完整 source17/cooked5、原主动/被动耦合、锁定 1 mm SDF 的闭合搜索。**没有得到通过全部静态约束的闭合端点，未发出原生配方。**

此处“完成”仅指离线程序和有界分析完成；Tripod 接触准入仍 blocked。没有新 Isaac 步、PPO 更新或成功样本，未更改既有控制器、USD、碰撞体、物理参数、执行器或旧模型。

## 实际代码

- `scripts/solve_dh116_tripod_contact_closure.py`：decode 保留独立固定重置姿态与原六主动关节合同；measure 同时检查全部碰撞体、桌面、非目标杆间距和三个指腹 SDF；bounded_search 有界 minimax 线性方向＋完整非线性回验，逐次保存查询、线性预测与回溯失败；candidate_admission 不把数值求解或 SDF 负间距当原生接触。
- `flexible_tripod_contact_closure_cpu_v1/v2/v3.json`：分别保留实际分析所用的 6 变量、9 变量及 minimax 配置；当前脚本只执行 v3。v1/v2 的对应脚本副本和完整失败记录在私有证据目录，非活动配置。
- `tests/test_tripod_contact_closure.py`：10 项新增测试覆盖限位/无效输入、完整约束、静态旋转不生成腕部旋转动作、预算保存和求解器成功不能代替几何通过。

端点的 X/Z 偏置范围各±0.4 mm；v2/v3 静态旋转向量各±0.01 rad；四个参与抓取的主动关节偏置各±0.01 rad。运动仍由原24D编译器生成，200个物理步，起止腕部方向相同；右手动作与左手其余两个主动关节保持不变。

原生所需指腹间距保护保持 -0.3 mm、桌面 2.1 mm、非目标间距>=0；本包新增独立 CPU 闭合规划使用更严格的 -0.15 mm 指腹下界及 0.01 mm 几何余量。不能把上一预接触证明的 -0.05 mm CPU 规划范围与此接触规划混称同一证明；本包没有继承上一包的连续路径通过资格。

## 真实离线证据

证据根：`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/tripod-contact-closure-20261114-runNN/`。每次预算最多600个不同几何查询、60秒，未执行原生物理。

| 运行 | 实际方法 | 查询数 | 最优最小余量 | 结论 |
|---|---|---:|---:|---|
| run01 | 6变量 SLSQP |600| -0.035108 mm | 查询预算耗尽，拇指未到目标 |
| run02 | 9变量 SLSQP，增加固定重置姿态 |600| -0.040109 mm | 查询预算耗尽，不能按状态码晋级 |
| run03 | 9变量 minimax＋非线性回溯 |600| -0.018257 mm | 原始线性预测不等于实际通过 |
| run04 | minimax 对齐最大违反量，局部差分/信赖域修正 |589| -0.019373 mm |40次迭代预算结束，仍 blocked |

run04 的逐项最优候选：
- `source:finger13_link` 杆间距扣除10 µm规划余量后为 **-17.509 µm**；原保守下界 -7.509 µm。这是“无法证明清晰”，不是实测穿透深度。
- `source:finger22_link` 规划余量 **-2.431 µm**；原下界仍 +7.569 µm，但不足新增余量。
- `cooked:finger33_link` 接触下界规划余量 **-11.126 µm**；不允许选择性忽略。
- 三指 SDF 接触目标均为 -15 µm。拇指距目标 **19.373 µm**，食指距目标 **2.933 µm**；不能因部分点负间距就宣称三指建立接触。
- 所有三维几何和 SDF 结果仅是原始源几何的 CPU 预测；没有力、接触对或闭环稳定性的新实测证据。

run04 result SHA256：`d15198dd0ee006f3189df46d01a33d5f8e21365c3b2ae18b9f9362936a9c3116`。
源父证明 SHA256：`85c84a028f36a094719d2c5f27967233fef5784f5ee5cbc63456597059ff30a9`。
消费过的原生节点身份/账本只读复用；新的杆姿态与跟踪误差仍需独立验证。

## 测试和运行

Ubuntu CPU 命令：
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_tripod_contact_closure.py tests/test_tripod_tracking_envelope.py tests/test_tripod_fixed_orientation.py tests/test_tripod_sdk_path.py tests/test_tripod_fixed_root_path.py tests/test_tripod_native_tracking_audit.py
```
**68 passed / 0 failed**，3.62 s；日志 SHA256 `2f1f84b675946f9d126774d286abff8fe984eeed733d8985a5a886776ef27b4f`。

保留了缺模块 RED、静态姿态维度 RED，以及 minimax 提前完成后原测试未再触发第二次查询的 fixture 失败。最后一种测试已改为初始不可行，确实检验第二次查询字段身份变化。所有日志在私有 evidence/tripod-contact-closure-20261114，未上传原始数据或模型。

## 剩余问题及后继

- 搜索失败不是全局不可行性证明；需继续定位拇指远端与近端的完整可达几何，不能放宽原保护。
- 没有闭合端点就没有新路径/跟踪证明，不生成或消耗新原生配方。
- 条件跟踪包络仍只说明既定延迟假设下的几何，不是未来实际跟踪保证。
- Pinch 的已完成原生账本、R0缺失测量资格、Power独立预检等工作可继续，不由此包全局阻断。
- R2、正式 Teacher PPO、BC/DAgger 和 Student 成功门槛未开启。PR #2保持Draft。
