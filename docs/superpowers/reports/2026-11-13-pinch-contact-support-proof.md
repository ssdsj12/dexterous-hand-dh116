# Pinch 实际接触失败与三维分离证明（2026-11-13）

本包基线 `35d492d49a808bb2b6ccab3260870acc2b1a18bf`。原生实验来自准备提交48d7d26、发布8402c6db，单次预算已消耗；不重跑、不改原回执。CPU几何证明不授予原生通过。

## 原生结果

- 配方 `09dbb0e387f7f879689399f39ad1c09f22dcbf545dc6647f80269142f830c47b`；调度指纹 `f7039b974394393f54026a051662d692553f4b8cb45ba4f4d1c2becc831036d7`。
- 最多234步，实际216步=2初始化+214测量；墙钟114.286942s，1环境，0PPO。
- SDK216、policy26/substep5：`closure_geometry_guard`，唯一几何原因 `left:non_target_not_proven_clear:source:finger22_link`。
- 原关节/Mimic/桌面/腕部保护没有触发。最小source桌面净空：左2.117664mm、右2.116345mm。
- 外层exit1，GDB inferior exit1、signal=null；没有SIGSEGV、SIGTERM或SIGKILL。app.close返回，但**不是自然exit0通过**。
- 旧配置/动作/模型/阈值保持；该运行不是Teacher或抓取成功。

| 左指腹 | 首个测量步source桌面净空mm | 锁定cooked桌面净空mm | 穿透mm |
|---|---:|---:|---:|
| finger14 | 2.992700 | 3.086673 | 0 |
| finger23 | 2.833199 | 3.172788 | 0 |
| finger43 | 55.877917 | 56.030032 | 0 |
| finger53 | 73.364324 | 73.510211 | 0 |

初始姿态来自已发布最大净空派生配置，本包没有再次修改姿态或网格。完整24D计划为48步静置、160步左指闭合、24步保持，原右手动作全零、腕部无上升。新报告不能把静态目标位姿当成原生实际位姿。

## 真实接触

| 原生partner | 首次SDK步/solver子步 | 最后SDK步/子步 | pair数 | 最大法向力N | 最大合力N |
|---|---|---|---:|---:|---:|
| left.finger14_link | 191/2 | 216/11 | 310 | 0.265465 | 0.299448 |
| left.finger23_link | 193/9 | 216/11 | 279 | 0.282487 | 0.330531 |
| 右手 | 未发生 | 未发生 | 0 | 未记录 | 未记录 |

两指同时正法向接触279个连续solver子步=48.4375ms；24个480Hz记录覆盖该区间，不能将24个记录等同24个完整同时接触步。累计杆-桌pair103032；桌面仍提供支撑，不能计入独立Lift。逐pair force×dt=impulse核验最大残差0。接触力为真实Solver仿真接触，不是硬件传感器。手到静态桌面的Tensor来源仍NA。

## 已定位的几何原因及实际代码

原 `tabletop_measured_closure_geometry.rod_support_lower()` 只用64个XZ方向，对全部41节点共同求支撑上界。柔性杆在左手作用下变形后，Y方向隔离被忽略，原界得到−3.674148µm；这代表无法证明净空，并非已经测得穿透。原程序按该证据停止是正确的保守处理。

新增 `tabletop_polyline_support_certificate.certify/validate`：
对原相邻线段[a,b]和任意长度不超过1的法向n，使用完整原顶点V计算
`max(0,min(V·n)-max(a·n,b·n)) - 0.005 - 0.000002`。
每段选一个可验证方向，全部段取最小。候选包括原64XZ、XYZ轴、实际source凸包面法向、形心至线段中点方向。凸包只提供方向；支撑验证仍使用全部750个原顶点，不删除困难顶点、不把隐含直线替换弯杆。取新旧两个有效下界的最大值，不改变半径、2µm数值裕量或零保护线。

SDK216实际finger22位姿/41节点重算：完整3D逐段下界**+0.497762mm**。3097候选方向；1496面仅用于方向生成。证书含原数组SHA、每段法向、两侧投影、下界，并可不依赖凸包求解器独立复核。此为CPU源几何证明，非新的cooked hull或PhysX接触资格；原source17/locked5范围不扩大。新函数尚未接入旧默认运行路径。

新增 `audit_dh116_pinch_contact_failure.py`：逐行检查完整JSONL、同env/episode/reset、连续SDK/solver时间、post-action身份、force/impulse单位、terminal后无no-op覆盖。输出独立trajectory和失败证书，不覆盖原始日志。

## 生命周期证据

本轮两个Articulation、Tensor views、composer、Rope/runtime、director reader、table source弱引用均在app.close前消失；SimulationContext仍由Isaac Lab构造帧引用。callback已显式unsubscribe，clear_instance返回，无额外gc。对象不可达不等于native view释放；本轮没有取得全部资源释放资格。GDB正常finalize栈及动态库映射已保留，不能把历史C5栈套用到本次。没有新崩溃栈，退出1来自明确几何保护异常。

## 复现与测试

CPU命令（Ubuntu scripts/python.sh，CUDA_VISIBLE_DEVICES为空）：

```text
python -m pytest -q tests/test_dh116_pinch_contact_failure_audit.py tests/test_dh116_polyline_support_certificate.py tests/test_dh116_measured_closure_geometry.py
python scripts/audit_dh116_pinch_contact_failure.py --ledger <native>/physical.jsonl --sha256 05fb4242cc11fb23017b32f1d4d2820dc3f669a16f18f9f5a1c27f8136070cb9 --output <new-output>
```

先红：新增证明11失败/1旧源码保护通过；新增账本9失败。实现后CPU **41 passed，0 failed，21.52s**。
原生日志重放是离线审计，不计新的原生步数。两项均未训练模型。

原始账本路径：
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-pinch-margin-contact-20261113-8XuHU4/native-runs/flex.f7-pinch-margin-contact-native/attempt-uoa3kszq/native/physical.jsonl`
（93026016字节；SHA如上）。

审计结果：
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-contact-proof-2NQpDq/audit/result.json`
SHA `a7ae3493f3dca125b82d74d2c13e27774b195efb9fcfd70708d3d4876f281291`；
trajectory SHA `b8b66b765f95b4a88fdab51478338e2f0c6d25a975dee5822600e96c934dadc9`。
大日志/数组不上传GitHub。

## 后继及仍未解决

1. 隔离接入三维证明，新配方先完整CPU所有权/终止测试，冻结新指纹后按持续授权单次≤234步/180秒。原234步配方失败保持blocked。
2. 右手、Tripod、Power仍需独立几何和真实接触，不能用本左Pinch结果代替。
3. Teacher仍0/3，371D来源/R0动态量、材料滑移与持续接触/Reset资格未齐；R2关闭，不启动PPO。
4. WireSeg保持已有128步seg+128步depth诊断，不因本包增加训练；相机问题独立，不阻塞无相机证据开发。
