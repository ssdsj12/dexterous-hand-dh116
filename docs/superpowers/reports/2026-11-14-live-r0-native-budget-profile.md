# 真实 Reset/R0 原生结果及 CPU 性能剖析

## 原生结果与失败门控

测试源码 HEAD：0786429f52af87f378a4c59b212a90e2373921b5。
真实完成 234 SDK 步（2 初始化、232 测量）、29 个控制步、0 PPO 更新。
同一原生进程在实际 Reset 后、首个动作前初始化 R0；初始化前后 SDK/solver 时钟一致。
232 条 R0 记录绑定实际物理事件 SHA、env/episode、policy/physics step；29 个控制步完整结算。
app.close 返回、外层自然退出0、GDB inferior signal=null，无强制终止。
但总耗时187.245029秒超过180秒，完整原生配方资格仍为false，Registry保留failed，不能晋级Teacher。

本次回执 SHA256：e92282a2a01596b551342ac56cd046f4f5a0c289eda51cf76578b8712e4d89d1。
原物理账本SHA：0b76e7ff40b5d7e5ff9d054313f12d82e3766109e4752539a6d4b2949afa8227。
R0账本SHA：01dc49d20814c710a213154db1e0cc2b4df8ef1000269e7c6cbd15c2ff5a5a97。
退出后只读同钟复核SHA：cff8d66676732bf9ac9dbec787c44e86c7b48ac898515e666cf92653ea7b7a3c。
旧六hook顺序未声明通过；实际采用独立耦合Reset后初始化合同。
本次配方已消耗，不自动重跑或延长180秒门槛。

## 实际生命周期分段

| ownership时间戳区间 | 上次无R0观察器 | 本次 |
|---|---:|---:|
| Reset完成至owned引用释放 | 89.674s | 120.678s |
| before_app_close至after_app_close | 8.392s | 9.026s |

这些是两次原生事件时间差，不能单独证明函数级因果。未出现新的SIGSEGV；额外时间主要落在Reset后、引用释放前。

## 新增可执行诊断

scripts/profile_dh116_live_r0_observer.py：
- 只读取固定回执、独立审计及账本；核对SHA和原运行源码摘要。
- 失败原生结果读取available_artifact_sha256，绝不冒充通过回执。
- 纯CPU回放真实1次Reset、29个动作窗口、232个物理样本及29个结算。
- cProfile记录编码、复制、检查路径。600秒CPU上限；不调用Isaac，不推进物理，不训练。
- 保留全部缺失reward source；原生失败状态和Teacher禁入不改变。
- compare_phases要求唯一且有序的事件，输出时间区间，明确不授予因果结论。

新测试5项通过。首次实际剖析由于失败审计采用available_artifact_sha256而不是成功字段而退出；已保留profile01失败证据并修复、增加回归，profile02完成。
此错误发生在CPU读取环节，不是第二次原生实验。

剖析耗时（含cProfile开销）46.440257s；编码输出59948724字节。写盘没有纳入本次函数剖析。

| 函数 | 累计CPU秒 |
|---|---:|
| record | 33.809 |
| encode_owned | 27.782 |
| <dictcomp> | 27.750 |
| <listcomp> | 25.183 |
| _physical | 21.810 |
| _digest | 20.514 |
| _copy | 14.032 |
| _measurements | 10.220 |
| _boundary | 10.196 |

累计时间互相嵌套，禁止相加或直接解释成原生节省量。
encode_owned大量逐元素Python递归是下一工作包的可检验优化目标。必须先做逐字节/摘要等价、非有限值、dtype、Mask、源身份和作用域恢复测试，再进行同账本对照。优化不删除物理步骤、不关闭验证、不改变接触或R0判据。

## 剩余阻断

原R0观察器仍缺approach、hold、material slip、instability等实际测量来源或适用性合同；原训练系数仍为null。
本次原生连接不启用之前的离线hold必要条件扩展。
teacher_admission=false、training_reward=null；没有开始Teacher PPO或成功抓取评估。
Pinch/Tripod/Power尚不能据此宣称可训练；528保护文件及历史Checkpoint未覆盖。

下一包：独立有限数组编码加速，严格保持序列化语义，并用同一账本测量；只有新冻结代码和配方才考虑新有界原生验证，不扩大预算。

## 私有证据摘要
- red.log `1edcd571c1c045d771fc283b322a66b15a424d48966d330c8976153e75511c53`
- green02.log `31aea369e724e06e40899f30e41b4d08322eb801791b7267f074241d775f0735`
- profile01/result.json `e0829dd32e9577f0a94666992352509d8fe253f3deff8bce5e4aca0e0177a2b4`
- profile02/result.json `fcde4f916872b978adbd6a46ae4a37c173331808fed0a650c35cf9ac0af6d33d`
