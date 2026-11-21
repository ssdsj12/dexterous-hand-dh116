# 原生 Tripod 首轮失败与参数来源字段修复
基线d388c30360cc07ebd3cf8390fd2bca19b28c7fa2。Ubuntu实际运行，经Windows发布同commit。

## 实际原生结果（失败，不授予资格）
实际包teacher.tripod-ppo-fresh-smoke8-v1，指纹c6b3b2804df1ed5f43c2fb69cd317669055241d424e45637aa34fb9de73d0a79。
GPU3，1实例，1环境。实际2个SDK初始化步、1次零步Reset；控制0、solver调用0、PPO更新0。
运行142.283234s，外层/GDB inferior均exit1，signal=null，未使用强制退出/信号，不重跑此指纹。
原始回执SHA076889e1d4714de4073cc8d424e1611af16214b5eb924731e644920725bebfb4。
账本已写入17槽supported_reset_readback、numeric_reset和双侧initial_geometry。reset_sources尚未写出即失败，所以本次没有完整参数源pack、决策、动作或更新样本。
真实Python栈指向NativeTripodSources.after_reset→MeasuredTripodResetApproach.capture_reset：
parameter and Tripod Reset source scopes differ。
故障清理路径app.close返回，但release审计false（残留SimulationContext frame引用）；没有shutdown_complete，不得将非SIGSEGV误记为自然退出0/生命周期通过。
GDB无C++崩溃信号，故无虚构C++根因栈。完整Python错误、lifetime referrer、动态运行回执和日志留Git外。

## 根因和最小修复
NativeParameterEndpoints._read原本在完成二进制/type/owner、SDK/solver clock、参数、质量和节点校验后输出参数，但没有source_scope。
已有真实Pinch A原始参数包也可见该字段缺失，未重新解释或重写这些历史数据。
新Tripod消费端严格要求native_getter，故在实际getter返回后拒绝；CPU fixture显式cpu_fixture，无法覆盖这个原生生产者字段缺口。
仅在新返回的真实getter packet补充source_scope="native_getter"，位于原有完整校验之后。
未改Tripod consumer guard、未把unknown当native、未删除校验或篡改旧数据，未升级后端、物理、PyTorch或Isaac。
task_physics_qualified与teacher_admission仍false，getter来源标签本身不提供接触/抓取资格。

## 测试和新冻结配方
在现有同钟参数getter测试增加Reset和后续endpoint两处source_scope断言；先KeyError失败，补生产字段后通过。未新增重复场景或测试专用生产接口。
getter、Tripod Reset/source、原模型与实际input audit共64通过/0失败（70.21s）。这是CPU doubles和原生历史只读数据，不是修正版原生运行。
旧配方保留失败身份并blocked。由于生产者源码变化，不能复用旧冻结SHA。
独立v2 core/launch保持同一8控制/66SDK/768内部子步/1更新/180内层和240外层范围；不改初始姿态、动作噪声、奖励或物理。
v2指纹d0986552aff722ec33bafe54f715d42aa38e1e48954c2ca5ff7b6416e2ae6369，
558文件源码闭包SHA7ca2bb71590d68cc2f4db8adf7ba306d1a4432cd0f77e7772d63977f6db56058。
原训练配置内容SHA602f0299625b17207fd0137ac52c84c9abf394679cf8575dec3ce013a60d0f7c不变；新增独立私有绑定目录，不覆盖v1。
frozen integration测试改为当前v2而非要求已消耗的历史v1通过当前源码SHA校验；先因v2缺失失败，再验证真实当前Repository配方。历史v1 JSON与失败回执均保持原字节。
修正版原生未运行，PPO/Teacher/抓取资格仍false；下一次执行必须使用新身份，不能把CPU64项当成原生修复通过。

## 仍未解决
- 修正版实际Tripod update及恢复尚待新回执。
- 故障路径SimulationContext frame释放未资格化；正常路径与最终外层状态都要分别核查。
- 完整R0、对向接触、固定腕稳定保持、完全离架30mm/5s/0.5mm尚未通过。
- Power全握/掌面仍按独立来源与Reset门槛推进，不用Tripod代替。
固定双RGB、24D/371D、528保护文件和三份历史Checkpoint保持；R2与ForceProxy继续关闭。Antigravity仍受quota限制，无本包新回复。


修正版冻结配方/启动/合同/Registry再验135通过、0失败（33.51s）；与前64项为不重叠文件，共199项。保护528文件、3份历史Checkpoint复验无变化。
