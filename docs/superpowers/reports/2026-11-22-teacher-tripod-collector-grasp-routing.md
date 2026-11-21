# Tripod Collector抓法绑定与训练诊断修复
基线e7bbec2960046c970c57b9ddafaa6a4d9591fb81；保留现有算法、模型、物理和原成功标准。

## 结果与根因
v4真实来源、Reset和live_preflight已通过到Collector入口；随后exact NativePinchSources硬编码拒绝NativeTripodSources。
同时发现反向漏洞：合同为Tripod但来源为Pinch，旧Collector类型检查可放行。不是靠放松校验解决。
本包按owner.contract.config.preflight.style选择精确NativePinchSources/NativeTripodSources，并要求model_identity.grasp一致。
保持exact NativeEpisodeOwner/LiveContract、同contract/backend_profile、PPO配置与rollout长度全部原检查。Power没有独立来源，继续拒绝。
preflight_training不再把已有Tripod距离来源报告为missing；无lease/native-ready仍为false。旧CLI的32步限制不替代actual worker独立8步合同。
全链检查Session、Provider、Factories、Context、Sources、Checkpoint和artifacts；独立fixed_s0_pinch_v2与Pinch专属几何入口保持原样。

## 红绿证明与范围
新增Collector入口行为回归：合法Pinch、合法Tripod、双向source错配、model错配和未实现Power。
使用原CPU物理episode、真实网络/Trainer/Reward和真实Collector构造；native创建和读取边界以显式CPU缓冲代替。
这不是原生I/O资格测试；初始化源对象不会制造真实SDK证据；未执行控制或PPO来冒充通过。
另有Tripod/Power训练诊断门控两项。先修正测试夹具对模型config属性、已消费reward的错误假设；相关失败日志完整保留。
最终pre-fix有效红日志3失败/5通过：合法Tripod拒绝、错误Pinch来源放行、错误报告Tripod missing。
修复后相关Collector/Training47通过；依赖和当前冻结配方见下表。不同组文件无重复，不与旧包累计。

|CPU组|通过|失败|秒|
|---|---:|---:|---:|
|tripod-collector-routing-green.log|47|0|125.99|
|tripod-collector-dependent-regression.log|204|0|416.82|
|tripod-frozen-worker-v5-green.log|135|0|36.53|

本包合计386通过、0失败。仅报告所选回归，不声称全仓库所有旧归档测试通过。
所有CPU命令使用/mnt/mydisk/xk/envs/dh116/bin/python，具体参数与日志SHA见readiness的groups。
第一组-m pytest -q；其他组-B -m pytest -q -p no:cacheprovider；测试文件均列于readiness。

## v4原生失败证据（不是本修正版原生通过）
实际2 SDK初始化、0控制、0solver、0PPO，171.223436s，外层exit1、signal=null，无强制结束。
回执SHA f8ae85d4f9f7f3b8d5652a3e786b9cd444822eafde945d6a23409ee2dfe06633。
账本SHA580b0a3aa2cac095f5be4797102728fd4541457095a9c49946672c9cc0aca57b。
参数native_getter与health0；两侧Reset几何间隙均值1.94697mm/9.66112mm，不是接触力。
fault-path释放仍false，app.close返回不等于自然exit0；无shutdown_complete或新Checkpoint。
完整原始JSONL、栈、映射、退出和恢复点保留服务器外部证据目录；Registry保留v1-v4各自失败，禁止相同指纹重跑。

## 新v5及晋级边界
冻结指纹c01d8d503a1e2b1cebcf5fc726c4018fa4d4a02ad329212977625e6e6b33aa79；源码闭包3b26929d049ce7a7656d8fc73f917fd879c7027d31bf0f85d041241bdf2c4287。
预算保持GPU3、单实例、1环境、8控制、2+64 SDK、768内部solver子步、1更新、内180s/外240s；尚未执行。
使用真实Tripod17槽/41节点Reset与自己的manifest，不借用Pinch数据、标签或来源。
完整更新、Checkpoint、外层exit0与独立账本审计通过后才可基于真实A回执建立恢复B，不能预造路径。
没有完整30mm Lift、5s Hold、0.5mm Slip的抓取成功，不能进入成功BC或正式DAgger池。R2/ForceProxy关闭。
固定双RGB两份配置SHA及528保护文件、3份原历史Checkpoint复验不变；不上传原始模型、日志或数据。
Antigravity额度不可用到2026-11-27，无本包新审查意见，未把旧回复写成新审查。

下一工作包：单次v5实际动作/更新诊断；失败只阻断自身恢复后继，继续调查真实源，不修改物理阈值换通过。
