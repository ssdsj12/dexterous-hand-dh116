# Tripod原生时钟舍入修复与v5失败账本
基线8ec2f08a0f3753a2f6d38275871717436f5482cf；修复奖励时间来源的抓法路由，不调整物理或训练目标。

## 实际推进
v5已经创建真实Collector，发出2条策略24D动作，执行16测量物理步/192求解器内部子步，写入1条完整rollout transition。
加2个初始化步共18 SDK步；PPO更新0，新Checkpoint0。16个原物理guard均None；短距离动作不等于抓取或力闭合成功。
第二控制窗口在SDK18时奖励校验拒绝。真实SDK tick3..18、rod tick12..192（每次+12）连续；
solver原始时间17/18分别0.03125和0.03333333507180214，间隔与1/480仅差1.7384688059778053e-9秒。

## 根因与最小生产修改
TrainingReward._solver_interval_matches已有严格binary32舍入分支，但只允许policy_owned_pinch_endpoint_v1。
Tripod使用自己的policy_owned_tripod_endpoint_v1，因而在这个相同的精度边界被错误拒绝。
现仅将真实Pinch与Tripod unqualified smoke来源加入精确允许名单，并将私有参数名改成native_smoke。
保留原ULP误差公式、上限、数值原样保存、连续SDK tick与episode/reset/rod/来源身份检查。
没有手加毫秒offset、时间插值或更改2ms双RGB同步门槛。其他未知来源、CPU fixture与完整奖励profile仍走原严格边界。
未知Contact/Slip/Force不从几何或舍入推测；R2/ForceProxy继续关闭。

## 红绿与回归
扩展现有主要边界回归，加入Tripod真实原始浮点值、缺tick、错时间、非binary32篡改四项。
修复前1失败/8通过；失败即SDK18合法Tripod。修复后Reward/Collector/Training/Tripod source共322通过（169.17s）。
独立新v6配方/实际启动/Worker合同/Registry共135通过（35.24s），两组文件无重复，本包457通过、0失败。
实际命令和日志SHA列于readiness；只声称相关CPU回归，不声称修正版已原生通过。

## v5退出仍失败
测试SHA8ec2f08a，外层255、inferior SIGSEGV11，192.477620s，未主动发送信号或重跑。
完整GDB栈落在libomni.physx.tensors.plugin.so和_physicsTensors析构，继而Py_FinalizeEx；
app.close确已返回，但手、asset SimulationView、composer、runtime等引用仍活着，故不能计自然退出0。
这是具体C++失败边界与持有证据；尚未证明哪个上游引用是最终根因，未声称已根治。
回执SHA3b63619186b655e685cac00200225c9a3c103fb762f1b548f7adaf7a442f85c5。
完整JSONL SHA85b0e9acd08933648083b99f0e31a6440c83d06e51fee943fbe2db6db9866387。
栈/动态库映射、生命周期快照、故障数据与恢复点均保留Git外；本轮进程已结束，未进入成功池。

## 运行预算与下一包
本次进入更新前剩12.2382628s，第一窗口4.382265s，第二失败窗口3.520079s。
没有证据表明原180/240秒预算可完整完成8窗口、更新和清理，不盲目再次启动。
CPU cProfile单次真实manifest复验17.0127s：88次box_relation约5.096s，decode_owned递归约4.121s。
此为带profiling开销的CPU读回，不是新的原生采集，也不能直接当成未加profiling的实际耗时。
下一包按纯数据等价原则剖析重复解析，保留每次SHA/完整几何/来源校验；同时审计异常帧持有链。禁止通过删除校验、缓存未知资格或强制退出换通过。
v6重新绑定当前561文件闭包，指纹5a848528eeaf5b57328d0acee01299254fad4cf80eee960d41c172b364c117b1，仅冻结、未执行。
预算仍GPU3、单实例、1环境、8控制/66 SDK/768内部solver子步/1更新、180/240秒，不自动扩大。
未变双RGB、左/右手定义、24D、371D、USD、Mimic、关节范围、30mm/5s/0.5mm标准。
528保护文件、3份旧Checkpoint核验原SHA。Antigravity额度仍不可用，未取得新意见。
