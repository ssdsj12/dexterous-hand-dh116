# Tripod smoke8 v5 独立冻结
修复Collector精确来源与grasp匹配后冻结。只允许一次8控制、66 SDK、768 solver内部子步、1更新、GPU3单实例、1环境、180/240秒。
相同源码不重跑；不改物理、动作std、SDF、保护、USD、左右手姿态或双RGB。
冻结时未执行；现已完成失败诊断，结果见下节，资格均false；成功动作/更新/Checkpoint/外层exit0须经独立原始账本重算后才可创建恢复B。
指纹：c01d8d503a1e2b1cebcf5fc726c4018fa4d4a02ad329212977625e6e6b33aa79

## 实际单次结果
tested HEAD 8ec2f08a0f3753a2f6d38275871717436f5482cf，attempt-t6gh4i7c。
实际18 SDK（2初始化+16测量）、2个issued动作、192内部solver子步、1个transition、0 PPO更新。
在SDK18，实际float32时间0.03333333507180214减前帧0.03125，与1/480相差1.7384688059778053e-9秒。
SDK与rod tick均连续，16行原物理guard均None；现有ULP分支只允许Pinch，遗漏Tripod。
before_update全局仅余12.2382628s；首窗口4.382265s，故不能假设相同预算足够完成8窗口及更新。
退出阶段SIGSEGV：libomni.physx.tensors.plugin.so -> _physicsTensors -> Py_FinalizeEx；完整栈和maps保留。
外层255/signal11，192.477620s，无强制结束；进程已结束，无新Checkpoint/成功轨迹。
回执SHA3b63619186b655e685cac00200225c9a3c103fb762f1b548f7adaf7a442f85c5，账本SHA85b0e9acd08933648083b99f0e31a6440c83d06e51fee943fbe2db6db9866387。
原配方不重跑；浮点修正、预算吞吐和故障释放分别验证，不伪造自然退出。
