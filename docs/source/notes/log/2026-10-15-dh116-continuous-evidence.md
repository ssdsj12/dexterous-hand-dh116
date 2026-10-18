# DH116 连续流程与流式物理证据

阶段 D4，父节点 [DH116 抓握](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)，按 [held transport 计划](../../docs/superpowers/plans/2026-10-15-dh116-held-transport.md) Inline 执行。Baseline/Candidate `5e3bcc4` + dirty，不提交、不推送。

## 连续流程纯核心

`TabletopGraspSequence` 已实现落定→实际接近误差→预载闭指→保持→双区实测抬升流程，拒绝首条即4s和过期控制请求。核心不能移动机器人或约束wire，只有后续实际运行测量才能授权运动。

测试：先缺模块RED，后专项30通过；新增时间漏洞RED 1失败/10通过，修复后 DH116/moving-contact **388 passed in49.18s**。日志 `/tmp/dh116-grasp-sequence-full.log`。仍未接入连续物理运行，不是实际抬升证据。

## D4-continuous-evidence 子节点

原因：原验证脚本将每帧大记录存在列表，长时100µs连续流程容易大量占用内存，且结束前无法读取trajectory。新增 `EvidenceJournal` 独占创建JSONL，每步严格JSON转换后flush，仅保留最后一条，总结通过流式重读。不能覆盖旧证据；异常退出不截断已有轨迹。flush不等于断电耐久保证。

改动：`Go2Pvcnn/go2_pvcnn/control/dh116/evidence_journal.py`、`Go2Pvcnn/scripts/verify_fixed_dual_panda_dh116.py`。控制、力目标、安全门、反作用和物理步长不变。已启动pilot7使用原脚本快照，不热加载本改动。

测试：缺模块RED；实现与运行脚本接线后6项专项通过（1000条顺序重读、异常保留、拒绝覆盖/NaN、NumPy转换、仅保留最后、finally不重写）。首轮完整回归1失败/393通过：旧AST合同只接受finally写trajectory；更新为检查loop写入、finally关闭不截断，同时保留失败步和最后尝试帧要求。再次完整 **394 passed in50.21s**，`/tmp/dh116-journal-full-green.log`。测试均core0/数值单线程；当前物理pilot仍是旧记录方式，不能把新代码测试当作其流式写入证据。

## 物理边界与下一步

pilot7 PID73847/session29342仍在原30000步闭指诊断运行。观察4441步四路有效、载荷约.521/.530/.478/.520N，最长四路连续.1565s、最大材料滑移.5013mm；未达到完整预载/1s保持，无抬升。保留原保护，不伪称成功。

后续：跟进pilot7；只有实际结果支持晋级后接入连续运行的测量、共同双臂速度、闭指与抬升门。真实保持/抬升及重复验收仍OPEN。
