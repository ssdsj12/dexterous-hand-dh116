# Teacher归档本体来源校验：实际最小修复

基线98f7d4aee6a0975b93141ffbb570c15aaac022cd，Ubuntu独立工作树11-17。目标恢复后，沿用已批准执行链修复设计的原始数据绑定要求，将前轮只读审查发现落实为生产修复；不新增模型、奖励或物理行为。

## 实际改动

tabletop_native_teacher_actual_inputs.py:_rebuild 现在在构造common371之前调用原validate_mounted_observation(common)。新增1个导入和1个调用；由原named关节、基座/速度、authored Reset seed读取包重新派生values/mask，保持原归一化和单位。原SHA、时间、owner、rod41、参数、bootstrap/terminal绑定全部保留。

test_native_teacher_actual_inputs.py 在原test_rejects_bad_durable_input_binding参数化边界新增common_values案例。该案例同时改变bootstrap common.values[0]与传给Critic的common371，重算pack/pre_action/observation引用摘要，保留原source_packets，使测试针对语义校验而非哈希不同。

## 实际验证

- 修改生产前，新案例实跑1 failed/9 deselected（4.29s），原因为DID NOT RAISE ValueError。
- 修改后原owner文件10 passed（11.27s）。
- 相关model/actual_inputs/contact_sources三个文件49 passed（63.41s）。10项包含于49项，不相加。
- 已有真实Tripod8档案再次独立重建16份输入，逐包摘要相同；原种子8次sampled动作重放最大误差0。没有新Isaac数据、没有Teacher/PPO更新。
- 前轮历史Pinch种子测试失败仍是独立问题；本包没有改脚本、模型身份或该测试，不把它写成已修复。

复现使用现有/mnt/mydisk/xk/envs/dh116/bin/python，CPU环境来自tripod-contact-fresh8-core-v1.json的cpu.environment，CUDA_VISIBLE_DEVICES为空。完整命令、日志SHA、读取源码SHA见readiness；日志/真实原始数据/私有分析脚本留服务器。

## Antigravity真实协作

按antigravity-desktop-review在Antigravity现有DH116 Server Collaboration Verification R1会话发送Teacher-Resume R1，角色codex-dh116-ssh-reviewer。初次实读失败，实际错误是只读MCP引用已不存在的旧CodexPrimaryRuntime Node路径；没有将失败当通过。

仅修复本地MCP配置dh116-readonly-ssh.command，保留原文件备份及其他配置，未改权限/server/reader；实际initialize/tools-list/repo_info/read_source验证通过。Antigravity重新发现工具后，经其只读MCP实读并返回本包source/test、历史重放脚本/test及model五份完整SHA，与Codex SSH逐项一致。

Antigravity实读覆盖actual_inputs.py前110行（含新校验与_check_pack）、新增案例所在测试全文及历史脚本/test全文；它未指出本包补丁的阻断。Codex按调用方及原来源校验独立复查，兼容结论限于上述49项回归及16份真实输入，不能把审查肯定推广为全系统/原生资格。

它建议历史重放单独重建原model identity并进行完整模型/初始state双摘要核对。该方案另包实施；Codex将补原冻结commit文件字节验证，不能仅硬编码两个SHA、删门控或修改生产模型身份。R1访问失败和恢复后真实回复均保留私有SHA证据。

## 保护及剩余路径

本包不改变24D/371D、双RGB、原MPC/Teacher/Student、模型、关节约束、USD、材料/物理、30mm Lift/5s Hold/0.5mm Slip。R2与ForceProxy关闭，正式BC/DAgger和三抓法成功资格未开放。

下一包优先修历史Pinch只读数值重放与当前模型身份的分离；之后继续独立接近信号、Pinch真实接触来源、Power拳握载荷与完整Lift/Hold。修复离线审计不代表学会抓取。

发布前Registry回归60 passed（14.62s），与49项分开记录；528个保护文件、3个历史Checkpoint SHA未变。
