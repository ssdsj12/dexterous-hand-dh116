# Surface Teacher Student PPO MPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用已成功原生teacher采集可训练的surface示范，预热student并真实执行PPO，随后独立验证MPC执行优化。

**Architecture:** 主线程负责原生重放采集与训练集成；独立sidecar审查MPC。旧teacher原件只读，新文件使用独立版本和完整合同。示范监督与PPO rollout分开。

**Tech Stack:** Python, PyTorch, RSL-RL PPO, IsaacLab/PhysX, pytest.

## 最新优先级：补二指，形成多样式teacher

用户进一步要求沿约7 mm二指试验继续完善，不把数据集限定为三指。三指原生重放已取得2个成功回合、2542组351/24样本，BC及64环境2次PPO更新已验证，保留该结果；先补齐二指真实成功，再构建明确包含不同预抓姿的混合teacher/reset分布。未启动三指专用长训练。

- [ ] 回到AA（`native_fixed3_priority_20261031_aa`）的二指candidate 177，原峰值6.979 mm；检查有效指腹失联时序。
- [ ] 应用当前已验证的触点回中/完整向上参考及离地后指令保持策略，保持原质量、材料、0.2 Nm/.3 rad/s和物理联动。
- [ ] 先两个控制参数例实测；成功窗口必须实际净升30 mm/5 s并只有每手拇指、食指承力，不能用其它手指偶然接触冒充二指。
- [ ] 独立复核、导出二指原生成功轨迹，再通过普通TensorEnv补观测与动作配对。
- [ ] 二/三指示范分别标记抓法与初始姿态；物理/观测一致后才构建混合监督及多初态PPO，不直接忽略初始化合同。

### Task 1: 可审计重放与采集

- [ ] 在`tests/test_surface_demonstrations.py`加入动作来源验证、执行前观测时序、失败reset拒收和成功终止截取测试，先运行失败。
- [ ] 创建`tabletop_surface_demonstrations.py`，`load_teacher_actions(path)`读取已资格成功的24维源动作；`collect_replay(env,obs,actions,contract,output)`通过正常`env.step`采集，只导出独立成功回合。
- [ ] 在训练CLI增加显式surface重放入口，在创建算法前执行采集；新launcher复用smoke的物理/静态初态配置，不使用教师隐藏目标覆盖。
- [ ] 单Isaac重放AK动作、保存原始结果和真实成功证据；只有真实成功才进入Task 2。

### Task 2: BC与示范辅助PPO

- [ ] 测试351/24合同、非有限数据拒收、拟合误差下降、迁移不覆盖critic/optimizer、衰减权重。
- [ ] 新surface BC模块直接训练当前同构actor；独立actor格式加载，旧模式入口不变。
- [ ] PPO每轮用自身新rollout更新，独立示范小批次监督，逐渐衰减；保存完整超参、示范哈希和监督损失到checkpoint及TensorBoard。
- [ ] 原生短预算验证BC初始化后确实更新并保存，student单独评估；根据实际表现启动持续训练，不宣称单条示范保证泛化。

### Task 3: MPC执行优化

- [ ] 审查现有预测模型、滚动时域、约束及与24维接口的兼容性；记录可复用组件和缺口。
- [ ] student具备可评估模型后，显式接入滚动预测执行器，记录名义/实际动作和模型预测误差，保留原限制。
- [ ] 同初态分别评估student与student+MPC，只有原生真实抬升保持改善才宣称有效。

### Task 4: 收尾

- [ ] 测试、原生证据和限制写入报告，更新README/CHANGELOG。
- [ ] 新源码及训练结果校验哈希后备份本地；不推送GitHub，除非用户再次要求。
