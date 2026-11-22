# Teacher真实输入与动作重放核查：两个离线审计阻断

基线：0e713240b38142b09afb4477e6ad120f18d4ca0b，Ubuntu开发分支11-17。本轮仅服务器核查：没有恢复Antigravity桌面，没有修改生产源码或测试，没有启动Isaac、Teacher查询或PPO。使用已有原生档案进行CPU分析，不能写作新采集。

## 已由实际记录证实的部分

原Tripod contact-fresh8的training_events.jsonl SHA为984c0c241daa098b01e61415305f3a5e98a887c0c46d9699feef4e8d1445e151。独立原输入审计器核验1次prime、64个物理样本、8个决策包和8个bootstrap包；16包逐个使用原_pack路径重建，摘要完全一致。额外逐个调用validate_mounted_observation，原16份本体数据均通过：66项有效、305项NaN/false。

按记录的初始化种子2026112001、探索种子2026112002构造原contact模型，并核对初始完整state_dict摘要与记录相等。8次sampled_action24重放最大误差0；动作并非恒零，各次最大绝对动作约0.23～0.36。

真实8帧中，66项公共输入均有变化；杆位置最大分量变化7.08696mm，杆节点最大速度69.6987mm/s，材料站位仍为0～1。首末帧latent mean最大分量变化0.0191938。Actor的common/rod/context分支激活均非零。首末两帧对rod position/velocity等输入的局部Jacobian非零，因此可排除该cohort“Actor完全没接到杆状态/本体输入始终冻结”的简单解释。

局部梯度是数值敏感度，不是物理可达方向、有效接近、抓握或Lift。单初态8决策的初始随机策略不具专家资格，也没有验证相机或Student。

## 发现1：独立归档输入审计没有重新检查关节来源重建

tabletop_native_teacher_actual_inputs.py:_rebuild（34～46行）从pre_action.common.values直接构造common371，_check_pack（49～98行）检查内部摘要、原杆/边界/参数、时钟和打包结果，但未调用已有validate_mounted_observation来重建关节来源。

负对照只在私有内存副本中将第一个决策的common[0]增加0.01（归一化单位），对应修改输入张量并重算pack/reference哈希，原source_packets不变：
- validate_mounted_observation明确拒绝：mounted observation mask/values differ from source reconstruction。
- 独立audit_teacher_input_sources仍接受。
- 原档案SHA不变，负对照不写入训练/成功池。

这是独立离线审计边界的缺口；不是证明实时pack_native_inputs接受错误关节值。实时路径在tabletop_native_teacher_model.py:256已经调用该来源校验。外层冻结的原档案SHA也会拒绝擅自改动原文件；内部一致哈希不能替代语义来源重建。

下一次实施的最小修复：归档重建入口调用原mounted来源校验，增加上述负对照的失败测试，并保留全部clock/owner/raw物理绑定。不能仅增加另一份自报哈希。

## 发现2：历史Pinch种子回归依赖当前源码身份，现已失败

相关4个CPU测试文件共50项：49 passed、1 failed，68.17s。失败为tests/test_dh116_pinch_seeded_actions.py::test_rebuilds_nine_real_action_samples_in_declared_seed_order；实际错误为reconstructed initial policy weights differ。没有将失败改成skip、放宽SHA或写成全部通过。

该档案记录HEAD af01c4f72d9686f79b417bfe603b1192b9cc83eb。只读git show核查后，当前模型与历史模型的依赖元数据差异为tabletop_native_teacher_model.py及tabletop_native_teacher_context.py两个文件。仅在私有候选字典中替换为该历史commit的真实依赖SHA：
- 重建完整model identity摘要与档案collector_origin一致；
- 当前初始化数值张量配合历史依赖元数据的完整state摘要，与档案初始state摘要严格一致；
- 9次历史动作数值重放最大误差2.98023224e-8，小于原1e-7数值容差。

此证据把故障定位到“历史档案使用当前源码身份重建”的路径，不支持“策略数值权重已损坏”。没有改model._extra_state，没有调用load_state_dict接纳旧身份，没有改变生产strict gate。原pytest仍是失败，取证重放不替代正式修复。

首次私有取证脚本把collector_origin.policy.model与model_identity_sha256直接比较；源码collector.py:174实际使用_digest(完整model_identity)。已保留v1并用v2按真实规则核验，不能混淆两种摘要。

下一次实施应分离当前模型加载资格与只读历史数值重放：历史重放从真实冻结commit验证完整源码身份，保留strict当前模型/Checkpoint准入；不能将旧身份改名为当前模型或简单删除摘要比较。

## 力反馈输入范围

common66选取实际主动q/qd、mimic残差、掌位姿/速度和腕角；rod41x13包含世界坐标、速度、切向、曲率、材料站位和应变。context50有左右required_contact_fraction和bilateral_contact_duration。

当前这组Actor输入没有连续force/wrench幅值；接触奖励和保护另外读取真实反作用力。这是当前输入设计的事实，不等于“没有力反馈”，也不自动证明必须增加输入。若下一版需要基于夹持载荷调节，须独立定义合格力源、单位、坐标系、validity、归一化和Checkpoint身份，不伪造NA。

## 状态与下一步

本工作包记录blocked：归档本体语义校验缺失、历史种子回归失败均待修复。未修改生产代码，按用户本轮服务器核查范围保留取证，不能自动把后续修复写为完成。

优先修离线审计语义和历史重放身份，再处理独立接近奖励候选；Pinch独立接触来源、Power真正拳握接触、完整Lift/Hold仍未完成。无新原生预算消耗。保护文件与历史Checkpoint发布前逐项核验，原始档案、私有脚本和日志不上传GitHub。

发布前Registry回归60 passed（15.38s），与49 passed/1 failed分开记录。528个保护文件及3个历史Checkpoint SHA未变。PR#2仍Draft/open/unmerged。

## 最小语义校验候选的私有CPU复验

后续仍限服务器核查，基线5f82e0a3733d208ad1b28ff4745e8be2c4b68823，没有修改任何生产源码或仓库测试。

先在服务器私有目录创建两个边界测试：真实原始档案继续通过；改变归一化common[0]并重算全部内部引用摘要、但保持原source_packets时必须拒绝。原审计器实跑1 passed/1 failed（9.06s），失败为DID NOT RAISE ValueError，符合已发现缺口，非导入/夹具失败。

随后只在独立pytest进程的临时插件中包装原_rebuild，在原逻辑之前调用既有validate_mounted_observation。未添加生产flag、未替换原数据、未修改任一SHA/clock/owner/物理校验。进程退出时恢复引用，磁盘源码SHA始终未变。

候选运行2项私有测试及test_native_teacher_model.py、test_native_teacher_actual_inputs.py、test_native_teacher_contact_sources.py三个现有文件，合计50 passed（66.35s）。其中48项与上一轮回归重叠，不能与49 passed/1 failed相加。存在1条PytestCacheWarning：跨私有目录和仓库运行使公共rootdir为/，pytest不能创建/.pytest_cache；不是验证错误，完整日志保留，后续应显式设置私有cache_dir。

这个结果仅支持“调用现有来源校验可在保持真实档案可读的同时拒绝该负对照”。未证明所有来源欺骗都能检出，也不替代外层冻结原始档案SHA。候选没有合入或部署。上一轮历史Pinch正式测试失败仍未修复，Registry继续blocked；没有将50项候选通过写成生产回归已全部通过。

生产实施时只需在归档重建入口补语义校验，并把本负对照并入现有owner边界测试；历史模型来源重放应另包修复。生产修复、原生和训练都不属于本轮仅核查范围。

本次候选报告发布前Registry回归60 passed（15.62s），单独记录；528个保护文件和3个历史Checkpoint SHA未变。PR#2保持Draft/open/unmerged。
