# WP3 Teacher R0 PPO 增量验证

基线 `ae307a55917519138b6da5179e8254ce2e9a7c8f`。本工作包复用原 `PPOTrainer`、Tanh Gaussian、371D/24D、R0 及原关节保护。旧默认配置和旧模型不变。新增受保护 Trainer 配置只通过显式 CLI 启动，不代表已授权长训练。本轮没有 push 或 merge。

## 实际修改

- `tabletop_privileged_ppo.py::PPOTrainer._optimizer_step` 为原 KL guard 增补 PyTorch CPU、已初始化 CUDA、Python、NumPy RNG 的候选回滚。事务边界是本 minibatch 的 permutation 和 entropy draw 之后。拒绝或异常恢复 policy、完整 Adam moments/step/lr 以及 RNG；不把下一 minibatch 的随机排列撤销。
- `PPOTrainer.update` 增加 `epoch_trace`。每个 epoch 保留 attempted/accepted minibatch、冻结全 rollout 上的 exact Normal(old||current) KL、样本加权的 pre-step Tanh MC entropy、24D 三组 log_std/std。Entropy 明确不是解析的最终 Tanh 熵。原逐 minibatch exact KL、实际 backtrack 候选和梯度保持。
- `tabletop_privileged_cli.py::train_teacher` 增加 128-step 有界完整 rollout 归档支持与 `fingers12 / translation6 / rotation6` 三组输出，原逐手六组也保留。允许的归档预算为 2 env、32 或128 steps、1 update。未放开任意长诊断。
- `tabletop_privileged_dataset.py::audit_expert_compatibility` 是只读审计。区分动作合同匹配、371D pre-action 数据存在和完整原生成功准入，复用既有 success verifier，不新增成功豁免。
- `privileged_teacher_ppo_guarded_v2.json` 为 opt-in 正式 Trainer 配置，rollout128、target KL0.02、原学习率3e-4和最多3次回退。R1/R2、辅助 BC loss 保持关闭。旧 `privileged_teacher_ppo_v1.json` 未改。

## MPC 先验数据审计

实际读取服务器 `stage6_sequence_20261109_v01/stage7_mpc_state_capture04/full1900_v01/native` 原数据，不以报告中“4/4”字样代替数据核对。

| 源文件 | 实际 observation / action | 结论 |
|---|---|---|
| all_executed_supervision.pt |10559×3844 /10559×24|动作合同匹配，缺完整371D pre-action标签|
| successful_visual_supervision.pt |4698×3844 /4698×24|动作合同匹配，不能直接用于371D Actor-only BC|
| successful_sequence_context.pt |4698条控制上下文、8子步q/qd/root/target和反馈|存在局部物理字段，未保存完整371D pre-action向量，不回算或伪造缺失状态|

两份 action contract 均匹配 `498cebedfe520cc2f6593b32776581c3da826ba95e4db441b7938d29176c55d9`。原成功 supervision SHA为 `664f30456a86a073a734be97ef56819787cf3c2664dd182ffdf667a32e0ec026`，全监督 SHA为 `b498ae8df9b80af8c5c431f4494233841cb400181de56ee01e05062581cc8cad`。旧3844D训练合同中的 `critic_observation_size=371` 只表示旧模型设计，不能证明文件存有371D样本。未执行 BC 初始化，下一步须在保持旧成功控制路径时同步重新采集当前371D和实际24D，重做完整episode与任务来源准入。

## 测试

先复现5项 RED，分别为 RNG拒绝/异常、epoch trace、128配置、128归档；新增专家审计再复现2项 RED。当前实际执行

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_teacher_wp3.py tests/test_dh116_privileged_teacher.py tests/test_dh116_privileged_cli.py tests/test_dh116_stereo_runtime_teacher_identity.py
```

Windows CPU **81 passed /5.63s**，冻结源码服务器 Python3.10/PyTorch2.7 CPU **81 passed /8.65s**。另执行原 indexed rollout/Student PPO 回归 **30 passed /17.41s**。两平台计数重叠，不相加。新 guarded checkpoint 测试验证下次 on-policy 更新的 policy、Adam 与所有 trace 在恢复 RNG 后完全一致。原同回合多次 policy checkpoint 绑定测试仍通过。

首次 snapshot v01 因另一工作包并发修改文件触发 SHA保护，未运行。冻结 v02 1259文件后才启动 native，源包 SHA `f2b4cbae35c38e732b014de8d621fa94173194233286d95910dcce6b04596ae1`，manifest SHA `098903054c81306af0912b5372157b079e360be55776b0bb658dbbc282477a5e`。本工作包的5个代码/测试/配置文件与执行快照逐字节相同。

## 新原生有界实验

唯一 Isaac 实例通过现有 `/mnt/mydisk/xk/dh116/scripts/python.sh` 启动，GPU3、renderer activeGpu3、480Hz物理、60Hz控制、seed116。独立目录 `/tmp/dh116-wp3-teacher128-v02/teacher_diagnostic`，2环境、128步、1次更新。Teacher 随机动作采样，未运行评估或成功示范采集。继承任务为 `rigid_proxy/solid_cylinder`，不是Cosserat柔性线缆。

```text
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
bash /mnt/mydisk/xk/dh116/scripts/python.sh /tmp/dh116-wp3-teacher128-v02/frozen_entry.py script train_dh116_privileged_teacher.py --config /tmp/dh116-wp3-teacher128-v02/snapshot/configs/dh116_training/privileged_teacher_ppo_guarded_v2.json --num-envs 2 --updates 1 --seed 116 --record-diagnostic-rollout --enable-experiment --execute-native --output /tmp/dh116-wp3-teacher128-v02/teacher_diagnostic
```

进程退出0，采集及更新耗时189.282s。保存256条policy transition与2030条存活480Hz记录。18个terminal后no-op未计入物理样本。七个实际episode UID，其中五个以joint_limit结束、两个仍开放，原生成功0。两个起始环境与其reset不构成七个独立物理初态。

| 抓法/env | 终止policy tick / physics step | 完整终止数 | 结束时开放回合 |
|---|---|---:|---:|
| 二指/0 |31/250，67/540，104/835|3，均joint_limit|1|
| 三指/1 |37/301，97/784|2，均joint_limit|1|

每条实际发出动作与相同env/tick的保存随机样本逐值一致。所有pre-action身份、checkpoint SHA、calibration SHA、frame、UID均匹配。terminal样本保持且无后续no-op覆盖。GAE重新计算逐值一致，五个terminal的bootstrap均0，reset状态未跨终止估值。物理时钟最大差 `8.69e-10s`。七条 policy epoch事件对应七个回合首次使用同一采集策略，不是七次PPO更新；多更新身份本轮仍仅由CPU测试验证。

| Optimizer epoch | 尝试/接受minibatch | 最终全rollout exact KL | 样本加权pre-step MC entropy |
|---|---:|---:|---:|
|0|8/8|0.010398866|-21.9866905|
|1|8/7|0.019917488|-21.9863780|

共接受15个minibatch，480个样本次。第16个候选经有限回退仍越过原0.02上限，回滚并停止。最终KL **0.019917488**，不是提高阈值所得。负熵是连续24D分布微分熵，不能解释为抓取评分。

原生动作 clipping=0、R0 reward difference=0。真实 SDK target、pressure mask及掌心安全投影前后数据保留在480Hz原始行。原控制器仍未暴露逐子步的未约束关节目标，因此 `joint_safety_projection=null`，不能以 pressure mask冒充完整关节投影测量。

最终 checkpoint SHA `c1735e5f112b7f10ff4dacf00b2ac547728cea82a08d5b051e5b78cca84a9999`。原生生成的模型在独立 Python 进程回读，policy、Adam、计数和 policy version 均相同；此次回读不是第二次原生 rollout。本窗口总仿真时长约2.13s，不能完成连续5s成功验收。KL达标只验证优化器受限更新，Teacher仍没有本轮成功抓取或成功专家资格。

## 证据与阻断

私有证据存放于 `artifacts/stereo-wp-20261111/teacher`。原始数据、checkpoint与日志不加入Git。服务器验证后独立下载18文件逐SHA检查，并在Windows再次检查原生256样本、2030物理行、GAE和终止一致性。指标摘要和代码SHA见 `docs/contracts/stereo_wp3_teacher_20261111.json`。

| 证据 | SHA-256 |
|---|---|
| native_physics.jsonl |470e8c5d0f24b2cf97c1664ea26dbd566aa58969677d0b3f64660ff22e2d6ffc|
| diagnostic_rollout.pt |7de6e75586a8a8f4d673062f0f2c795cad451ddee24fa76781bd81dff971a4ab|
| iterations.jsonl |374886ace58cf79c4bd99249cd40b6eba1a44a0038fa4f2cdecd513deddf9dcc|
| evidence_v02.tar.gz |49dedfdc6374dc011961adde6f3f76491f78b47d408279e3d840921f36df582e|

下一阶段仍需原成功MPC路径同步371D/24D重采与独立资格、稳定Teacher完整成功回合及更长但明确授权的验证预算。R2、Force Proxy、正式BC/DAgger和Student训练保持门控。旧模型、旧成功案例与失败实验均保留。
