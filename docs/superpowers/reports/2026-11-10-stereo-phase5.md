# Phase 5：真正在线DAgger与独立聚合

状态：已实现并CPU测试Student访问状态上的在线Teacher查询、完整动作owner切换、append-only dataset和独立BC更新。没有Teacher资格、native DAgger原生实验、Student-only原生成功或promotion。

## 新增模块

|文件|责任|
|---|---|
|`tabletop_stereo_dagger.py`|collect_step与D0～D4 run，真正调用环境step的在线访问分布|
|`tabletop_stereo_dagger_dataset.py`|DAggerDataset transactional append、identity去重、failure-state隔离、exclusive存取|
|`tabletop_stereo_dagger_trainer.py`|StereoDAggerTrainer独立Adam，完整minibatch/epoch覆盖，不写PPOstorage|
|`configs/dh116_training/stereo_dagger_v1.json`|候选schedule、正式Teacher gate、分组holdout及pool边界|
|`tests/test_dh116_stereo_dagger.py`|8项CPU访问状态、owner、reset、aggregation与隔离测试|

## 在线执行合同

`collect_step(env,student,teacher,teacher_probability=0.,schema_only=False,...)`先clone唯一pre-action snapshot，Teacher使用该state的privileged371查询deterministic label，Student只看同state的SensorHistory。Teacher查询自身不推进环境，没有历史trajectory或MPC fallback。随后按每个env独立选择一个完整24D owner，再真实调用`env.step`；从不逐维混合或线性插值。

Teacher label只作为detached训练目标；p=0时始终由Student执行，Teacher即使输出不同动作也不会接管。可选env owner telemetry hook在step前写明owner，原生trace不通过比较action数值推断所有权。

默认D4 Student-only，候选schedule为D0=1、D1=.5、D2=.2、D3=.1、D4=0。D0返回`teacher_reference`、Student-owned count0，不算DAgger Student访问；DAggerDataset拒绝把D0 reference掺入visited-state pool。mixed每一条仍保存实际owner，Student样本计数从owner显式统计。

formal run在第一步前必须通过公共Teacher资格gate（checkpoint SHA、canonical action/state、完整原生proof及raw bytes复核），并来自明确native runtime路径；任一不满足就拒绝，不以CPUfixture或qualified bool绕过。schema_only=True仅用于CPU协议/算术测试，record与dataset固定notnative。

## Aggregation与优化

`DAggerDataset(source_sha256,contract=None,schema_only=False,qualification=None)`支持`.append(record_or_records)`、`.payload()`、`.batch(indices)`、`.save()`与`.load()`。每次append先验证整个candidate，只有无duplicate identity/clock/reset/contract错误才原子替换内部rows；拒绝操作不改变已有pool。对外payload为copy，不暴露内部mutable list。

失败/terminal状态可以加入Student访问分布，但`successful_demonstrations=False`，不能被成功BC admission使用。episode、reset、frame、time、physical ranges和previous issued action都逐行核对；终止后的新history只留当前真实帧，旧episode不允许污染标签。

`StereoDAggerTrainer`继承独立StereoBCTrainer；`.train(pool,batch_size=32,epochs=1,generator=None)`使用无放回permutation并覆盖最后不足batch，Teacher label detached。没有伪造PPO sampled action/log_probability，不触及on-policy storage。正式aux BC必须由root独立metadata/qualification wrapper授权；本模块的CPUfixture优化不构成训练资格。

## 本次验证

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_collection.py tests/test_dh116_stereo_dagger.py
```

联合最终结果：46 passed（Phase3 27、Phase4 11、Phase5 8），exit0，无warnings。实现前Phase4/5联合RED为16个module-missing失败；另对Phase3 force metadata-only漏洞先RED复现，再接公共proof verifier，真实缺证明路径coverage0。

CPU FakeEnv实际推进tick/状态，连续三步验证Teacher接收到每一时刻真实访问的pre-state、Student nominal确实执行、pre/post物理范围分开、mixed owner为整24D选择、D0不计DAgger、正式无资格不执行第一步、terminal-reset history清理、重复append/非单调clock拒收、Teacher参数与PPO storage隔离。回归也明确检查env telemetry收到相同owner列表。

本次只优化了小型CPUfixture minibatch作为测试；没有启动Isaac、GPU、长训练、native采集或写入真实model checkpoint。既有30个action/control/training源文件摘要仍按Phase0基线复核；没有改变控制器、reward、成功判据或旧checkpoint。正式30mm净lift、5s qualified hold、0.5mm正式slip和全部原生门槛仍待Teacher/Student独立证据。

尚缺原生证据：合格Teacher checkpoint、原生Student访问分布、独立初态DAgger轮次提升、各动作组/状态head持出误差、Student-only任务成功。当前只属于implemented/CPU-tested，不能标为native-validated/task-qualified。
