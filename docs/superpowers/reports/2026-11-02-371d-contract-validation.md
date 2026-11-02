# 11-2：351D → 371D 合同与阶段验证

## 范围与不变量

依据用户的纯 Student 路线，24D 顺序固定为 left/right 各6个主动关节速度、left/right world XYZ 平移速度、left/right intrinsic XYZ 旋转速度。原速率0.3 rad/s、0.01 m/s、0.5 rad/s，480 Hz物理与60 Hz控制。追加20维只读实际物理反馈，351D输入前缀与原执行器、物体、碰撞、成功判据保留。

真实成功始终要求净抬升30 mm、连续5 s、固定锚相对漂移≤0.5 mm。Teacher成功、输入扩展一致性、离线BC误差均不计为Student成功。

## Checkpoint保全与已完成原生A/B

- 原始351D：`55d2a041cab933542f7c8f49b060e30d7f009467937396da1c4566c0b471c58a`。
- 独立371D：`e096b0f4d76e84db281674b9cfd91f3d3770ca09f8cd70cc5bda0a4a70a30033`。
- actor/critic首层新增列与相应Adam一阶/二阶矩为零，已有PPO、辅助BC Adam、RNG与计数逐项保留，原件未覆盖。
- 新主机 `a21725000819798016954020`，单实例串行，A=`student_a_351_v5`，B=`student_b_371_v1`，均在`/root/hand/runs/deployment/11_2_contract/evidence/`。
- 各2环境×1800控制步；nominal和executed动作逐值相等，最大差0。351D观测前缀最大差4.440892098500626e-16；新增反馈(1800,2,20)全部有限。
- A/B均0/2成功、最长Hold 0 s，最高净高度分别0.00273/0.01234 mm。这只证明派生保持旧策略行为。
- A合同SHA：`36f35a9521b16d69722e7adf783237b6f235d4d3d418a7dc26244f3443620703`；B合同SHA：`0e61f5f7fc5a012eeef5a9dd22f90cdf6a8f5d9b8507b67f87b256e08ca01e81`。
- 本地派生checkpoint与`migration_ab.json`另存仓库外`hand-deployment-a21702624634466304168937/11-2-evidence/`，SHA已核对。

运行器显式传入原pose、load-profile、初始化文件、stable_hold_v2和holding_v2。此前provenance/初始化合同错误的v2/v3及参数错误的v4都在各自输出目录保留，不计为策略效果。

## 独立审查与运行接线修复

审查固定提交26b2c4a，发现奖励注册、baseline多加合同字段、BC模式冲突、SHA最终化、DAgger标签绑定和支撑测量等问题。后续修复：

- baseline不新增奖励合同或积分状态；单个消融在原stable_hold_v2 BC/PPO上替换一个因子，注册独立TensorBoard项。
- lift只奖励合格接触下的新高度纪录，避免往返刷分；Hold里程碑每回合只支付一次；接触中断为离散事件，不随物理步长缩放。
- 支撑使用当前原生normal+可归属friction及实际物重，缺失即拒绝。滑移从真实杆点/手腕固定锚读取，失去接触后继续观测，不能静默使用0。
- end-support元数据最终写入后再生成和验证371D完整合同SHA；BC比较先验证各自SHA，再比较允许规范化的语义。
- 纠正器在Student动作前读取实际接触/Jacobian，只生成24D候选，不安装Teacher控制器、不写Student执行目标。失败前30控制步按episode和(tick,env)绑定，未通过独立native纠正验证的候选不能进入BC。
- 纠正池有独立消费者，共享保留的BC Adam，PPO storage保持纯on-policy；改变纠正源需显式阶段checkpoint。

## 本次测试与后续门槛

远端修复树：`/root/hand/runs/deployment/11_2_fixes`。

```bash
env PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 \
 /opt/conda/envs/hand/bin/python -m pytest -q \
 tests/test_surface_contract.py tests/test_surface_checkpoint.py \
 tests/test_surface_feedback.py tests/test_surface_feedback_wiring.py \
 tests/test_surface_action_proof.py tests/test_surface_reward_ablation.py \
 tests/test_surface_reward_runtime.py tests/test_surface_dagger.py \
 tests/test_surface_failure_teacher.py tests/test_surface_baseline_compatibility.py \
 tests/test_surface_imitation.py tests/test_surface_exploration.py \
 tests/test_surface_student_evaluation.py tests/test_surface_hold_imitation.py
```

最终远端项目树使用 `/root/hand/scripts/python.sh` 执行与本次改动直接相关的合同、反馈接线、Student评估、奖励消融和DAgger专项，结果为190 passed、24 subtests，5.94 s。整组 `tests/test_surface_*.py` 还包含既有历史基线用例；该组本轮为948 passed、48 failed、1 warning，未将其宣称为全量通过。Windows的`python`命令并未提供可信pytest执行输出，本次以远端明确pytest结果为准。

371D Teacher普通TensorEnv回放现已完成：1175控制步、2/2原生成功、2349组观测/24D动作。与351D Teacher示范动作和episode_ids逐值一致，前351维最大差2.22e-16，新观测全有限。输出`teacher371_v1`。这是Teacher回放一致性证据。

该回放在合同最终化修复前启动，因此其记录的SHA遗漏了最后写入的end-support元数据。`derive_dh116_surface_dataset.py`显式另存摘要修正副本，仅改合同摘要，不改原生样本/成功证据，不把元数据派生当成新回放。源示范SHA `029266635fc701d3cfbafa6cf31d93eae4cfa35e35ecaef576bc560864ccce64`；派生SHA `c845cec86171fe0cda885fbb1f842931f4356335217d021815132477ac8e5335`。

原371D checkpoint再显式派生单轮PPO冒烟入口`baseline_371_full_state.pt`，SHA `933e32e2f297bac9fe0d7df2b32484220cb23af84f62785ac0f99cd8e7ebf724`；所有非合同/示范来源元数据项逐项相等，包括两个Adam、RNG、iteration、timesteps、episodes、successes。命令以4096原有计数和371轮为起点，只允许达到372轮，不是4096环境长训练。输出`/root/hand/runs/deployment/11_2_fixes/evidence/ppo_baseline_one_update`。

完整状态续训冒烟已完成371→372，4096环境×32控制步，仅1次更新；从28147 episode/0success恢复，结束计数仍28147/0。PPO Adam有13组状态、BC Adam有6组，所有checkpoint tensor有限。新checkpoint SHA `b9639f15b017b36b7fa20db93f28f7a6968ebbb29b924a4dd7a532e45d5d38db`。PPO value loss0.00825、surrogate0.02639、BC loss0.001515，原生有效对向接触比例为0；没有稳定抓取证据。

该冒烟明确使用新采集的2349组二指371D示范池。原351D二指/三指多抓法示范及其模型/优化器原件仍保留；本次不把二指接口冒烟声称为完整多抓法续训。完整多抓法371D长训前还需以同合同补采三指371D真实反馈，不能给旧示范追加虚构零反馈代替原生观测。

随后串行启动第372轮纯Student确定性1800控制步及只读原生纠正候选采集，输出`/root/hand/runs/deployment/11_2_fixes/evidence/student372_dagger_native`。此任务不更新模型；评估和候选结果需另行读取核验。

每项奖励的native消融与独立native纠正验证仍未完成。未通过前不开始4096环境长训练。DAgger候选边界/投影有效不等于已证明动态安全；未取得独立native纠正证据前不能声称闭环纠正阶段通过，也不能声称Student已稳定抓取。
