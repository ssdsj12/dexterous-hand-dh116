# T1 训练入口与显式派生验证

## 当前阶段

独立修复分支 `codex/dh116-lift-repair-20261106`，基线为 `11-5` 的 b42f1aa。T1 核心已验证并发布为 bd2dc76；本次加入训练入口、完整恢复保护、显式派生和诊断日志。未切换正在运行的 v2 策略，未启动修复版 Isaac，不能据此宣称 Lift 退化已经解决。

新选项 `--vision-action-distribution tanh_gaussian` 使用 `dh116-vision-student-ppo-v3-squashed-gaussian`；默认 `gaussian_clip` 保持 v2。Actor3844D、Critic371D、物理动作24D、原物理参数与0.2Nm上限保持。新模式将保存的有界命令原值发送到环境和历史；非有限或超界命令在进入物理前拒绝，安全投影仍属于原有环境动力学。

新旧版本不能隐式互载。离线派生明确为 `warm_start_not_exact_resume`；相同 v3 合同下的恢复保留模型、Adam、LR、已有 Torch RNG 和计数，场景及视觉历史仍在启动时 cold reset，不声称恢复中途PhysX/solver状态。历史未保存 Python/NumPy RNG，不能补称已保留。

## TDD 与审查发现

集成最初 RED：14 failed / 2 passed；metadata补充 RED：2 failed / 15 passed；离线工具 RED：13 failed。第一轮集成及指定旧回归101项、离线13项转绿，根代理独立组合复跑114 passed（13.54s）。

独立规格审查随后发现两个P1：其一是完整权重/Adam形状、std范围及CPU RNG有效性尚未前置检查；其二是CLI只验证保存合同自身，未在启动Kit前比较已知静态运行参数。根代理独立CPU probe复现错误weight shape、Adam moment shape和std1均可通过初版guard。修复引入共享完整状态校验，私有CPU Generator验证RNG且不改全局状态；Adam参数、moments、LR一致性与合法flag组合均前置检查。已有CUDA RNG字节原样保留，CPU预检不冒充GPU原生语义验证。

静态预检复用原有纯函数，按新增/缺失/数值双向比较object、material、reward、load metadata，并检查solver order和support None/clearance/radius/length。默认solid cylinder的空metadata语义保留。前端先在内存生成真实reset JSON字节再比较SHA，错误输入在创建初始化文件、目录和AppLauncher实例之前拒绝；未知native读回仍由运行时完整合同再次验证。经过各轮新增RED/GREEN，指定集成套件281项通过，独立规格复审PASS。

质量审查还实测发现 `maximize=True` 会反转PPO loss的优化方向。根代理增加独立真实Adam上升/下降数值oracle、恢复前拒绝及离线源保护三个反例，先RED 3 failed / 3 passed，再仅增加最小的maximize拒绝保护。缺失和False保持兼容，其它合法flags保留；differentiable=True没有复现崩溃，未将其误报为bug。

最终根代理独立定向命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
 PYTEST_DEBUG_TEMPROOT=/tmp OMP_NUM_THREADS=2 \
 /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider \
 tests/test_vision_squashed_training.py tests/test_vision_squashed_options.py \
 tests/test_vision_checkpoint_state.py tests/test_vision_action_derivation.py \
 tests/test_vision_training.py tests/test_vision_resume.py tests/test_vision_options.py \
 tests/test_vision_fixed_integration.py tests/test_vision_palm_lift.py \
 tests/test_squashed_gaussian.py tests/test_vision_squashed_policy.py \
 tests/test_vision_squashed_ppo.py tests/test_vision_evaluation_trials.py -q --tb=short
```

结果：**327 passed，22.52s**。规格复审PASS；maximize补强后质量复审PASS，质量审查另独立CPU定向复验9项通过。原始RED/GREEN及复审证据保留在服务器/tmp和本机artifacts，不上传训练日志。物理相关验收仍须原生验证，不能由这些CPU测试替代。

## 真实1439轮断点的离线派生

CPU调用 `derive_action_checkpoint(source, output)`，源为T0冻结原件 `baseline_iter_1439.pt`：

| 项目 | 实测结果 |
|---|---|
| 源SHA256 | `d8d94faca22638584e10eb67ff55f9cfde904984c7232433ffd68f867ab5bda9` |
| 新派生SHA256 | `fe04af732c205f37180ad8dd003f1a5f36d4edfa3a538fed78067827163a9abe` |
| PPO / transitions | 1439 / 47644672 |
| 保存状态比较 | 除明确版本/合同升级外，原字段逐项一致，含policy、Adam、LR、已有RNG、计数及旧resume_migration |
| 确定性动作差 | 0，使用原run启动时真实视觉输入的全局环境0/511/512/1023，逐元素严格比较 |
| CUDA初始化 / 原生评估 | false / 未执行 |

输出在独立目录 `/mnt/mydisk/xk/dh116/runs/lift_repair_derived_1439_20261106T0532Z`，未写入原训练目录；源前后SHA一致。启动期输入仅作同权重数值parity，不能称为最新训练状态、固定初态原生评估或抓取性能验证。因分布升级保留同权重的确定性动作，未学习前的v2/v3确定性对比只能验证执行parity；真实早晚表现需用实际129和1439权重配对测试。

最终guard还用原训练进程实际启动的相机、初态文件、材质、load/reward和物理选项进行了正例验证，原件与派生均通过，两个SHA保持上述值，全局CPU RNG不变、CUDA未初始化。原项目1207个源码/资源文件重新校验全部与冻结清单一致。

1439轮原Gaussian保存的24维标准差最小0.292049855、最大0.299999982（原上限0.3）。这个快照不支持“方差已塌缩到下限”的推断；物理动作变化还受均值、clip和安全控制影响，不先盲目提高探索下限。

## 全仓库CPU基线

从T0完整源码压缩包解压到独立 `/tmp/dh116_lift_repair_cpu_baseline_20261106`，在原Python环境隐藏CUDA，并提供USD库路径和vendor/DeformX模块路径，实际执行：

```bash
dh116_usd_root=/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$dh116_usd_root:src:.:vendor/DeformX" \
 LD_LIBRARY_PATH="$dh116_usd_root/bin:${LD_LIBRARY_PATH:-}" \
 PYTEST_DEBUG_TEMPROOT=/tmp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
 timeout 900 /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -p no:cacheprovider \
 tests --continue-on-collection-errors -q --tb=short \
 --junitxml=/tmp/dh116_lift_repair_baseline_20261106.xml
```

基线结果：96 failed、4628 passed、49 skipped、28 errors、91 subtests passed，223.94s，exit1。错误/失败包括缺少open3d、历史绝对路径资源、未归档的archive/runs夹具、Python3.10没有hashlib.file_digest，以及旧接口夹具不匹配；没有为了改绿而删除测试或改动这些旧模块。14项collection error加14项setup error构成28 errors。原始JSON/XML/日志保留在artifacts或/tmp，不上传Git。

修复分支使用相同命令，在隔离源码目录完整执行并按test node对比，而非只比汇总数字。第一轮完整回归为4829 passed，随后静态合同/Adam参数补强后第二轮4903 passed；两轮均保持原96 failed、28 errors、49 skipped和91 subtests，第二轮用时240.87s。最后maximize拒绝补强后的完整结果为 **4906 passed、96 failed、28 errors、49 skipped、91 subtests passed，243.14s，exit1**。不能宣称全仓库所有旧测试通过。

最终JUnit逐节点对照：基线4801个节点全部保留，状态逐项一致；修复分支5079个节点，新增278个全部通过，没有新增失败/错误或既有节点状态变化。最终输出为 `/tmp/dh116_lift_repair_final3_20261106.{txt,xml}`，对照证明为 `/tmp/dh116_lift_repair_final_status_comparison_20261106.json`；本机artifacts保留副本。所有19个本次代码/测试文件与服务器被测副本SHA256一致，原项目1207个源码/资源文件均未变化。

2026-11-06 06:03 UTC只读检查原v2训练：1548/2048次完整PPO、40189 episodes、最慢环境32回合、原生successes0，PID150108仍运行。这是原训练状态，不能并入修复版学习效果。

## T2统计基础设施与未完成项

新增冷重置manifest与first-terminal统计模块，最初41项通过后，独立审查发现float32量化顺序仍会破坏四元数hash幂等性。根代理增加独立struct.pack oracle，先复现2 failed / 41 passed，再修为先float32量化、校验单位长度、确定符号；最终根复跑43 passed（0.44s），独立复审43 passed（0.37s），PASS。生产manifest每抓法至少100个不同float32写入物理状态，不能靠trial_id、origin或quat正负号增加数量。统计在物理reset前接收终止样本后冻结，缺失滑移保留null和coverage；物体高度、合格hold、原生success、原始对向接触时长和估算力矩饱和分别计量。

模块始终保留 `authored_unverified` / `native_validation_complete=false`。合成200条fixture只证明schema和统计，不能代替原生每抓法100次验证。尚未实现完整视觉原生adapter、真实readback认证、验证选模/周期保留、固定模型回归、奖励R0–R5实验、可部署传感驱动或修复版长训。这些工作按用户顺序及当前原训练自然结束2048的单Isaac门槛继续，不能跳过诊断。

269/482/1328历史权重仍缺失；实际129与1439只能诊断可用的两个时间点，不能冒充峰值策略回归。需要保留后续独立验证和模型，才能前瞻地检查训练退化。

## 文件差异

| 文件 | 变化 |
|---|---|
| `tabletop_squashed_gaussian.py` | 稳定概率、Jacobian、熵、KL及finite guards（上一核心提交） |
| `tabletop_vision_squashed_policy.py` | 一次输出tanh的独立视觉策略（上一核心提交） |
| `tabletop_vision_squashed_ppo.py` | latent/action存储、加权微批与诊断（上一核心提交） |
| `tabletop_vision_training.py` | 新旧模式dispatch、issued/history保护、v3日志和保存/恢复 |
| `tabletop_vision_action_contract.py` | v3合同、静态预检及新输出保护 |
| `tabletop_vision_checkpoint_state.py` | 共享完整policy/Adam/noise/RNG状态验证 |
| `tabletop_vision_action_derivation.py` | CPU显式warm start、源SHA与独占输出/sidecar |
| `tabletop_evaluation_trials.py` | T2 authored冷重置身份与first-terminal统计基础设施 |
| `scripts/derive_dh116_vision_action_checkpoint.py` | 离线派生CLI及无依赖help |
| 两个既有训练CLI | 显式新选项、转发和写入/Kit前的预检 |
| 八个新增测试文件 | 概率、PPO、训练、CLI、状态、派生、试次统计回归 |
| README / CHANGELOG / 设计、计划、审计及验证文档 | 实际阶段、证据和未完成门槛 |

上述八个DH116模块均位于 `src/go2_pvcnn/control/dh116/`；测试为 `test_squashed_gaussian.py`、`test_vision_squashed_policy.py`、`test_vision_squashed_ppo.py`、`test_vision_squashed_training.py`、`test_vision_squashed_options.py`、`test_vision_checkpoint_state.py`、`test_vision_action_derivation.py`、`test_vision_evaluation_trials.py`。既有CLI为 `scripts/run_dh116_surface_student.py` 和 `scripts/train_dh116_tabletop_gpu_ppo.py`。未修改原始视觉actor/resume模块、奖励/噪声配置、USD/URDF、材料或执行器参数，未删除旧测试；未上传checkpoint、archive/runs或本机凭据。
