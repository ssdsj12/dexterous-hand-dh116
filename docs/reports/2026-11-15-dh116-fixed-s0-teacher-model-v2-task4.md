# Teacher model51 CPU 子包：Task4 batch 修复报告（2026-11-15）

**当前验收范围仅为独立 model51 CPU 子包。真实 fixed S0 尚未绑定。**

同一个 unresolved model 可接收 synthetic CPU fixture S0 SHA=1…1 或 2…2，
model extra_state.fixed_s0_manifest_sha256 仍为 None。
非 null resolved_reset_manifest 没有正路径；当前构造器只接受未准入 CPU candidate。
这不证明真实固定 S0、phase-context-model 整体来源、正式 checkpoint 或 native 闭环。
这些整体门仍 blocked。后继代码任务必须实现
verified manifest → model identity → packet 一致性；本次没有自造 S0 接口或 adapter。

Base: 4894022e832098f879fa86825afef9e35c3448cf。
Branch: codex/fixed-s0-teacher-model-v2-batch-fix-20261115。
Tree: /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/fixed-s0-teacher-model-v2-batch-fix-20261115。
Evidence: /home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-model-v2-batch-fix-20261115。

原 4894022 worktree 和 QUALITY 失败证据冻结保留。本包仅修改原有的新
model_v2/test/report 三文件，profile_v2/config 保持原样；旧 v1 model/profile/context、
collector/checkpoint、PPO、物理、关节以及 success 30 mm / 5 s / 0.5 mm 未修改。
integration、最终 Isaac 入口和 registry/readiness 均未写入；未 push。

## QUALITY P2 与最小修复

原错误位于共享 _pack_v2：validate_context_v2 只验证 context 自身内部合同，
没有验证其 batch 与 common/rod 相同。common/rod 为 N2 时，内部合法的 context N1
能够 export 非法六键包；context N3 则在 _context_clock 索引中抛 IndexError。

本次仅在 _context_clock 之前新增三行 guard，明确验证 context values、valid、
metadata.identities 的 batch 均等于 common batch。common/rod 的 N 已由原节点、
mask、rod identity 和形状检查绑定。两种反例现在都在 pack 期间以
ValueError(context/common/rod batch differs) 拒绝，不等待 forward，也不 export
坏 batch；失败前后源 digest 不变。

新增 16 个用例覆盖 model/module 两个公开 fixture 入口、N1/N3 合法但跨 batch
不匹配的 context、valid/metadata identities 独立坏 batch、N2 正常 packing、
owned 输出和有限 Actor/critic。私有 raw 路径沿其实际已有的完整 expected identity
guard 拒绝额外 context 行；收紧后的负例保留完全匹配的首行身份/clock，
仅添加其余内部合法行，且所有源 digest 不变。

## 保留的 CPU 合同与准入边界

- INPUT_KEYS_V2 仍按原顺序保持六键：
  common371/common_valid/rod_nodes/rod_mask/task_context/task_context_valid。
- common 仍选择 66 项，另外 305 项为原始 NaN/false；rod 全 41×13；
  context51 的 index50 为独立 prelift duration，编码 102→64→64，fusion448、action24。
- 独立 PrivilegedTeacher 初始化、原 Tanh 分布、_MaterialEncoder、原 PPO 保持。
  新 MODEL/INPUT/CONTEXT/PHASE 各有 version/SHA，source BackendProfile 保持真实原标签。
- extra_state/load_state_dict 仍在载入权重前拒绝旧/错 identity。
  原 C05 exact-type 拒绝新模型；CPU state_dict 往返不等于正式 checkpoint 闭环。
- config 仍为 fixed_s0_pinch_v2，prelift_stable_s/prelift_scale_s=0.25 候选，
  resolved_reset_manifest=null，native_ready/checkpoint_adapter_ready=false。
- W4 真实 SourceAdapterV2/SnapshotV2/PhaseSourceWitnessV2 与 validator 未实现，
  ContextMetadataV2 没有已签发 witness 字段，context51 生产 constructor/validator 仍封闭。
  pack_native_inputs_v2 的精确签名不变：
  (pre_action,raw_rod,parameters,task_context,*,backend_profile,source_adapter,
  measurement_snapshot,phase_receipt)。
  source_adapter=None 报 full_v2_phase_source_adapter_missing；
  任意 caller adapter 当前仍报 full_v2_phase_source_witness_missing，
  不调用其伪造 validator，不给 typed receipt/caller True/native scope 以资格。
- 本包没有声称 66+533+51 数值槽的 650 项实际来源已通，未扩展 Student143×3、
  DAgger 或 Student own PPO。

## 验证与证据

所有 pytest 使用 canonical CPU 前缀和 PYTHONDONTWRITEBYTECODE=1：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q -p no:cacheprovider
~~~

- 原两个反例 RED：2 failed in 3.11s，pytest exit_code=1。
  N1 为 DID NOT RAISE；N3 为 IndexError。日志 batch-red.log。
- model suite GREEN：74 passed in 11.57s，原58+新增16，exit_code=0。
- 原冻结 direct-probes.py 直接在新 tree 执行：42/42 passed，exit_code=0。
  两个反例均 ValueError 且 source_unchanged=true；原 probe 未编辑。
- 指定相关六组回归：313 passed in 73.07s，exit_code=0，原297+新增16。
  路径为 test_native_phase_contract_v2.py、test_native_teacher_context_v2.py、
  test_native_teacher_model_v2.py、test_native_teacher_context.py、
  test_native_teacher_model.py、test_native_teacher_session.py。
- 回归后仅收紧 raw 负例首行匹配，定向补验 2 passed in 3.15s，exit_code=0。
- 原 Repository(root,evidence_dir=evidence).verify_protection()：528 源码/脚本/配置
  + 2 历史 checkpoint 通过。
- fixed-s0-unbound-limitation.json 记录同 model 接受 fixture S0 SHA1/2，
  extra_state 固定 S0 为 null，非 null candidate 拒绝和未实现 verified manifest 正路径。
- native=0、GPU=0、grasp 未验。未 push。
- 原 probe/log/report SHA 复核前后一致，见 frozen-before.json 与 frozen-after.json；
  原 worktree 的 HEAD=4894022 且 status 干净，最终检查见 manifest。

回归 log SHA256：b97f311ec482e5281f46c0725488cd10498580ca9d5d3f46c5eecefebf846c5f。

RED log SHA256：d9e4d97247f68ab128292acbd926fd27c6529fca3bb9d9bff980071ef9c7bfda。

修后原 probe log SHA256：3e446066ffc074cb0fea114c6af352532d13aa7bef8ca8d93656f5c1b0a5c229。

raw 补验 log SHA256：8ce35df22afa0df0231b1b950a9959987e910f49bc3e7de0e5414e122e1fd5d5。

原冻结 probe SHA256：e91903e4b6fef242b508247c5ee6e091f4fec341296de46686ad746437248b24。

SELF review：原两个反例、索引前 batch guard、公开入口、raw 实际 guard、
同批正例、source digest 和 S0 未绑定范围均已复核。完成前 AST/git diff --check
与原冻结证据复核记录于 evidence manifest。等待 root 针对新 commit 独立 SPEC→QUALITY，
本报告不替代独立审查；仅允许登记 model51 CPU 子包，整体门仍 blocked。
