# DH116 351D→371D Student Contract and Reward Ablation Design

## Goal

修复 Teacher 与纯 Student 之间的 24D Action/Observation Contract 不一致，并让 Student 在自己的失败前状态上获得闭环 Teacher 修正，同时分阶段验证 Hold/Lift 奖励。最终目标是让纯 Student 自己完成至少 30 mm 抬升、连续 5 s 稳定保持、相对滑移不超过 0.5 mm 的原生成功判据。

## Constraints

- 保留原 351D PPO policy、完整 PPO Adam 状态、训练计数和 CPU/CUDA RNG；原始 checkpoint 只读，任何迁移结果写入新的派生文件。
- 新观测固定扩展为 371D；新增 20D 必须来自真实 native 物理反馈，并记录字段顺序、时间对齐、归一化、无效数据策略和 contract SHA。
- 动作固定为 24D；禁止 26D 动作切片、补零或静默迁移。
- 物理参数、执行器限幅和原生成功标准保持不变。
- Teacher 专属 Support Servo 不得进入纯 Student 的实际执行闭环；Teacher 只提供轨迹或失败状态上的 24D BC 标签。
- deterministic Student A/B、Teacher replay 一致性和单元测试必须通过后，才能进入 PPO 续训；所有 reward 因素逐项启用和消融，不能一次性打开。

## Canonical 24D action contract

新增一个共享 contract 定义，训练、Teacher replay、BC、checkpoint migration 和 evaluation 全部使用同一份定义：

| Slice | Meaning | Physical execution |
|---|---|---|
| `0:12` | two hands × six active joint velocity commands | normalized `[-1,1]`, scaled by the existing per-joint velocity limits and integrated with the existing `dt` |
| `12:18` | two hands × world-frame XYZ translation velocity | normalized `[-1,1]`, existing wrist rate and bounded vector-step semantics |
| `18:24` | two hands × intrinsic XYZ rotation velocity | normalized `[-1,1]`, existing rotation rate, XYZ composition order |

Zero action retains the current policy-owned target. Force target fields remain measurements/protection ceilings and are never interpreted as extra action dimensions. The contract records `action_order`, units, scales, frame, rotation order, clipping, zero-action semantics and a canonical SHA256.

Teacher data is accepted only when its metadata matches this contract exactly. A mismatched action dimension, order, scale, frame, or contract SHA is a hard error.

## 351D→371D observation contract

The existing 351D observation columns remain byte-for-byte compatible. The appended 20D columns are generated from the same completed native control interval as the action:

1. `support_n` for the two hands, 2D;
2. total support margin, 1D;
3. bilateral support imbalance, 1D;
4. paired effective-pad fraction for the two hands, 2D;
5. ordinary-contact fraction, 1D;
6. maximum accepted pad depth, 1D;
7. fixed-anchor drift, 1D;
8. slip speed, 1D;
9. continuous native hold duration, 1D;
10. current native hold-window validity, 1D;
11. pressure-limited flags for the two hands, 2D;
12. mimic-blocked flags for the two hands, 2D;
13. per-hand executed wrist-Z request, 2D;
14. per-hand executed wrist-Z increment, 2D.

Each field has a fixed physical scale in the contract. Data must be finite, shape-correct, current for the control interval, and consistent with the native validity masks. Invalid or stale feedback is zeroed only for the diagnostic output and marks the environment invalid; the corresponding transition cannot produce a BC label or update PPO. A contract SHA binds the ordered field list, slices, scales, source state, interval boundary and invalid-data policy.

The migration creates a new 371D checkpoint. Actor and critic first-layer weights receive 20 zero columns; their corresponding optimizer moments receive matching zero columns; all other policy weights, optimizer state, counters, episode/success counts and RNG state are retained. The source checkpoint SHA, derived checkpoint SHA, old/new contract SHA and migration manifest are saved separately. The source file is never overwritten.

## Reward changes

The existing native success tracker remains the only success source. The following terms are separate named TensorBoard scalars and are enabled one at a time:

- `lift_delta`: net height increment only when effective opposed contact, ordinary-contact fraction and pad-depth gates pass;
- `hold_milestone`: continuous native hold progress, reset on a broken native window;
- `slip`: fixed-anchor drift and tangential slip penalty after the minimum lift onset;
- `contact_break`: explicit penalty for losing paired effective contact or breaking the native hold window;
- `support_margin`: measured upward support margin against object weight, with no fixed Teacher force target;
- `action_smoothness`: nominal action rate plus actual executed joint/wrist increment and projection/limit discrepancy.

The 30 mm, 5 s and 0.5 mm thresholds remain unchanged. Each reward-only checkpoint/run contains its reward profile, contract SHA, source checkpoint SHA, PPO/BC state, and native validation report. Results are compared by hold duration, drift, valid contact, measured support, contact interruptions, maximum net lift and native success rate; a higher total reward alone is insufficient.

## Closed-loop Teacher correction

The qualified Teacher pool remains unchanged and continues to be used as auxiliary BC data. During deterministic Student rollouts, record the Student's own pre-failure states, including contact loss, hold-window break, support deficit and slip onset. From each recorded state, invoke the measured-contact Teacher only to produce a safe 24D corrective action under the canonical contract. Save these rows as `student_failure_state_teacher_relabel` with source state, action contract SHA, reset family and physical diagnostics.

Failure-state labels are auxiliary supervision only: they are not success trajectories, do not alter the native success tracker, and do not enter PPO rollout log-probabilities or critic targets. The independent actor Adam step runs after PPO and samples from the qualified Teacher pool plus the failure-correction pool with an explicit, recorded ratio.

## Verification gates and sequence

1. Run unit tests for the canonical action contract, 351D prefix preservation, 20D feedback generation, time alignment, invalid data handling, migration state retention, reward term isolation and unchanged success thresholds.
2. Run Teacher replay consistency with one Isaac instance, matching the qualified 24D trajectory, reset family and native success semantics.
3. Run deterministic A/B with the original 351D checkpoint and the derived 371D checkpoint. Use identical resets and no exploration. Verify that the zero-initialized appended columns preserve the old-input policy action; record action hashes/MSE and all physical metrics. This is a migration consistency proof, not evidence that new observations improve performance.
4. Run deterministic Student failure collection and generate closed-loop 24D Teacher correction labels from the Student's actual visited states. Verify contract and physical provenance without training.
5. Resume PPO from the 371D derived checkpoint with the original reward profile only. Verify a complete checkpoint write and full state retention.
6. Enable exactly one reward term per isolated run, record TensorBoard and native results, and only then consider the next factor. Do not start the 4096-environment long run until the preceding stage's validation report passes.

## Evidence and release rules

Each stage stores a separate checkpoint, training/evaluation logs, reward configuration, contract SHA, source SHA and native validation result. README and CHANGELOG must state the exact commands, fresh results and limitations. No result may claim stable Student grasping until native episodes satisfy the unchanged success predicate.
