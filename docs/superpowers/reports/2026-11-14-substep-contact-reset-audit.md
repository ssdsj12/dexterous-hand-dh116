# DeformX 子步姿态与 Reset 提交审计（2026-11-14）

本包只做 CPU 数学/时序回归和 **已有 234 步原生账本的离线重算**。没有新 Isaac、CUDA kernel、物理步或 PPO 更新。原始账本、消耗完的配方、旧控制器与二指成功路径不变。

## 发现与范围纠正

实际后端 `2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec` 中：
- `csrc/sdf.cu:37–64`：set targets 只写 target/teleport 标志；stageMotion 到下一次 begin_step 才处理 teleport。`teleport=False` 会覆盖前一 pending True。
- `sdf.cu:157–159`：add_mesh 会 commit 构造姿态并同步；`267–300`：begin_step / end_step 建立运动与提交末态。
- `iteration.cu:10–34`：12 个子步各用 alpha=(sub+1)/12 的位置与最短弧旋转插值，XY​​ZW 增量为 world 左乘。
- 原 `LiveEffectivePadSource.capture` 的 timing_scope “post_SDK_link_pose_held_during_12_solver_substeps”不符合上述实现。旧分类是步末坐标近似，不能作为严格正/负有效接触资格。

原共同 Reset 的 True 尚未生效就被第一步 False 覆盖。首步起点由构造时 add_mesh 的实际读回姿态推导；之后起点是上一目标的 end_step commit。这里没有 native mesh-current getter，不能声称逐位验证了 GPU 内部 pose。

真实账本中的构造/Reset 后 link-origin 平移差（mm）：

| 指腹 | 左手 | 右手 |
|---|---:|---:|
| finger14 | 65.435887 | 24.064560 |
| finger23 | 76.192149 | 36.081906 |
| finger33 | 48.881742 | 60.070297 |
| finger43 | 54.050569 | 76.831227 |
| finger53 | 80.245481 | 92.744360 |

这不是放大关节范围或调奖励能够修复的 Reset 时序问题。旧 outer=0、234 步完整与清理证据仍保留，但**不能授予 mesh Reset 对齐资格**。后续原生配方必须先修这一缺陷，不能复用已消费预算。

## 真实历史重算

同一 receipt、232 个测量行、2,958 个手部 raw pair；按每条 substep/node/mesh/episode 身份重算，未删除失败点：

| 指腹 | 旧步末候选 | source-derived 子步候选 | 分类变化 |
|---|---:|---:|---:|
| 左 finger14 | 300 | 289 | 11 |
| 左 finger23 | 0 | 0 | 0 |
| 右 finger14 | 276 | 276 | 0 |
| 右 finger23 | 312 | 307 | 5 |
| 总计 | 888 | 872 | 16 |

NumPy float32 的 FMA/三角函数不保证与 CUDA 逐位一致。因此重算仍是诊断候选，不授予 positive 或 negative effective contact，effective_contact_value 始终 NA。真实 `gap=phi-radius` 在 kernel 中存在但 pair record 未保存；不能用末态 rod node、合力或 surface proxy point 冒充原子步接触深度。30mm/5s/0.5mm 未改且未通过。

## 实际文件与函数

- `tabletop_deformx_substep_pose.py`：`pose_at_substep`、`DeferredMeshPose`、`verify_source_lock`、`classification_status`。
- `audit_dh116_substep_contact_frames.py`：`named_poses`、`summarize_candidates`、`run`，只读 SHA 锁定账本。
- 两个新测试文件覆盖 pending teleport 覆盖、shortest arc/world quaternion 顺序、substep+1、独立左右身份、异常输入、拒绝计数和资格隔离。
- 未修改旧 consumed source / 原始网格 / material / safety / checkpoint。

## 测试与可复现证据

先运行缺模块失败测试，随后 **22 新 +18 既有 =40 CPU passed，0 failed /0 skipped**（9.74s）。首次脚本实际回放还捕获 Python 索引展开语法与 mesh_names 所属层级错误，均保留日志并修正后完成整份真实账本；没有把失败写成通过。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q \
 tests/test_deformx_substep_pose.py tests/test_substep_contact_audit.py \
 tests/test_live_effective_pad_source.py tests/test_deformable_pad_region.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh \
 scripts/audit_dh116_substep_contact_frames.py --output <new-independent-output.json>
```

所有大文件保留 Ubuntu evidence，不上传 GitHub：
- 原 receipt SHA: `04ebfe2460b6366960d87cda6f36200f503072e4fdfa2abffd25bc8dfcf775c3`
- 原 physical.jsonl SHA: `9cec6ed43c06f23cf23b327a22009c7f1b3a98574de27883ba89c60b0bf255d4`
- 新 `projected-pinch-substep-reclassification-20261114.json`: `4955ce70bc491fb0e9908dcdecae936a5a7196525c1b76ca537aec59a0732597`
- 新 CPU log: `d997275587c176fb3a6b8405e581c41eebc79b70d11b796a7245a087046cb9d8`
- 新完整重放 log: `4e401d338cf7a39e66439edc2835c90dfd1c154c346266d3d46861f4a6188c28`
- 初始失败 log: `4786592aac2f139a326ea55d916a8de1334b0ee95c2b5cec5dfe041a23542b5e`

证据根目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence`。
四个编译源的完整 SHA 锁在新模块 SOURCE_LOCK 及新重算结果。实际后端二进制只读核对，未重新编译。

## 后继工作

1. 独立 zero-step Reset 提交方案，每次 episode 均处理；保留同11槽SDF、table/material及时间，失败 poison。
2. 新 source capture 增加真实 per-substep mesh pose / gap，才能严谨资格化有效接触。
3. 当前健康 getter native 准备保持 blocked；先解决 Reset 后再冻结新配方、单实例有界验证。
4. Teacher 新输入适配可独立继续，但训练 R0、真实 Reset、接触源及 rollout 资格尚未通过；不启动长训练。
