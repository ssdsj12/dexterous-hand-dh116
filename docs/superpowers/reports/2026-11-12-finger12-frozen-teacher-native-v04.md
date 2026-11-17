# 冻结 Teacher finger12 原生配对诊断 v04

四个独立 Isaac 进程顺序运行并由外层 supervisor 读回退出码 0。每路在两个实际环境槽位发出原 24D 动作，场景和控制保持 480 Hz 物理、60 Hz 策略、原生 `solid_cylinder`。128 步为上限。任一首回合终止后，本批在该策略步结束时停止，保留原生终止子步和同批另一槽位的完整该步前缀。四路 PPO 更新数为 0。

| 实验 | 实际控制步 | 二指原生物理子步 | 三指原生物理子步 | 首回合终止 |
|---|---:|---:|---:|---|
| 4 步 sampled | 4 | 32 | 32 | 0 |
| 4 步 mean | 4 | 32 | 32 | 0 |
| 128 步上限 sampled | 34 | 269 | 272 | 二指 joint_limit，physics_step 269 |
| 128 步上限 mean | 35 | 273 | 280 | 二指 joint_limit，physics_step 273 |

## 同初态与原始动作核验

四路均使用冻结 update-0 `ec5831ac4de039e5e77e2ee4a3420497a0c103a2a5966a1a73662d4997966511` 和相同的实测初态。二指完整 SDK 初态 SHA 为 `c3c21d5d71604698115e87b2d53772e3c383721ad1a1d32d896d7729b7df2e76`，三指为 `af5ebbc2028cd294d4ba755f139e9d2f3c5a813c37bfdee6624ac01763f57f39`。四个进程的 8 个 episode UID 各自唯一，seed、控制器、实际 pre-action 371D 摘要和双目标定一致。首步前 native world/SDK 时钟读回为 `[0,18,0.03750000195577741]`。

逐行核对 1222 条环境级 480 Hz 原始样本与 pre-action 张量记录、首回合 UID、动作 owner 和实际发出的 24D 动作。每手每个样本包含 6 个主动、5 个被动手指及 6 个腕部关节的有限 q/qd。sampled 路动作等于实际采样，mean 路动作等于确定性均值，采样值为空。两个终止记录与各自原生 JSONL 的末次有效二指样本完全一致。

## 右手 finger12 越限证据

右手 finger12 对应 canonical 动作索引 7。两路唯一记录到的位置保护跨越均为该关节。运行时原名义下界为 0.007912622765 rad，硬保护容差 0.020000 rad，下保护线为 −0.012087377235 rad。

| 终止量 | sampled | mean |
|---|---:|---:|
| policy_tick，physics_step | 33，269 | 34，273 |
| 实测 q（rad） | −0.012462520972 | −0.012090511620 |
| 实测 qd（rad/s） | +0.1765247136 | +0.1663924754 |
| SDK 目标（rad） | 0.008516647853 | 0.007912621833 |
| 控制器目标（rad） | 0.008516647679 | 0.007912622218 |
| q 减 SDK 目标（rad） | −0.020979168825 | −0.020003133453 |
| 保护余量（rad） | −0.000375143737 | −0.000003134385 |
| 实际动作索引 7 | +0.1932881474 | −0.0624272525 |

mean 的 SDK 目标比名义下界低约 9.31e−10 rad，处于既有审计器的 1e−6 rad 目标比较容差内。原保护仍按实测 q 触发。两路终止时的 qd 幅值低于 0.300000012 rad/s 限制，逐关节审计无速度、被动关节位置或目标越限，无观测到的掌位安全投影。sampled 的该维策略 mean 为 −0.0906583890。mean 路关闭采样后同一保护仍被触发，采样噪声在该初态下不是越限的必要条件。

| 终止前四步及终止步 | 硬保护余量（rad） | 右手第一个有效载荷通道（N） |
|---|---|---|
| sampled，265–269 | 0.000997671，0.000775922，0.000324459，0.000346722，−0.000375144 | 0，0.2352，0.2188，0.2082，0.3748 |
| mean，269–273 | 0.000984470，0.000505444，0.000471485，0.000004610，−0.000003134 | 0.3517，0.3292，0.2887，0.3641，0.3075 |

这两个终止样本均有有效接触读回，双手对向接触均为 false，记录的保持时间均为 0 s。压力、接触与关节跟踪偏差的并发观测不足以判定物理成因。三指轨迹因二指首终止而截断，右 finger12 最小保护余量分别为 0.0169961 和 0.0113372 rad。本次没有完整三指回合或包络抓法运行。两路均无原生抓取成功，成功池新增为 0。历史 MPC 合同和来源依旧独立，尚无三路因果对照。

## 证据与回归

冻结源码提交 `65148f724fb9279557736c2d98a860831930a82b`。冻结源码清单 SHA `e0b5193c71d4ba8dfa8e7bc44f20de389e297cd66544f5ac1fe94a5852aef2da`，运行副本的路径重定位记录 SHA `9cd3aa7c925fe7f62c9e6a292b677adbdbdb24266f2ceb7d5477c17f32f8f9cc`。

| 原生账本 | SHA256 |
|---|---|
| 4 步 sampled | `3cf584ee8ab925df21111f1e4fa8847ecf8f8b62df5e56ca4df4614399561e95` |
| 4 步 mean | `db9095a61dd88dc2a6857456cf89217671e08fe9df2007b23642a2d7a1de18dd` |
| 128 步上限 sampled | `ee0afbb9e3d6c266a0b3e19c136d4b221eb60bb1ebdc54e17f5c424110f46308` |
| 128 步上限 mean | `565d6468532f412ad817cab3f2e45dba4fbae3cd9cf956a19a67435d10df5652` |

Ubuntu 私有目录 `teacher-finger12-pilot-v04-20261112-Ikp0J5/review-dhr2gy` 保留逐行核验程序和输出，SHA 分别为 `bca499e79c520ac1d873f64f67c56307b643f08baf129520d04bac23b190e43f` 与 `fef0fe09363c35122d5ede3bcb7edc1c4966a4df4a818e7ed9dc840bf674bb2c`。重新运行以下 CPU 回归通过 56 项，失败 0 项，日志 SHA `ebdfc28fcc0d5354ebc36f5bd3d8dcf982c64a7f4e5b88d0d8911de4063634c6`。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= DH116_NUM_THREADS=2 \
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_teacher_pair_diagnostic.py \
  tests/test_dh116_stereo_runtime_teacher_identity.py \
  tests/test_dh116_privileged_action_audit.py \
  tests/test_dh116_stereo_candidate_cold_trials.py
```
