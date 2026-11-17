# 三指首回合独立终止原生配对

使用冻结源码 `a3a6d1a2435b7eaf9df1899c4686c2f4f96e5dae`、同一个 update-0 Checkpoint 和原 `solid_cylinder` 场景。两个独立 Isaac 进程顺序运行，外层退出码均为 0。每路由二指和三指各一个实际原生槽位组成，启动器合同 32 步，诊断回调上限 128 步，原 480 Hz 物理、60 Hz 策略与全部保护照常执行。PPO 更新为 0。

| Teacher 路径 | 实际控制步 | 二指物理步数／首终止 | 三指物理步数／首终止 | 原生成功 |
|---|---:|---:|---:|---:|
| sampled | 40 | 269／`joint_limit` | 314／`joint_limit` | 0/2 |
| mean | 39 | 273／`joint_limit` | 310／`joint_limit` | 0/2 |

两路同一二指、三指完整初态 SHA 分别为 `c3c21d5d71604698115e87b2d53772e3c383721ad1a1d32d896d7729b7df2e76`、`af5ebbc2028cd294d4ba755f139e9d2f3c5a813c37bfdee6624ac01763f57f39`。Checkpoint SHA 为 `ec5831ac4de039e5e77e2ee4a3420497a0c103a2a5966a1a73662d4997966511`。配对读回检查确认首态、物理与控制器摘要、pre-action 371D、标定、seed 和世界／SDK 起始时钟一致。四个首回合 UID 各异。

## 首回合边界与动作审计

独立核对 1,166 条连续 480 Hz 样本。每条的原始 Teacher owner、实际 24D 命令及相机／状态身份与策略输入账本逐值一致。每手的 6 个主动手指、5 个被动手指、6 个腕部关节均保留有限 q/qd。两路的二指首终止样本各出现一次。其后的 6 个 sampled 策略步、4 个 mean 策略步均实际向已结束二指槽位发出零命令，标记为 `diagnostic_zero`，后续身份和观测为空。原生账本中每个槽位的最后一行是该槽位首终止。三指保持真实 Teacher owner，直到各自首终止。

## 三指右手 finger12

四个终止样本的唯一实测位置越限均为右手主动 finger12，canonical 动作索引 7。名义下界为 0.007912622765 rad，下保护线为 −0.012087377235 rad。三指两个终止样本的 SDK 目标均处于名义关节范围内。控制器与 SDK 目标差值不足 6e−10 rad。二指两路复现 v04 的相同首终止位置与速度。

| 三指末次物理样本 | sampled | mean |
|---|---:|---:|
| policy_tick，physics_step | 39，314 | 38，310 |
| 实测 q（rad） | −0.012199343182 | −0.012371548451 |
| 实测 qd（rad/s） | −0.0709040016 | −0.3121336699 |
| SDK 目标（rad） | 0.029628375545 | 0.025149773806 |
| 控制器目标（rad） | 0.029628374950 | 0.025149773582 |
| q 减 SDK 目标（rad） | −0.041827718727 | −0.037521322258 |
| 位置保护余量（rad） | −0.000111965947 | −0.000284171216 |
| 实际动作索引 7 | −0.0180750247 | −0.0087514818 |
| 终止接触前两通道（N，右手） | 0.0736，0.6252 | 0.3908，0 |

sampled 在末五步持续接近位置保护线，末五步均低于速度限值。mean 在 309 步记录 qd −0.3097391725 rad/s，尚在位置保护线内。310 步同时跨过位置保护线，速度幅值超过 0.300000012 rad/s 限制约 0.012133658 rad/s。原生 `joint_limit` 的位置审计归因与并发速度越限分别保留。四个终止样本的被动关节位置跨越、目标越限和观测到的掌位安全投影计数均为 0。

四个完整首回合的双手对向接触样本均为 0。三指最大净抬升分别为 3.45e−8 m 和 4.48e−8 m。现有证据定位了两个抓法共同出现的保护触发点及速度伴随现象，接触、驱动和耦合的物理成因仍待区分。两路均无成功专家数据，原有 30 mm／5 s／0.5 mm 标准适用。包络抓法和原 MPC 配对均在本实验范围之外。

## 证据与 CPU 回归

| 项目 | SHA256 |
|---|---|
| sampled 原生 480 Hz 账本 | `f9badf7c090a4a64cf9db74a40624d88e6449bb295c9247a5073524b13bd68b3` |
| mean 原生 480 Hz 账本 | `0c3b7bfffa181850e3c585f5cbf99adfc6d76c833aefad288c0e3f8fd32b364b` |
| sampled completion | `bc519c6a2863260f6c7e8ab4d92d2ebec628a49d5f086ea5ec4fb0a88fb821f5` |
| mean completion | `0c6ea841f093f1a870b27ddbd106c34bf8c07d959b7b92c0a8f35816c5abec5b` |
| 冻结启动预检 | `04f7a3e894fb3ac970a6face02091244b6b5716854ec889621faf08e105418c5` |
| 快照资产路径重定位记录 | `8edc94520425207d49c40a75db28548e664bcab371263bca922b4162d7a0461e` |

原始账本与核验程序保存在 Ubuntu 私有证据目录 `teacher-finger12-tripod-20261112-K1Inmj/review-pJ3PBW`。核验程序 SHA 为 `ca34279068dd49f75ff3d36056db892299485687429c3a02a6008c7a492d55a5`，指标 SHA 为 `2606afaf41760be0ed24e07b2ac9de827846244a84243b1c64feb13c591884ef`。Ubuntu 相关 CPU 回归 57 项通过，失败 0 项，日志 SHA 为 `493c2e38a41cd2668d7534603c6686bf716902a418b0e039997a2fbe64b6827d`。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= DH116_NUM_THREADS=2 \
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python \
bash scripts/python.sh -m pytest -q -p no:cacheprovider \
  tests/test_dh116_teacher_pair_diagnostic.py \
  tests/test_dh116_stereo_runtime_teacher_identity.py \
  tests/test_dh116_privileged_action_audit.py \
  tests/test_dh116_stereo_candidate_cold_trials.py
```
