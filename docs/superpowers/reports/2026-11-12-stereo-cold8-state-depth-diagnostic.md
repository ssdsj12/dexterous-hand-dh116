# Cold8 状态标签与双目深度辅助头持出诊断

本次在 Ubuntu CPU 回放现有 `solid_cylinder` 的原生采集。Cold8 清单的八个初态原本都属于 `test`，本次全部保留为持出数据。没有启动 Isaac、保存模型或改变训练资格。

## 状态标签和物理变化

使用 `read_candidate()` 复核 32 条同钟候选记录与 256 条 480 Hz 物理样本，分别校验原始 SHA、初态、episode、视图、标定和现有可见性门控。中心位置为原 371D 的 61～63，单位为世界轴上的物体中心相对双腕中心的米值。线速度为 68～70，单位 m/s。

| 抓法 | 初态 | Center XYZ 行 | 线速度 XYZ 行 | 四元数 / 角速度行 | Center 与 SDK 读回最大偏差 | 线速度与 SDK 读回最大偏差 |
|---|---:|---:|---:|---:|---:|---:|
| 二指 | 4 | 12 | 12 | 0 / 0 | 2.98e-9 m | 5.68e-15 m/s |
| 三指 | 4 | 12 | 12 | 0 / 0 | 5.96e-9 m | 5.68e-15 m/s |

同钟偏差检验的是标签和归一化。二指中心标签的 XYZ 跨度为 0.06825 / 0.00767 / 0.03213 mm，三指为 0.05825 / 0.00645 / 0.02506 mm。物体在同一回合内距首个有标签帧的最大位移只有 1.02e-8 m 和 1.11e-8 m，零速度基线的绝对分量 P95 为 8.27e-6 m/s 和 5.03e-6 m/s。样本不足以衡量动态线速度泛化。SDK 四元数留作物理参考，现有对称圆柱的完整姿态和角速度没有视觉监督资格。Deformation、Contact、Slip 和 Force Proxy 无合格标签。

将现有 `split_estimator_rows(seed=116)` 作为分组审计时，初态可分为 6/1/1 组。二指在 validation 缺席，三指在 test 缺席。该结果只用于检查分组行为，原本的 Cold8 test 身份没有变动，未进行状态拟合或正式状态评估。

## 归档 RGB 匹配误差

只比较左 RGB 匹配有效像素与独立原生实例 ID 筛出的 Optical-Z。覆盖率的分母是可见线缆的有效参考深度像素。下表的 5 mm 是光轴深度误差条件，三维正确点完整率仍沿用原评估合同。

| 抓法 | 参考像素 | RGB 匹配有效 | 匹配覆盖率 | 有效匹配 ≤5 mm | MAE / RMSE / P95 / P99（mm） | 稀疏样点 ≤5 mm |
|---|---:|---:|---:|---:|---:|---:|
| 二指 | 32,399 | 10,482 | 32.35% | 9,832 | 2.997 / 5.469 / 20.559 / 21.804 | 1,411 / 1,536 |
| 三指 | 33,067 | 11,431 | 34.57% | 10,799 | 2.907 / 5.382 / 13.812 / 21.824 | 1,444 / 1,536 |

## 深度辅助头持出复验

用已有训练诊断的固定配置重现两次 SGD 更新，seed 0、学习率 0.01、历史归档帧 21。两条训练初态来自 SDK 实测，训练帧左侧有 264 个粗网格标签，右侧有 0 个。右输出头参数变化为 0，右侧预测会随共享编码器更新变化。每个网络状态随后在同一组 32 条 Cold8 持出记录上评估，光轴深度只用作监督或评估参考。冷重置初态 SHA 和 source group 两类身份都验证了训练与持出交集为空。

| 抓法 | 视图 | 粗网格线缆标签 | MAE 前 → 后（mm） | RMSE 前 → 后（mm） | P95 前 → 后（mm） |
|---|---|---:|---:|---:|---:|
| 二指 | 左 | 32 | 93.618 → 23.094 | 97.838 → 26.255 | 128.380 → 36.611 |
| 二指 | 右 | 17 | 205.630 → 185.479 | 206.556 → 187.482 | 216.322 → 199.194 |
| 三指 | 左 | 32 | 73.816 → 27.448 | 78.109 → 32.845 | 105.787 → 45.028 |
| 三指 | 右 | 32 | 241.870 → 204.116 | 243.058 → 206.150 | 263.762 → 233.288 |

16 个双环境帧组均通过同帧、时钟、标定 SHA 及双侧实例文件回读。网格为 15×20。全深度范围的粗网格 P95 前 → 后，二指左 351.781 → 272.178 mm、右 365.837 → 354.845 mm，三指左 345.781 → 262.189 mm、右 382.962 → 360.471 mm。历史两步预算保持固定。误差仍远超 5 mm，当前证据只支持有界持出诊断。正式估计器需要更多独立动态初态，深度辅助头需要合格的双侧训练标签与独立验证集。

| 集合 | 抓法和来源 | 实测物理初态 SHA256 |
|---|---|---|
| 历史训练 | 二指，`db63bfc2-6edd-485f-a07c-0ff964b19fdb:0:1` | `098176add663f869e9e10d19dbd85ec4bf89160c96032faf4d5476a321a0eb26` |
| 历史训练 | 三指，`db63bfc2-6edd-485f-a07c-0ff964b19fdb:1:1` | `5521434f566fb235e0322e1fcd8ab6b1661aea997195456f51eccc910e59e479` |
| Cold8 test | 二指 0000 | `b5d01de45b730cafc4d14f177ada16f6139e33e5122aa6e84085c91b735cc0f3` |
| Cold8 test | 二指 0001 | `c1405ca1cab5623ff8c268048cf4e279bc1089ecf67da74043b5803763ac8d57` |
| Cold8 test | 二指 0002 | `7e995566f4165e291db4db51b73b967f15401c7335b0bb1ea4f8ee865fbe374c` |
| Cold8 test | 二指 0003 | `7a35d11cb42476d0cf552d7e159669c226b514e74d881870c9e9b410a7f395ca` |
| Cold8 test | 三指 0000 | `93f7bcc42b8e7c759a452a6b8054a9b901cb03c244aed624fcd5fdbdcb9a0852` |
| Cold8 test | 三指 0001 | `953ec1e2f95de30dbaad7a0a79dd18dcb6f72fe033560cd399e06222c56cd475` |
| Cold8 test | 三指 0002 | `5df3ecd6eed4009b0520a4651e836992472c92f79bf0f27d96d4eece7a080c59` |
| Cold8 test | 三指 0003 | `f93d6cf222d4c17f2ad4c7b69f2787bad6d0daaf4336a57e40f5c67f998bb65a` |

## 证据与复现

代码基线 `b2f5d84ff5e3f9d87d5cd24ba55a3f4edb6f8ca6`。双目标定 SHA `02fabb87c548cd982728dac5d6357ca08327258514269b163e880427b2556700`。候选 SHA `1366bf862643c0c7ecd629e785da27cdb4c5e01087dc23b04b74e8cf937678c8`，480 Hz 原始 SHA `b4d34bacf2dea7175d8339432de38e96c26d47da5105f353449102c25593a64b`。历史训练 proof SHA `fda6bb67d0d9a141baa2352a7b9b8a4cc4cd7f3e90a7aae8ec411f1269de4172`。

私有证据目录 `/tmp/dh116-cold8-state-audit-20261112-a1zDiM` 内保留脚本和结构化结果。状态审计脚本 SHA `11265fcc566b27b34278de46bcd2b3863fe7769a81e2c3b1793da10ef288fc8d`，结果 SHA `216f100cf199f1c8742d4cfb6d761cefaaf1842067bc2366e1c2cf3cf166996b`。深度诊断脚本 SHA `8be30786d9cbca9a1aa2a4167dca3543582327032fca1062b4795f66a63e2e2f`，结果 SHA `218fff95d4f2022a88db781311f3894ef79850f01388b90429f33bcad8e93891`。

```bash
evidence=/tmp/dh116-cold8-state-audit-20261112-a1zDiM
export PYTHONPATH="$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python=/mnt/mydisk/xk/envs/dh116/bin/python
"$python" "$evidence/audit.py" "$evidence/state_audit_recheck.json"
"$python" "$evidence/auxiliary.py" "$evidence/depth_auxiliary_recheck.json"
"$python" -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_state_supervision.py tests/test_dh116_stereo_depth_auxiliary.py tests/test_dh116_stereo_cold_candidate_admission.py tests/test_dh116_stereo_wire_segmentation.py
```

Ubuntu CPU 回归 55 项通过，退出码 0，日志 SHA `2b52d6c9d0d74e78dc42de10f214bf2fa4a4c49d042cd7a13bce777b1085f716`。本包的原生物理步数为 0，原生数据来自 Cold8 已完成采集与历史归档。
