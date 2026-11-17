# 实际自由杆静置窗口与下一批预算依据（2026-11-14）

基线 `725c907edc99a8dd073da05a41557a64c9924470`。
只读解析历史原生Pinch，**新Isaac步骤0、PPO更新0**。没有修改任何物理参数、控制器、recipe、原静置长度或成功标准。

## 修改与验证

新增 `scripts/audit_dh116_rod_quiet_window.py`：
- `load_actual` 验证整个不可变101,334,727-byte账本SHA、全部JSONL完整性、总232个真实物理样本、唯一Reset；只提取前48个quiet样本。
- `extract` 绑定rod/实际关节post-readback的run/env/episode/reset/rod/physics/timestamp；节点速度必须来自RopeBatch，q/qd必须来自PhysX backend。接触partner、solver substep和normal force都保留来源，不拿目标值冒充实测值。
- `summarize_window` 验证480Hz连续时钟、每步12个solver ticks、60Hz action identity、零24D动作、无guard终止。逐关节区分rad/s与m/s，不能把17D的平移关节混进rad/s平均。
- stable_grasp保持null，native_reset_qualified/teacher_admitted/cross_grasp_admission保持false；没有凭静置速度授予完整抓取资格。

新增2测试文件；先10个RED，再CPU回归 **48 passed /0 failed /0 skipped**（含实际历史证据及24D回归），3.43s。
日志 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/rod-quiet-window-20261114/regression.log`
SHA `528ef5bcc62f517ca465f9110788a928ceb7e4325401a71fc2aa0537a4dcf650`。

## 实际来源

历史原生目录：
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-bilateral-owned-20261114-xizgjb7a/native-runs/flex.f8-bilateral-owned-pinch-native/attempt-w0i1tg3o/native`。
`physical.jsonl` SHA `d53b446425640fbbbf59ba952c52ec84ff952f0eca8b539131f9197bc899158d`。
这次原生已在以前完成并消耗预算；本包没有重跑。

只读分析输出：
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/rod-quiet-window-20261114-run01/result.json`
SHA `d283ee366cdb32ebce9967be73934e1fd3ea00baa3201ad604287aa8f3a572d2`。

原始默认rod几何z=0.308m，但该次**实际配对Pinch reset已采用独立supported-seed proposal z=0.305m**。
不能继续假设这次在3mm高处自由落下。实际前48步节点z均为 `0.3049999177455902 m`，table=.3m、radius=.005m；数值差异不能被描述成无误差或接触为空。
没有固定节点、没有固定root orientation，保留真实重力；原生当次table持续承托，不能称为自由悬空抓持。

## 窗口结果

每行均从首个测量物理步3开始；样本间跨度(N-1)/480，所覆盖完整控制区间N/480，分别保存，避免混算。

| 物理样本 | 时间区间 | 最大节点速度mm/s | 相对首个样本最大位移mm | table载荷solver子步 | 手载荷solver子步 |
|---:|---:|---:|---:|---:|---:|
| 8 | 16.667ms | 0.038329238 | 0.00000372529 | 96/96 | 0 |
| 16 | 33.333ms | 0.038329238 | 0.00000372529 | 192/192 | 0 |
| 24 | 50ms | 0.038329238 | 0.00000372529 | 288/288 | 0 |
| 32 | 66.667ms | 0.038329238 | 0.00000372529 | 384/384 | 0 |
| 48 | 100ms | 0.038329238 | 0.00000372529 | 576/576 | 0 |

左手最高手指角速度 `9.446207513974514e-7 rad/s`（finger32）。
右手最高 `4.1577586671337485e-6 rad/s`（finger31）。
右finger12最高 `1.3649122365677613e-6 rad/s`；位置峰峰值 `2.7939677238464355e-9 rad`。
全部逐关节完整统计在私有result，未上传原始账本。

## 预算含义及不能跳过的门槛

上一包新Tripod目标路径最少200步。保留48 quiet和24 hold需要274步，超过240。
历史数据支持**研究**新候选：2 setup +8 quiet +200 approach +24 observation = **234步**。此处24步observation仍是50ms诊断，绝非5秒Hold成功。
本包没有自动修改recipe，没有宣称8步足以适用于新Tripod，没把Pinch的低速度移植成Tripod资格。

后续实际代码必须先完成：
1. 同一新Tripod实例的首8步实际速度、姿态、q/qd、contact和clock检查；不符合即停止，不暗加静置步。
2. 三指独立required-pad配置、真实移动掌位guard；现有固定腕Pinch validator不可直接复用。
3. 新Tripod seed/geometry/action/validator/预算完整冻结，使用新指纹和standing批准；保留单实例、自然退出、失败不重试。
4. 对正SDF gap的最终候选，不把几何邻近视为载荷接触；后续仍须实际接触建立与R0/Reset资格。
5. 独立右手Tripod、Power继续保留自身阻断，不由本包解除。

## 复现命令

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m scripts.audit_dh116_rod_quiet_window --output <new-private-output>
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_rod_quiet_window.py tests/test_rod_quiet_window_actual.py tests/test_tripod_sdk_path.py tests/test_tripod_sdk_actual.py tests/test_tripod_sdk_required_contact_guard.py tests/test_dh116_deformable_bimanual_step.py
```

Teacher PPO仍未获准进入训练；没有成功池新增、没有BC/DAgger。CPU窗口统计通过、历史原生静置、完整抓取成功是三个不同结论。
