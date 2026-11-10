# Stage 7 — 统一 MPC 状态与原生只读接触采集

当前只完成统一状态、测量桥和默认关闭的采集入口；本报告不授予新 MPC、专家数据或 Student 成功资格。M0、旧 Teacher/PPO、3844D Actor、371D Critic、canonical24 Tanh Gaussian、原物理/材料/执行器和 30 mm / 5 s / 0.5 mm 标准保持不变。PR 继续 Draft。

## 实现范围

- `tabletop_bimanual_mpc_state.py` 使用不可变快照和逐字段来源，明确 source、unit、frame、timestamp、valid、env、episode、World/SDK/policy/reset 时钟、力的作用侧、速度原点及依赖。缺失字段必须显式 NA；缺键、拼写错误、非有限有效数值及混合时钟被拒绝。
- terminal/reset 记录可以归档，但不能进入规划。规划校验必须显式给定所需字段与最大年龄，不把归档格式正确等同于物理可用。
- `tabletop_bimanual_mpc_measurements.py` 复用原始 `TensorWorld`、原生接触读取、`TripodNativeEvidenceBuilder`、真实 backend target reader 和 body COM 数据；不调用 step、refresh、写目标或改变任务。
- 原接触证据增加中性的 `MPC_MEASUREMENT` 标签。它不证明固定手腕、无外力、A–E 实验资格或任务成功；原 A–E 的严格布尔声明保持。
- `tabletop_bimanual_mpc_capture.py` 在实际 60 Hz issued24 和 pre-reset 的每个 480 Hz callback 间建立关联，校验连续 8 子步、SDK/World 相对时间和每环境身份。首 terminal 后关闭该 cohort 槽；后续 reset/no-op 不能覆盖。实际 live 与首回合 cohort mask 分别保留。
- `--vision-mpc-state-capture` 默认关闭，仅允许独立原 constrained Teacher、真实固定 224×96 RGB-D、原生批量接触、480 Hz 和 sequence sidecar。排除旧模型 resume、M2、其他控制 wrapper；CPU camera fixture 不能声称 native。

## 物理语义与能力边界

| 字段 | 数据来源与当前限制 |
|---|---|
| 主动 q/qd、关节 margin、跟踪误差 | 每手具名 ACTIVE 关节；跟踪误差读取实际 PhysX backend target，命令缓存只作辅助诊断 |
| wrist/base 位姿 | 原控制器的 legacy palm 参考实际绑定 body/base link；保留真实 link 来源，不能虚构独立 palm_link 测量 |
| 速度 | SDK COM 速度通过 `v_COM + ω × (point−COM)` 换算到相应实际参考点；不能把 COM 速度直接当 link 或指腹速度 |
| 接触位置与载荷 | 本次 accepted raw 点按正法向 scalar 加权；scalar sum 与向量 resultant norm 分开，ON_HAND 与 ON_OBJECT 符号分开 |
| 整手 rejected/friction | 保留原始范围和归属，不能广播为逐指 accepted/friction 或合格承托力 |
| partner body ID | 无逐接触 partner ID 时明确 NA；cfg/filter/view 路径元数据不冒充已解析的逐接触对象读回 |
| qualified support、桌面/端支架反力 | 尚无完整可归属测量时明确 NA；辅助 filtered force 不授予离地或承托资格 |
| centerline/曲率 | 未绑定实际 shape 来源时 NA；刚性圆柱代理几何不能称为柔性线缆动力学或形变测量 |
| 硬件 torque/触觉 | 未确认硬件接口，明确 NA；PhysX 诊断不进入 Student 输入，也不冒充电机实测 |

SDK DATA generation 不能直接当作物理时钟。独立采集器只读包装真实原 `TensorWorld.refresh` 的一次返回，记录其源码 SHA、调用前后实际时钟与 DATA generation；桥独立校验 receipt。reset 额外刷新不会伪装为物理步，旧 receipt、裸 DATA 更新、错误 SHA/时间/序列均拒绝；提供 receipt 却校验失败时不回退到宽松路径。

无效 raw 中的 NaN/±Infinity 仍需保存。JSON 使用 null 占位和精确树路径、类别、captured Python float64 bits，区分原始 None，有限值不变；严格 DTO 仍为 NA/invalid。这不是原传感器 buffer 的独立逐位转储。

## 实际 CPU 验证

最初 freeze03 冻结源码 1479 文件，manifest SHA `9dfe287ff142805692b425343f98879a0a0321d5b10bdf9c36b70f1ce5a2a475`；四个原控制器源码 SHA 与 Commit 1 相同。该历史03验证在独立 `/tmp` 副本运行：

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 DH116_NUM_THREADS=2 \
bash /tmp/dh116_stage7_mpc_state_capture03_cpu/scripts/python.sh \
  -m pytest -q -p pytest_mock -p no:cacheprovider \
  --junitxml=/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_state_capture03/cpu_validation/pytest.xml \
  tests/test_bimanual_mpc_state.py tests/test_bimanual_mpc_measurements.py \
  tests/test_bimanual_mpc_capture.py tests/test_tripod_native_evidence.py \
  tests/test_tripod_raw_contact_capture.py tests/test_native_sensor_batch.py \
  tests/test_vision_teacher_collection.py
```

结果 **185 passed、1 skipped、58 subtests passed，8.17 s，exit 0**。skip 是该 CPU 验证进程未开放 CUDA 的 ownership 参数分支，不计为 CUDA/native 通过。validation SHA `f0446571f2af3732b7ccd8a7f833bc28e982d467eda0426527adbec0c7613aae`，日志 SHA `b8a021acbf8e8dc08669184ef8dbb68ca159a8be38cbd024c672122ac15bd683`；冻结源与 CPU 副本前后字节一致。

独立定向审查实际复现并关闭：旧缓存贴当前时间、tracking 使用 command cache、未核对 centroid、terminal 槽再次进入 bridge、receipt/reset 边界、少于 8 子步切换 policy、物理时间不匹配、初始化/close 失败阻断资源清理、invalid nonfinite raw 丢档。真实原 `TensorWorld.refresh` 方法也在 CPU SDK-shaped 存储上接入真实 wrapper/bridge/observer 做了边界 probe；这些仍然是 CPU 证据。

准备失败也保留：capsule01 私有打包脚本将两个控制器文件名写错，在源码 SHA 校验处退出，未运行 pytest/Isaac；capsule02 修正路径后 184 passed、1 skipped、58 subtests，随后发现并修复 nonfinite 归档边界，因此重新冻结并验证 capsule03，未把旧验证当作新字节证据。原有 reset lifecycle fixture 的缺 `roots` 失败在旧源码独立复现，未修改该历史 fixture 或原控制器来掩盖。

提交前仅清理 `tests/test_bimanual_mpc_measurements.py` EOF 的一个空白 LF，AST 完全一致：原 SHA `d49943d3b1a97831b62c7357438a0eb9eec6af99558f48bcfb4cbe0bbfbb1c3a`，发布测试 SHA `5f0919da5ced2b7b24d709d4beaaef0b5a58f870eb7fe4152580f997d67cfab2`。其余8个本次运行/测试文件仍与冻结测试字节相同；该测试另在独立目录定向执行 **33 passed /5.07 s/exit0**，不累加到185。两次从远端 home 用绝对临时测试路径启动的收集先因 `/tmp/fuse` 权限退出、未运行测试；改用独立工作目录后通过，原尝试保留。

```sh
cd /tmp/dh116_stage7_mpc_eof_validation01
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/tmp/dh116_stage7_mpc_state_capture04_cpu/src \
/mnt/mydisk/xk/envs/dh116/bin/python -B -m pytest -q -p no:cacheprovider test_mpc_measurements.py
```

## 原生验证

首次 `smoke180_v01` 在原脚本启动参数检查处 exit 2，未进入 Isaac：私有启动器提前创建了 child 输出目录，而原脚本要求该目录不存在。源码及原输入 SHA 前后不变，失败原件和旧启动器保留；新增反例先 RED，再修复为由 child 自行建目录，25 项私有启动器 CPU 测试通过。

随后独立 `freeze03/smoke180_v02` 在 Kit 启动后、世界/物理 setup 前 exit 2：Git 资产清单仍引用旧机器的 `/home/xk/...`。实际服务器资产已迁移，四个 USD 的 SHA 与原清单完全相同；失败输出和旧启动器保留，没有物理 rollout 或新测量资格。

`freeze04` 仅从旧已验证快照复制实际资产清单：1479 文件中只一个 manifest 改变，JSON 恰四个 `path` 字符串变化，所有源码和 USD/物理内容不变。新 manifest SHA `bdccbc8003fe9bd1d9cd486cd4bb605dc17df7b9f5e678710d1d32e6720d3356`；独立 CPU 副本重验 **185 passed、1 skipped、58 subtests，8.00 s，exit 0**，validation `7f3562ffbb44bd7cadf2c0dc433d8302918889d680f9b6e34915a4f9b0836b0c`。私有启动器现提前绑定并在执行前后核验全部资产依赖；新增反例 RED 后 32 项测试通过。

最新04快照的实际 CPU 命令如下，不能用03历史命令冒充04执行：

```sh
cd /tmp/dh116_stage7_mpc_state_capture04_cpu
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 DH116_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
bash /tmp/dh116_stage7_mpc_state_capture04_cpu/scripts/python.sh \
  -m pytest -q -p pytest_mock -p no:cacheprovider \
  --junitxml=/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_state_capture04/cpu_validation/pytest.xml \
  tests/test_bimanual_mpc_state.py tests/test_bimanual_mpc_measurements.py \
  tests/test_bimanual_mpc_capture.py tests/test_tripod_native_evidence.py \
  tests/test_tripod_raw_contact_capture.py tests/test_native_sensor_batch.py \
  tests/test_vision_teacher_collection.py
```

`freeze04/smoke180_v01` 已自然结束，**process/launcher exit 0**，实际 8 环境、180 policy step、180 批真实 RGB-D、1440 条监督记录、1440 个物理帧、11520 条逐环境状态，每环境恰 1440 条。采集用时 508.23 s，源码和全部原输入 SHA 前后不变。命令 plan SHA `0561d7fa70845e7ea85b386d7befa35f35febb16ab44f384d0c85ac65adf1c31`；单一全局非阻塞锁覆盖完整进程。

实际原生执行入口（私有 plan 保留完整原 Teacher02 参数及全部输入 SHA）：

```sh
/mnt/mydisk/xk/envs/dh116/bin/python -B \
  /mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_state_capture04/run_stage7_native_state_capture_v03.py \
  --execute --plan /mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_state_capture04/smoke180_v01/command.json
```

首回合 UUID `7a83008b-a74e-40ef-a4ce-63c7750d7019`，物理 UID 为 `UUID:env:0`。所有环境的 World step 为 1…1440，SDK step 为 19…1458；两个绝对时间起点不同，按原 1/480 s 相对增量核对。没有实际 terminal，8 回合全部为右截断；成功计数 0 不是 8 个完整失败，也不代表新 MPC、专家池或 Student 成功。旧 Teacher 的二指/三指来源各复制到四个 nominal 场景，不构成不同初态或正式 100 次配对。

已用独立 CPU 分析器读取两个 gzip 完整 EOF/CRC，并逐帧重建 DTO、核对 raw/issued24/clock/UID、accepted scalar 和原整手 force reduction；11520 条全部对齐，失败项 0。q/qd、wrist/base 位姿与速度、COM、质量、backend tracking error、joint margin、table clearance、accepted load 和 contact mask 均 11520/11520 有效。接触位置与切向相对速度有 7569 个有效但逐指 partial 样本、3951 个完整 NA；没有接触的位置不补零。指腹源几何法向、未绑定的中心线、接触 age、合格 support、被动反力、partner ID、完整 collision clearance、未接硬件信号等仍按能力缺口保留 NA。

归档摘要 SHA `3e4243c98f5ee144329617acb49f92d9e42246a5af733d30681040f3e0a53966`；capture report `66320bf4d2abd03dbae6dfac5143d8d028e3bd368630e80fa11837f0d58ae324`；owned outcome `cb686334f547d9651122f93bfa7dceaaf90772fb4a0292aa4eae22c207b2e3e4`；states gzip `35a61504a0b1248e1d422a41f7cc7a7c51cd6bb5cc3e69aae3ec930934401940`。完整原件留在独立 run 中，不提交 Git。

分析器独立审查发现 JSON 浮点指数溢出可绕过 `parse_constant`；新增 `±1e9999` 反例先 RED 后修复，8 项受影响解析检查通过，合法 tagged-null NaN 仍保留。修复只在 PRIVATE 分析器，不改 native 源码。原分析01保留，最新分析02已对同一闭合原件重新执行，exit 0、integrity passed、失败项 0，全部统计与分析01相同。最新 analysis SHA `75bae12809e52586042534b5f010897532219500ce333213fdc358d66f519b05`，分析器 SHA `4e22ef26252cca47566ef5e8ea55ae720f09216cf375f0ff0f2d79ce917c40d1`；格式审计不独立授予物理、成功、支撑或模型资格。

原 1900-step 录制预算的独立完整首回合作业已开始，当前正在执行；plan SHA `7f5537aea9683e719b85fe6e56ed9d6bf22df65168d7148aa775b7691a204b0c`。它仍使用旧 Teacher 和同一冻结测量器，用于真实 terminal/no-op 与跨环境 reset 验证；实际执行长度取原 source/tail/终止规则，不能预先声明成功或恰跑满 1900 次。新 MPC 尚未执行。

```sh
/mnt/mydisk/xk/envs/dh116/bin/python -B \
  /mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_state_capture04/run_stage7_native_state_capture_v03.py \
  --execute --plan /mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_state_capture04/full1900_v01/command.json
```

## 未完成事项与下一步

先完成上述原生测量验证及有效/NA 覆盖审计，再开始 H8 模型、目标、安全约束和求解器。缺失的支撑/反力/滑移不能补零来让 solver 可行。逐指恢复、载荷转移、共同暂停完整回合、二指/三指各 100 个真正不同配对初态与新专家池仍未完成。B3、长 PPO、默认 Teacher 替换和 PR Ready/合并保持关闭。
