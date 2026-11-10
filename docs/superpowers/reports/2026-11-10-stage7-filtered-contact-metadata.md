# Stage 7：实际 filtered-contact 路径元数据原生验证

本单元基于 `c443c7dd3cf86f4a2a68ab4a7970a2dcf85f19ce`，只扩展默认关闭的观测路径，不是新 MPC 控制器。M0、原 Teacher、3844D Actor、371D Critic、24D Tanh、原物理/材料/执行器和30mm/5s/0.5mm均保持；不启动B3/PPO，不更换默认Teacher，PR仍Draft。

本次对象仍为原 `solid_cylinder` 刚性代理；场景路径名 `Wire` 不代表已实现或验证柔性线缆动力学。

## 实现与限制

- `tabletop_bimanual_mpc_filtered_contact_provenance.py` 审核实际34个view、每手17个body的 `sensor_paths/filter_paths`、Wire root路径、SDK/source/stage/view生命周期及逐环境时钟。按实际行与env绑定，不按排序猜测，不从配置表达式补值。
- `tabletop_bimanual_mpc_filtered_contact_measurements.py` 只读原34个sensor和现有Wire view，逐getter立即取得owned副本并保留时钟/source括号。未新增World、sensor、物理step或设置写入。
- 新 `--vision-mpc-filtered-contact-capture` 默认关闭，要求原Teacher只读state+aux路径，并与Model、delivery probe分臂运行。有限预算和原source-chain/Shadow校验保留。首次异常、partial raw、terminal及逆序清理分别保留。
- **F0只审核路径元数据。** numeric actor/collider/partner ID仍NA；未审核normal/friction数值、count/start所有权、完整性、符号、承托或无支架状态。全部force/model/planning/expert/task资格仍false。
- 当前严格v1在任意previous clock已terminal后拒绝继续。实际CPU反例证明：一env先terminal、另一env仍live时，下一帧会被拒绝并保留partial/首terminal。因此本单元仅验证未跨terminal的有限窗口，不能当完整异步首回合采集已通过。独立v2逐env退役/显式generation收据尚未接线和原生验证；不得静默清空previous绕过。

## 本次实际 CPU

独立Linux冻结source10：630 passed、1 CUDA-unavailable skip、223 subtests，13.43s。公开观测范围独立复跑，排除尚未发布的objective算术48项：**582 passed、1 skipped、223 subtests，13.57s**，source与独立CPU副本字节前后未变。不是从总数相减得出的未执行结果。

公开范围的实际命令（在冻结source的独立CPU副本执行）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 bash scripts/python.sh -m pytest -q \
  -p pytest_mock -p no:cacheprovider \
  tests/test_bimanual_mpc_state.py tests/test_bimanual_mpc_measurements.py \
  tests/test_bimanual_mpc_capture.py tests/test_tripod_native_evidence.py \
  tests/test_tripod_raw_contact_capture.py tests/test_native_sensor_batch.py \
  tests/test_vision_teacher_collection.py tests/test_bimanual_mpc_scene_measurements.py \
  tests/test_bimanual_mpc_contact_provenance.py tests/test_bimanual_mpc_aux_capture.py \
  tests/test_bimanual_mpc_aux_integration.py tests/test_bimanual_mpc_articulation_measurements.py \
  tests/test_bimanual_mpc_contact_delivery_probe.py tests/test_bimanual_mpc_model_capture.py \
  tests/test_bimanual_mpc_model_integration.py tests/test_bimanual_mpc_experimental_options.py \
  tests/test_bimanual_mpc_delivery_integration.py \
  tests/test_bimanual_mpc_filtered_contact_provenance.py \
  tests/test_bimanual_mpc_filtered_contact_measurements.py \
  tests/test_bimanual_mpc_filtered_contact_integration.py
```

公开复跑保存了实际argv/cwd、上述环境变量、独立JUnit/log和SHA；命令另带指向独立输出的 `--junitxml`。私有finite launcher21项和新closed analyzer18项均由独立审查者实跑通过；后者root另跑18项/40.752s。它们全是CPU fixtures，不计Native试验。

## 本次实际 Isaac 与闭合后审

根目录 `R=/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_mpc_filtered10`。两个实验分别cold reset，8混合环境、原Teacher02动作、相同原参数；单实例锁、原输入/资产、完整冻结SHA均由原guard检查。全部正常退出0，source及原始输入前后未变。

| 独立目录 | UUID | 完整控制步 / 物理帧 | 每env状态 | fresh全局元数据 | 配对行 | terminal |
|---|---|---:|---:|---:|---:|---:|
| `filtered1_v01` | `9518e1cb-0fbb-4507-979b-701ede998fa0` | 1 / 8 | 8 | 8 valid / 0 invalid | 2176 | 0 |
| `filtered16_v01` | `6942d0ef-377c-40a3-88e3-6802d9bd16b7` | 16 / 128 | 128 | 128 valid / 0 invalid | 34816 | 0 |

每个全局快照为34×8=272配对行，未将整批数据再按env重复计数。两臂各8个回合均预算截断，**不报告0/8失败率，也不增加Teacher或Student成功**。这是原标称布局的副本，不是各抓法100个不同初态。

原生执行实际命令分别为：

```bash
python -B "$R/stage7_filtered_native.py" --execute --plan "$R/filtered1_v01/command.json"
python -B "$R/stage7_filtered_native.py" --execute --plan "$R/filtered16_v01/command.json"
```

此处python实际为 `/mnt/mydisk/xk/envs/dh116/bin/python`。两次均先分别执行显式prepare并绑定独立command.json；没有自动队列/重试。完整Teacher argv在各command中保留，末尾是state、aux和filtered三个开关，不修改原Teacher源文件。16步RGB-D采集耗时260.404742s，**不具备60Hz墙钟或solver deadline资格**。

进程完全结束后，匹配冻结source的独立analysis capsule执行：

```bash
python -B "$R/analysis_capsule01/artifacts/analyze_stage7_filtered_metadata_capture.py" \
  --run "$R/filtered1_v01" --output "$R/filtered1_analysis_v01"
python -B "$R/analysis_capsule01/artifacts/analyze_stage7_filtered_metadata_capture.py" \
  --run "$R/filtered16_v01" --output "$R/filtered16_analysis_v01"
```

两个后审均 `integrity_passed=true`、`metadata_binding_passed=true`、failures0：先检查退出/闭合/source和压缩文件SHA，再完整EOF/CRC；复用原Core/Aux的逐环境issued24、clock、UID、reset/no-op和terminal关联，并从每帧owned raw重新计算配对DTO。每个全局快照的343条metadata getter receipt及source/view括号也核对。后审通过只代表本范围归档完整性；没有修改旧NA或签发物理能力。

## 证据身份

| 证据 | SHA256 |
|---|---|
| source10 manifest | `2c74de18b5b7f621d63c13559d10046dfe9b20a9f7b72235db3dd95980c772a2` |
| 冻结CPU630 validation | `2993a646b3ae3a72f16f7ee6064766c04bcfabf26a39472a2078d0b2072aa3a5` |
| 公开CPU582 validation | `fe64e1e341b484a596e534d2c7233a7803ea0eade105ab4f89cd951f54d91e57` |
| filtered1 command / closed analysis | `a3ebf686b6b0f84cdc2185f16c38ea6047d2e720c4040f218466383508817ac3` / `2471627c28b9de878cacc40eae81c8520823db254e35b41fd4e052a062a7d425` |
| filtered16 command / closed analysis | `03933d9b7edf41b3c17f978ddb4c8d8583f0f1a5f61f31dd07a5ca8f6b5cf178` / `2f66540b21cf71d1b978984cdf8f48109ed8372af9dbc893bfb67f0b4b7ea562` |
| closed analyzer / analysis capsule manifest | `9b7b3aa8c20f0faabcb91b36a2dc6e480f6adfa4eb1cedc3b266f74a1e3c9e05` / `65daa0b0f0c6e52b120987b2f02864a07ca00f7399450ee29912426b00e71be1` |
| F0 resolver / fresh provider | `81aab33c45cf8255b13242f68de2dc4377a0cbcaadf3dcddb393596e6669a5ee` / `689df7697e6fb4bf7069e4eaa79657f75c6a4a00e6cc18923233062aea6887e4` |

原件和失败/截断证据保留在独立目录，本报告不上传raw、日志、artifacts或模型。

本次另对先前source09已通过的Model1/16封存数据完成当前矩阵算术审核，没有再启动Model Native：左右手分别64/1024个selected样本，17×17当前矩阵全部finite、raw相对不对称为0、数值求得谱为正；Model16对角缩放后condition的P50/P95/max左62.843694/68.293901/68.294249、右57.459217/62.022775/62.023224。这不是物理SPD、隐式motor分离、Mimic消元或未来候选动力学资格。工具先绑定passing proof和唯一provider SHA，19个CPU测试通过；私有工具SHA `d8da0820c112fc601667b369f61d15bdefe2755179068c0ae35de5915257f2f4`，Model16结果SHA `bbb161faa9f89f9b7d80c85e94d42defb77beae2d07d35a6e0c8f230f232fbfe`，原raw/NA不改。

## 后续门槛

先独立验证逐env终止/退役与generation变更，再读同帧实际normal/friction/matrix，复用原数值归约并检查各自所有权、SI、容量、正负对照与作用侧。不能把配置表达式、CAD/FK预测、总力负号或零报告当作真实逐指承托。H8预测/约束/solver/runtime、双侧恢复、LOAD_TRANSFER、共同暂停完整回合、各抓法100不同初态配对和合格专家/Student迁移仍未完成。
