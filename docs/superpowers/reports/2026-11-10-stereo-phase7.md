# Phase 7：逐回合统计、原始证据与晋级门槛

本阶段新增隔离的统计/验收软件、配置与CPU测试；初始代码开发冻结时native作业数为0。下文早期native0描述均指该开发阶段，随后由root显式执行的实际640×480采集及reference v2修复见文末附录。没有通过任何CPU fixture授予Teacher、Stereo、力学来源、Student或部署资格。当前NativeBridge记录的是runtime capture unqualified，force/support仍NA，因此正式任务门槛仍拒绝。所有原物理参数、原控制器和原成功算法保持不变。

## 实现与使用边界

`tabletop_stereo_evaluation.py` 提供 `evaluate_records(records,provenance=...,evidence_artifacts=...,min_independent_initials=100)`、`aggregate_episodes(...)`、`wilson_interval`、`select_best_checkpoint`。输入是逐480Hz原始record与episode_start，不接受跨环境拼接的peak替代回合。结果保留每个owner/style和episode，分开Teacher、MPC、BC、DAgger、Student归属；包括Lift P50/P90/P95/max、正式Hold/slip及coverage、contact retention、failure breakdown、成功率和Wilson95%区间。

实际物理初态hash算法已经移到公共 `tabletop_stereo_contract.physical_initial_sha256`，evaluation仅import/reexport，Phase1/2不反向依赖Phase7。字段是实测object_pose7、两掌pose2×7、active q12/qd12、object_velocity6；world坐标先去environment_origin，quaternion归一化并去q/-q符号歧义，按10um/10urad数值量化。seed、env UUID和episode名不构成独立初态。这里的“独立”是声明容差下不同的实际状态，不证明宽广任务多样性或统计随机独立性。

同物理初态的clone/retry保留raw descriptive count，但独立success_rate/Wilson只取观察到的第一条完整、首回合、未恢复/未resume trial，绝不挑clone最好结果。`raw_success_rate`另存。不同initial的Lift分布不代表同一个回合满足全部条件；完整正式native task success与更强`qualified_success`单独记录。invalid measurement即使带有限数字也不计有效coverage，NA输出None；3mm diagnostic tracker不能冒充30mm正式anchor slip。

`evaluate_records/aggregate_episodes` 只给初步统计，不能由caller写 `qualified=True` 或source_scope字符串直接升级。实际门槛位于 `tabletop_stereo_promotion.py`，会重开原始artifact、验证SHA与producer合同、重算回合或立体深度。公开接口失败统一为ValueError（缺少合法native材料默认拒绝）：

```python
verify_native_episode(proof, *, expected_identity=None, subject=None)
verify_force_qualification(binding)
require_teacher_qualification(report, checkpoint_sha256, styles=STYLES)
qualified_episode_identities(report)
verify_stereo_qualification(proof)
measure_stereo_evidence(proof)
verify_student_qualification(report, checkpoint_sha256, styles=STYLES)
promotion_ledger(phases, *, teacher_report=None, teacher_checkpoint_sha256=None,
                 stereo_native_proof=None, student_report=None,
                 student_checkpoint_sha256=None)
```

proof/report支持字典或JSON路径；相对artifact路径相对于其所在proof文件解析。`expected_identity`与proof.model_identity精确匹配，subject可owner字符串或期望checkpoint等字段字典。episode verifier返回顶层run_uuid/env_id/episode_uid/episode_id/reset_id；qualified_episode_identities返回对应4元组集合，供BC成功pool重新绑定当前采集episode。Teacher历史资格不能自动把Student访问的失败轨迹变成成功BC示范。

## 原始成功episode协议

`NativeEpisodeProof.version=dh116-native-episode-proof-v1`，包含source_scope=native、schema_only=false、process_exit_code=0、canonical action/state合同SHA、model_identity、实际checkpoint SHA、sample_count和raw_artifact的path/SHA。producer包含source_artifact、config_artifact、entrypoint、kwargs、kwargs_sha256；config.resolved_kwargs必须精确等于kwargs，480Hz/60Hz固定。当前只接受已审查的 `NativeTensorBridge._capture` producer源文件原始字节；任意文件与“native=true”不能代替来源身份。

原始JSONL第一行是episode_start，带唯一run/env/episode/reset、owner、grasp_style、实际SDK物理初态、初始tick/time、first_episode=true、resumed=false、recovered=false、initial_measurement_valid=true、physical_initial_condition_source=actual_sdk_readback。其后live sample逐480Hz无洞至第一个terminal；terminal后非live no-op允许不记录。不同env/episode不能拼接，重复tick、非单调time、截断、首terminal之后仍live均拒绝。raw非有限值需显式null/NA，不接受NaN JSON常量冒充有效数值。

正式成功必须同时有actual native task success且failure_code=0，并有最后5s/2400个raw样本支持净Lift≥30mm及物理有效性；完整回放原SurfaceLiftSuccess至少300个60Hz完成窗，contact loads/vectors由原算法作dt mean，而正式slip是每窗全部raw三点相对固定anchor的最大Euclidean，绝不是均值。首个完成窗前7个raw没有既有good-window/slip证据，不把False误判失败，也不把占位0当有效slip。末尾Hold沿用原算法数值容差 `>=5-1e-12`，不修改2400 raw/300窗预算或其它阈值。外部承托力、外加向上力、attachment数与source qualification必须逐sample有资格、有效且分别为0/0/0，不能从接触接近、物体高度或未报力猜测。这里是更强qualification gate，不修改或误称原SurfaceLiftSuccess已额外使用这些force条件。正式失败回合仍可作为统计分母，不能收入成功池。

Teacher qualification重开每episode raw proof，checkpoint必须实际artifact字节一致，owner必须Teacher；每种抓法至少100个不同实际首回合初态，并至少有完整qualified native success。它不要求宣称100%成功率；实际失败数与区间必须保留。报告同时需要原始checkpoint/source/config artifact路径与SHA。旧MPC、旧Teacher重放、BC loss或CPU算术不能替代本新checkpoint的证据。

## 力学来源资格协议

`verify_force_qualification` 输入 `{proof_path,proof_sha256}`，所指proof版本为dh116-force-source-proof-v1，必须通过同一native producer/source/config/kwargs验证。要求ON_OBJECT、N、world、已解析raw contact body ID、完整object-filtered与tangential reports、external-object contact和authored-force/attachment独立审计，并绑定至少两份独立audit artifacts。

原始contact JSONL每条需同run/env/episode/reset/tick/time、10个互不复用的2×5逐指sensor path映射、原始buffer offset、raw partner ID到实际object prim路径映射。每contact保存ON_HAND与ON_OBJECT normal/friction向量以及position，逐项检查对象侧符号反转。sensor path与digit一致，同sensor buffer slot不能广播到其它手指。两手需要实际contact coverage；外部承托/外加力/attachment也有同tick原始审计值。episode verifier把该source trace按完整身份+480Hz tick与episode raw逐样本联结，不能借另一环境/另一段日志的力资格补当前回合。

该协议明确当前真实缺口：现有Bridge/PhysX capture尚未输出完整已qualification的逐指partner、完整external反力与ON_OBJECT证明。本轮没有生成假native材料来使其通过。Phase1成功BC、R1/R2与Phase3 force loss统一复用这些公共门槛，缺测继续NA。

## Stereo与最终Student晋级

Stereo proof版本为dh116-native-stereo-proof-v1，绑定当前cameraCLI producer源字节与实际resolved kwargs；绑定calibration JSON、validation JSON、stereo_probe.pt和每frame独立depth reference artifact。capture须有真实environment_origins、native_evidence、ReferenceTime/IsaacReadTimes、相机K/T读回与完整pre-action identity。每frame左右RGB重新运行实际StereoDepth，对saved depth/valid逐项比较，独立reference必须同frame/time/calibration、scope native、明确validation-only且从未送Actor。依重新计算的MAE/RMSE、有效覆盖与像素数判定；不是信任报告中的一个calibrated=true。

`stereo_validation_v1.json` 当前政策为MAE≤5mm、RMSE≤10mm、valid-reference coverage≥50%、至少1000像素/2帧。这是明确的软件验收政策，不宣称整数SAD已达到，更不把5mm深度误差等价为0.5mm任务slip精度。实际resolution、K、baseline和depth搜索参数来自绑定profile/config，代码不固定224或64 disparity；分辨率改变必须重算calibrationSHA、读回、视差/资源预算和误差。更高分辨率本身不授资格。

Student gate重开完整heldout native episodes及source/config/checkpoint artifact，严格owner=student与实际checkpointSHA。每个raw sample的runtime声明必须与episode_start相同，Teacher execution/MPC/privileged Actor inputs/critic deployment均false；Actor仅stereo RGB+proprio，深度来自左右RGB。训练initial SHA与source_episode_group不得和holdout重叠；每抓法≥100不同实测初态。不得拿混合owner、3mm诊断slip、recovery回合或同源训练episode晋升。

ledger分implemented、cpu_verified、native_verified、physical_qualified四层。只有重新验证的Teacher+Stereo可令formal_dagger_allowed=true；只有重新验证的Stereo+最终Student可令deployment_allowed=true，缺证据时保留pendingfalse。正向分支已实现，未硬编码永久false。它允许没有物理资格时继续隔离CPU开发，不把“实现完成”当作实验完成。

best checkpoint选择核实际checkpoint文件SHA，仅独立heldout initial/source groups，比较相同实际初态cohort；不使用Peak训练reward别名。native报告重新读取原始proof；CPU候选可用于软件实验，但即使报告手写qualified=true也始终physical_qualified=false。

## 验证与范围

先写测试观察新模块缺失RED；随后针对无效measurement有限数字、checkpoint缺实际artifact、clone重试膨胀分母、native积极晋级接口缺失分别观察RED后修复。CPU正例仅验证Wilson/metric门槛算术、真实CPU权重文件绑定及mock verifier后的ledger布尔控制流；**不把mock或fixture登记成native成功**。负例覆盖任意bytes+native flag、metadata force proof、无完整原始证据、source/time/tick/terminal错误、diagnostic slip、NA、clone去重、owner归属与训练/heldout泄漏。

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_evaluation.py
```

最近包含evaluation、benchmark/reference、camera CLI、resolution、training lineage和原source freeze的组合fresh **81 passed、exit0（13.89s）**。最终全阶段集成命令与准确计数由root记录。native0；Teacher force/support资格、Stereo真实误差、最终Student100初态验收均pending。没有本地安装新依赖、没有训练、没有修改旧checkpoint或原528个受冻结文件。

## Resolution v2：实际可执行的基准与验证reference

新增 `tabletop_stereo_reference.py`、`tabletop_stereo_resolution_benchmark.py`、`scripts/benchmark_dh116_stereo_resolution.py` 与CPU测试，并把validation-only reference及实际性能捕获接入新的camera CLI。当前用户要求先完善全部代码，再考虑分辨率native实验；**本轮没有执行这条native命令，native仍0**。

独立ReferenceReader在同一个真实2N相机rig上新建仅供验证的depth+instance-ID CameraView render product。远端SDK只读审计的camera_view.py SHA为 `58557bbcb900574a4b9858c13e9e479b96fd5dfe09d872b07d5c010cca64d99e`；该版本的depth输出实际绑定distance_to_image_plane，即optical-Z，并支持 `instance_id_segmentation_fast` 的uint32数据。实现会在真正native构造时读取并核对SDK源字节，未知版本拒绝；在run里归档源快照，不把只读源码核对称为原生运行验收。

ReferenceReader只读当前depth/instance-ID，不render、不step。独立ReferenceTime/IsaacReadTimes与实际RGB pair严格同frame/time/calibration，并核对render frame、SWH及有理时钟；同simulation time但异render frame也拒绝。不一致会保留失败frame记录并重试整对RGB，绝不改metadata冒充同步。reference单独文件保存，显式validation_only=true/source_units=optical_axis_z_m，永不进入StereoCamera packet、SensorBatch或Actor。RGB warm-colour candidate用于明确命名的 `rgb_candidate_point_cloud_completeness`，不是GT语义mask。

真正wire reference从 `RigidObject.root_physx_view.prim_paths` 读取每env的实际Wire根路径，读取同product的原始instance-ID并先转signed long，保留完整idToLabels。仅接受绝对USD primPath或显式primPath字段，核实际stage存在，只匹配本env根或其descendants；拒绝名称contains-wire、class语义猜测、属性路径与跨env广播。ID/map/roots/stage读回/SDK source/time另写 `wire_instance_reference_####.pt` 并绑定SHA，promotion重开并重解析，mask必须逐像素可复现。未知SDK info格式继续NA并记录原因，当前没有声称远端实际metadata格式已跑通。wire completeness按dense reference-visible wire点在5mm 3D容差内的重建率计算，不是隐藏整物体完整率；任意frame缺独立mask则整个正式completeness为NA，另报有证据子集及reference-frame coverage。

新的 `measure_stereo_evidence` 复核同一套source/外部退出/artifact/sync/calibration/RGB重算条件后，返回真实MAE/RMSE/P95/coverage、cohort与performance，即使accuracy不合格仍 `measurements_verified=true, qualified=false` 并保留实测坏值。`verify_stereo_qualification` 再执行晋级门槛，不放宽标准。只有没有真实native材料或材料不能通过来源/对齐核验时才是NA。分辨率comparison复用此measurement接口，不信任普通report中的自填metrics。

callback不能知道最终OS退出，故只写 `native_stereo_proof_pending.json`、process_exit_code=None。serial benchmark使用当前sys.executable逐个运行probe，观察真实subprocess returncode后另写completion.json，绑定实际command、stdout/stderr SHA、runner源SHA及所有child产物SHA；退出0且source freeze未变才exclusive创建新的completed proof，不覆盖pending或失败日志。source gate同时要求该外部completion，capture、producer config、validation config、depth/instance reference与performance必须已经出现在实际退出时冻结的child清单，不能退出后拼入新reference。单独probe不会自授exit0。

frozen probe默认1帧、0warmup。benchmark多帧显式加 `--advance-control-between-samples`，由caller在完整RGB/reference pair之间执行一次原60Hz/480Hz的zero24控制，owner=diagnostic_zero，不调用Teacher/MPC。zero不保证碰撞后世界冻结，实际policy/physics tick、timestamp、reset、failure/terminal及zero动作都会保存；pair内部仍绝不推进物理。cohort重新hash实际SDK物理初态、rig及实际相对policy/physics/reset schedule，不用seed或env UUID伪造相同条件。

默认accuracy profile是640×480；可显式并列224×96、320×240、640×360等config。原224文件不改；controlled benchmark在内存统一blockmatch/5-pixel block/subpixel/processing-batch，实际参数进入validation config与producer kwargs。integer ablation必须另行标注，不能把integer→subpixel提升全部归于分辨率。所有分辨率必须重新标定K/SHA和视差预算，代码不偷偷resize图像。

GPU记录实际Torch peak allocated/reserved及实际mem_get_info的device used/free；后者是所选device的采样值，不是整个多GPU应用或NVML进程峰值。reference render product处于激活状态，相关资源开销明确计入验证实验，不能冒充纯RGB生产峰值。CPU或不可用GPU值为None，不是0字节。实际capture/depth分别计时；可选matching actor-only package执行真正forward，actor_ms还包含实际RGB proposal/point/history前处理（另存forward_ms）。没提供Actor包时actor_ms=None，不拿0推吞吐；profile包的calibration SHA不匹配则保留失败，不隐藏转分辨率。

实现后的显式执行入口如下，仅供代码验收后另行启动：

```bash
bash scripts/python.sh scripts/benchmark_dh116_stereo_resolution.py \
  --configs configs/dh116_training/stereo_camera_v1.json \
            configs/dh116_training/stereo_camera_320x240_v2.json \
            configs/dh116_training/stereo_camera_640x360_v2.json \
            configs/dh116_training/stereo_camera_640x480_v2.json \
  --output runs/stereo_resolution_new_run --seed 116 --num-envs 2 \
  --frames 4 --warmup-frames 2 --image-processing-batch-size 1 \
  --enable-experiment --execute-native
```

Actor吞吐需要另加同calibration的 `--actor-package`。不带execute-native只写新目录的计划/冻结证据，check-config只验证配置且不创建run。不能把这些模式写成实际native实验。serial benchmark退出后自动运行公共measurement/comparison，忽略callback自报metrics；未达accuracy门槛仍显示通过来源验证的实测误差，final selection必须另给 `--selection-thresholds` 完整JSON门槛、相同实际cohort、Actor吞吐与wire证据。每profile另存immutable verified result和整体comparison。实际matcher配置从 `StereoDepth.from_config(...).to_config()` 标准化并随proof返回，包含真实视差预算/defaults/实验override。

正式heldout还要求 `provenance.training_manifest_artifact={path,sha256}`，版本 `dh116-training-source-manifest-v1`。evaluation重读外层bytes，再由共同training-lineage loader重读实际模型、递归parent和全部登记raw headers，重算初态SHA/source groups；手填空training lists、缺登记或不完整inventory不能晋级。manifest只证明已登记清单的内容及闭包，无法独立证明登记人从未遗漏外部训练数据；缺完整登记时仍可输出统计。native最佳checkpoint的physical_qualified另调用完整Teacher/纯Student gate，不能仅因raw source已验证即为True。

当前未接通的实际资格逐项为：没有运行任何native采集/benchmark；Bridge raw rows仍是source_scope=unqualified，force_source_qualified/support_source_qualified=False，external support/upforce/attachment为NA；完整ON_OBJECT/逐指partner/tangential/external/authored-force审计source尚未产生；没有真实instance-ID info格式与depth误差/GPU/Actor计时数据；没有本次checkpoint每抓法100实际独立首回合与完整训练来源登记。因此Teacher、formal DAgger、force supervision与部署仍blocked/pending，并非只差一次成功事件。上述读取、测量、晋级正向代码路径均已实现，CPU正例仅证明schema/算术/控制流，不授native资格。本轮native结果始终为0。

最近CPU：evaluation 28项与benchmark/reference 22项，连同camera CLI、resolution、lineage、原528source freeze的fresh组合为81 passed/exit0；真实SurfaceLiftSuccess回归复现2400步后duration=4.999999999999988/event=True并验证首7条pending窗口不会误拒。全仓最后集成由root记录。

## Reference v2：真实instance-ID schema修复与Wire ROI误差

root随后在冻结source `/tmp/dh116-stereo-20261110T114000Z/snapshot` 显式执行native benchmark。只读核查的640×480结果位于 `native_resolution_04/640x480_verified_result.json`，SHA256 `139d05fb45fb43471d00a22306ace57391545903a3e86ebb3800cb05ffaffb85`。公共measurement记录 `measurements_verified=true, qualified=false`：global MAE=47.1807mm、RMSE=95.4598mm、P95=180.9252mm；wire completeness=NA，RGB candidate完整率=19.4558%，Actor latency=NA。这些是明确不合格的真实结果，不用RGB候选替代wire，也没有降低global gate阈值。其余旧分辨率结果由root保存汇总；本次子agent仅CPU读取已完成文件，没有另启native。

真实首帧 `depth_reference_0000.pt` SHA为 `55e2286209ef2c2b4331d4d6c6d1717cbca5746d1b8a246c29d7fdc6ecdfb62b`；独立 `wire_instance_reference_0000.pt` SHA为 `ac7263910684709cb7dec3642c119e7635b611fafe159c4a7a6c7d43a9f40a07`。实际schema是int key到absolute renderer path的 `idToLabels` 字典，包含6→`/World/envs/env_0/Wire/geometry/mesh`、1→env_1的同级Wire mesh；actual RigidObject roots也分别正确。首帧自属Wire ROI像素为env0 2765、env1 2637，env0图另能看到env1的Wire，不能跨env广播。四帧21..24中每帧均有473187个ID0像素，全部optical-Z无效，没有任何finite-depth/ROI像素。ID0未提供label；这里**不宣称它已由fast annotator证明是BACKGROUND**。

旧parser拒绝整帧的原因是未映射ID0和手部 `.../mesh.proto_mesh_id0` renderer token不能作为真实USD prim验证。只读审查的SDK `annotators_default.py` SHA为 `756dbfb4f262e623606bcbff2ddb315be723cf371e7a72983493d7e9977ddd56`，973..1006行把 `instance_id_segmentation_fast` 定义为renderer instance-ID/token-map调试输出；不能套用其它semantic annotator的0=BACKGROUND示例。该SDK源现在与CameraView源一起真实校验并归档。

新采集写 `dh116-wire-instance-reference-v2`：独立记录actualWire根的可见USD Gprim渲染几何清单、每项stage存在读回、完整raw IDs/map和同frame optical-Z validity。每个Wire渲染几何都必须有明确、真实stage验证的Wire label归属，才认为inventory完整。只有完整inventory存在时，已明确映射到非Wire绝对renderer namespace的精确 `.proto_mesh_idN` token可以排除；保留原token/namespace，不把它冒充真实USD prim，也绝不通过去后缀猜Wire ID。没有label的ID在任何有效optical-Z像素出现都继续NA；仅完全处于无效光学reference区时记录为unresolved-outside-validity，不能改称已识别背景。缺Wire label、Wire自身仅有不明proto token、缺stage几何或未知finite ID均fail closed。

promotion重读v2 raw artifact并重新解析inventory/map/validity，验证SDK annotator源与外部process completion产物清单；必须与实际reference.valid一致。旧v1 artifact的stage清单为空，保留原样，**不能补写或事后升级资格**。root须在新source freeze和独立run下重新采集四profile，才可能得到新的wire实测。

公共measurement额外返回 `wire_depth_mae_m`、`wire_depth_rmse_m`、`wire_depth_p95_m`、`wire_depth_count`、`wire_depth_validity_coverage`，均来自同一已验证raw-ID Wire mask的reference-visible ROI；completeness仍是可见渲染Wire像素对应3D点在既定5mm容差内的重建率，不补遮挡线段、不声称完整隐形几何。global depth指标和原验收政策完全不变。comparison仅透传经过公共measurement验证的Wire ROI值，不读caller自报值，也没有用新ROI字段暗中改变selection标准。

本修复先观察6项新测试RED，再实现并运行完整目标组：最新reference/benchmark、evaluation、cameraCLI、resolution及原528源冻结检查 **83 passed，exit0（10.36s）**。CPU测试覆盖真实schema形态、逐env归属、unknown finite ID、缺Wire label、Wire proto、无关synthetic nonWire token、缺stage路径、旧证据不得升级以及Wire ROI独立误差。v2代码尚未native执行，不把这些CPU正例记作新的native采集或资格。
