# J2-A：matcher合同、可靠亚像素与安全回归

基线为GitHub/本机`692e09d2473502574d0870dfaffff38b6c0a544c`。已完整读取455行任务书，SHA256 `d4497f4a56e705b37f5f4bad6b75449378f1fa41396f3887be22f83712d01d68`。按第十六节本轮只执行J2-A；旧默认SAD保留，候选显式启用，没有新Isaac、GPU计算、模型训练、push或merge。**J2-A软件验收通过，不授予原生深度、物理抓取或部署资格。**

## 文件和实际改动

|文件|改动及原因|
|---|---|
|`tabletop_stereo_depth.py`|保留旧SAD数值；增加aliases/能力说明/逐项拒绝，接入opt-in reliable_sad；修复非正视差换算和NaN LR采样；raw validity约束RGB边界proxy|
|`tabletop_stereo_matcher_candidates.py`（新）|五成本模型验证、独立图像空间分数搜索、单频递推重复纹理证据；常数数量成本平面|
|`tabletop_stereo_error_diagnostics.py`|复用实际compute诊断，避免候选误走SGBM；保留J1兼容报告key及NA|
|`tabletop_stereo_error_cli.py`|诊断producer身份加入候选实现SHA|
|`tabletop_stereo_preprocessing.py`|processing source合同绑定新候选文件；禁止旧包静默切换实现|
|`stereo_matcher_reliable_sad_j2a_v1.json`（新）|独立候选配置，默认不启用，native_qualified=false|
|`stereo_matcher_five_cost_j2a_failure_v1.json`（新）|保留失败单因素对照，仅关闭photometric_verification，不覆盖旧配置|
|7份新增J2-A测试/fixture文件|固定旧源码oracle、连续解析RGB、合同/数值/安全/集成回归；不是生产平行实现|
|`test_dh116_stereo_geometry.py`|只将旧all(values).shape检查明确为原五个Tensor；原深度精度及mask断言不变|

README、CHANGELOG、readiness、文件清单和[实施计划](../plans/2026-11-11-stereo-j2a-matcher-contract.md)同步更新。没有删除文件，旧528份保护源码/脚本/配置、原物理参数/控制器/Teacher/Student/MPC和历史Checkpoint不变。

## matcher合同及兼容性

输入是float RGB `[B,3,H,W]`、已标定rectified K/T及raw bool image masks；无GT/reference/privileged参数。原五个Tensor `depth_m, disparity_px, valid, occlusion, confidence`保留；`valid_mask/occlusion_mask`与旧mask共享Tensor。无效depth/disparity为NaN且valid=false。

`diagnostics.version=dh116-stereo-matcher-diagnostics-v1`。SAD confidence为相对成本margin，不是概率；SGBM保留0/1兼容数值，但明确`confidence_kind=compatibility_validity_indicator`、`quality_confidence_available=false`。OpenCV内部uniqueness/subpixel拒绝与真实物理遮挡来源均NA，不补0。低纹理左/右、全局搜索边界、LR、唯一性、组合遮挡、深度范围、亚像素和patch支持分别记录；组合occlusion不能冒充物理遮挡。

旧实现按`git show 692e09d`原样固化为测试专用oracle，SHA `36b35189d7576dde7790f22b468b1be4393e7acfbea8060dc0380073aed28008`。整数/亚像素、textureless、mask/遮挡及batch五种场景中，旧SAD全图五Tensor有限值/NaN分布/布尔mask逐值相等。新的元数据不会要求调用方把字典当深度Tensor。旧default `to_config`保持12字段；只有新method序列化reliability与photometric_verification。

实现源码SHA变更会使旧raw-RGB模型包的processing绑定拒绝新实现，这是显式保护；本轮不重写、不迁移旧包。新包需独立contract/source SHA，不能因数值兼容就忽略来源变化。

## 可靠候选的公式、形状和来源

`method='reliable_sad'`显式启用；旧默认仍为`blockmatch`。整数proposal使用原SAD。两次按disparity扫描，仅保留winner的±2五成本和远离winner至少2px的竞争模式；没有完整`[B,D,H,W]` cost volume。所有运行平面为`[B,H,W]`，视差始终float、深度m。诊断selected adapter仅接受`[P,D]`小样本，非生产VGA成本体。

1. 原三点偏移 `δ=.5*(C-1-C+1)/(C-1+C+1-2C0)` 保留为legacy trace。中心模型提出quadratic，外侧±2独立验证；低曲率、非凸、边界、非有限、多峰和RGB边界proxy分别记录。
2. SAD还可能近似L1 V。由外侧斜率估计a，`δV=(C-1-C+1)/(2a)`；要求斜率一致、非负floor及中心成本独立残差合格。失败只能使用声明的整数fallback或invalid；不能把正曲率等同几何准确。
3. 单独的图像空间验证直接在winner周围`-.75:.125:.75`十三个位置采样原始灰度，两个流式pass，不从五个整数成本继续外推。整个patch使用中心winner统一位移；只接受实际采样的`[-.5,.5]`位置、完整raw像素支持、正局部曲率和分离模式。不得先逐像素各自warp再pool，也不能二次腐蚀已检查的raw mask。
4. 新候选的唯一性使用分离整数模式，而不是将同一半像素盆的相邻整数tie错误当作两个模式。confidence仍是margin，未经概率校准。
5. 非整数远处周期别名不能仅由整数margin排除。新增RGB-only单频递推证据：`x[n-1]+x[n+1]=a*x[n]+b`，在31px有效triplet窗用float64统计拟合，至少17个triplet；`period=2π/acos(a/2)`，normalized residual≤1e-5、非平坦、窗口覆盖至少两周期且别名落在搜索范围时拒绝。无效像素先以有效triplet剔除，不让NaN乘0污染统计。该能力只覆盖所检验的单频模型加整数远模式，**不是任意重复图案不存在的证明**。

默认可靠性：min_curvature1e-6、quadratic残差.25、absolute残差.05、mode margin.05、fallback integer。RGB边界proxy仅由raw有效梯度对计算，灰度gradient>.02且max/mean>8；它不是semantic background truth。primary成本模型被拒绝但独立photo模型接受时，原拒绝trace仍保留，final model来源分别为0=fallback、1=quadratic、2=absolute、3=photo。不能将这些条件假装为独立可相加的物理错误率。

## 先失败再修正的证据

失败材料保留在私有artifacts，不删除或改写原raw：

- 初始合同RED 13 failed/14 passed：缺aliases/diagnostics、候选尚未支持、非法raw视差通过clamp变正深度；固定oracle和旧config已通过。
- 仅五整数成本模型的解析覆盖不达标。最终同源码、同RGB/config，只开关`photometric_verification`的6.25px对照：关闭为47/2800有效（1.6786%）、P95=.25px；开启为2800/2800、P95=0。失败配置独立保留，没有通过放宽coverage≥70%或P95≤.2px门槛晋级。
- 真实half-pixel相邻tie不能视为远模式；先新增RED再允许±.5端点，并单独检查远竞争者。
- 审查发现未使用的NaN high邻居仍参与`lo+(hi-lo)*0`，导致`accepted=True`但LR sample=NaN，比较绕过拒绝；seed41真实候选也可复现15个错误有效样本。修复仅安全替代未使用的非有限hi，并检查实际使用样本finite；有限旧样本算术顺序保留，兼容1e-6采样容差。
- 无效RGB曾影响patch边界proxy，使有效点disparity3.75→4。现在只有两端raw-valid的梯度对参与proxy；masked fill改变不影响输出或真实proxy。
- 单频2.7/5.3px别名×phase0/.37曾全2800点被误valid，confidence很高；真实RED保留。递推guard后四例均0/2800 valid，原丰富纹理控制仍100%覆盖；不因此宣称所有周期/遮挡已解决。
- 诊断源码绑定和SGBM NA可视化/来源语义均有明确回归；最终新增156项J2-A用例。

RED/GREEN分别见`artifacts/stereo-j2a-contract-*`、`stereo-j2a-safety-*`、`stereo-j2a-20261111/`及`stereo_j2a_*_tdd_results*.json`。这些是CPU软件证据，不是Isaac原生验证。

## 最终验证与实际命令

新增156项组合：**156 passed，29.86s，exit0**。最终相关全组：**Windows751 passed/1 skipped，115.60s；服务器752 passed/1 skipped，78.03s；均exit0**，两平台重叠不可相加。包含旧matcher/perception/geometry/model processing、privileged/surface及OpenCV回归。服务器输出`native_modules_imported=[]`。本机可选OpenCV未安装，服务器使用现有OpenCV完成数值路径；未新增依赖。

Windows：现有Python3.12.13/Torch2.14 CPU，`PYTHONPATH=src;artifacts/stage7_lifecycle_cpu_deps_v01`、CUDA_VISIBLE_DEVICES空、PYTHONDONTWRITEBYTECODE=1、PYTEST_DISABLE_PLUGIN_AUTOLOAD=1。

```text
python -m pytest -q -p no:cacheprovider
  tests/test_dh116_stereo_j2a_contract.py
  tests/test_dh116_stereo_j2a_reliability.py
  tests/test_dh116_stereo_j2a_analytic.py
  tests/test_dh116_stereo_j2a_integration.py
  tests/test_dh116_stereo_j2a_safety.py
```

完整组选择全部`tests/test_dh116_stereo_*.py`及既有11份privileged/Phase0/surface/squashed-Gaussian回归，实际日志`artifacts/stereo-j2a-20261111/full-local-01.log`。服务器使用Python3.10.12、Torch2.7.0+cu128、torch CUDA runtime12.8、NumPy1.26.4、OpenCV4.11.0、SciPy1.15.3，CUDA显式隐藏：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 \
/mnt/mydisk/xk/dh116/scripts/python.sh \
/tmp/dh116-stereo-j2a-20261111-v01/frozen_entry.py pytest
```

冻结归档1202文件，SHA `e339c4d1986973c4601aa6476eddbb50e129571310c283115522277083e0d037`；服务器日志`/tmp/dh116-stereo-j2a-20261111-v01/server_cpu.log`。本轮Isaac/GPU测试数0，GPU显存/完整sensor-to-action延迟/训练吞吐均未测，不填0或外推。

文档收尾后再次只读核验：528份保护文件raw SHA均不变；1120份`src/scripts/configs/tests`文件与服务器实际执行快照raw SHA一致，另1份解析测试仅清理末尾多余空行，逐字节去尾换行及Python AST比较均相同；14项当前源码/配置/测试证据绑定匹配；五目录166项与readiness逐文件清单一致；JSON解析、Python语法编译及暂存区`git diff --check`通过。所有已测生产代码未变。完整性结果保留在本地`artifacts/stereo-j2a-20261111/final_integrity.json`。

## 解析精度与真实Pixel A离线检查

解析图像直接采样连续band-limited辐射，不对离散left反复插值制造right；`I_R(x)=I_L(x+d)`、`Z=fB/d`。固定reference，不随输出valid缩分母。测试要求coverage≥70%、P95 disparity≤.2px；这是软件数值门槛，**不是原生5mm点完整率或30mm/5s/0.5mm成功标准**。

|解析case|固定reference|有效coverage|P95 disparity误差|错误有效点（>.2px）|
|---|---:|---:|---:|---:|
|d=6.1|2800|100%|.0250001px|0|
|d=6.25|2800|100%|0px|0|
|d=6.5|2800|100%|0px|0|
|d=6.75|2800|100%|0px|0|
|missing stripe|2352|90.476%|0px|0|
|observed occluder|2072|90.589%|0px|27|

0分位误差不等于所有像素零误差。27个遮挡边缘点依然被接受：d6对GT6.25、LR residual=.25在测试lr=.5内，26个photo/1个integer fallback，confidence可达.8662而非概率。这是明确保留的混合窗口限制，不能据P95=0说遮挡完全解决。四个非整数单频歧义例全拒绝，但一般重复纹理与跨场景鲁棒性仍待J2-B/F和J3。

额外只读已有J1 raw，对frame21/env0/像素(268,347)做19×400离线crop，K仅平移principal point而不重author相机：旧SAD仍d42.75、误差7.218mm；新候选拒绝该offset、model0 integer fallback d43、误差2.681mm。原probe/reference/calibration SHA前后匹配。它不是新native capture、全wire准确率或整scene重测；没有向matcher传GT。

实际CPU命令：`CUDA_VISIBLE_DEVICES= /mnt/mydisk/xk/envs/dh116/bin/python /tmp/dh116-stereo-j2a-20261111-v01/check_pixel_a.py`，exit0。单次crop耗时约.123/.943s，无预热、不含capture/Actor，不能作为VGA/GPU/训练吞吐选型。原数据仍位于J1旧run；本轮新输出为`pixel_a_check.json`。

## 状态和下一步

[机器验证摘要](../../contracts/stereo_j2a_validation_summary.json)、[readiness](../../contracts/stereo_phase0_7_readiness.json)及[文件清单](../../stereo_phase0_7_file_inventory.md)分开记录implemented、CPU verified、native capture verified、accuracy qualified、physical task qualified、deployment qualified。原10份native accuracy仍false，selected_resolution=null；Teacher/Student无新训练或成功证据，正式BC/DAgger与部署blocked。无模型Checkpoint、训练数据或物理参数被覆盖。

本轮不进入J2-B算法扩展、不执行J3原生复测；按任务书只提交J2-A本地软件改造。下一工作包在现有统一接口下增加Census/ZNCC/gradient/coarse-to-fine，保持相同cohort/reference和单因素配置，再顺序推进分割、拓扑/时序跟踪、depth辅助head及比较。已有[J2后续计划](../plans/2026-11-10-stereo-j2-evidence-driven-perception.md)保留，但不能据本轮软件通过提前训练Teacher、DAgger或Student。
