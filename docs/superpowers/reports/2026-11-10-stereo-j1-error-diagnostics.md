# J1：已有原生深度误差与线缆可见点完整率诊断

本轮按786行任务书第11节执行 **J0/J1**。以远端最新 `codex/dh116-stereo-privileged-dagger@7c808781ebd2cf969ca08776a83a65e3a2834582` 为起点；没有重建Phase0～7、修改matcher/controller/reward、训练、调用Isaac或新增native capture，也没有push/merge。最终Teacher→BC→DAgger→Student PPO目标保留，但当前不能跨过Stereo/Teacher资格。

## 实际交付与验证

新增的三个诊断模块、一个CLI及两个测试文件仅用于评估：

|文件|作用|
|---|---|
|`tabletop_stereo_error_diagnostics.py`|固定reference分母的区域统计、同像素3D误差、右深度投影可见性proxy、分开拒绝原因与selected-pixel成本曲线|
|`tabletop_stereo_error_geometry.py`|实际可见mask局部PCA方向、宽度proxy及严格simple-path表面曲率分桶；不补隐藏几何|
|`tabletop_stereo_error_cli.py`|父benchmark→capture report→pending/completed proof交叉绑定、原文件SHA、逐帧/逐env报告、源码版本绑定、可视化与独占输出|
|`scripts/analyze_dh116_stereo_errors.py`|离线CLI，不导入/启动Isaac，不授予qualification|
|`tests/test_dh116_stereo_error_diagnostics.py`|分母、NA、欧氏射线、遮挡方向、replay四象限、来源错配、SGBM能力及独占输出|
|`tests/test_dh116_stereo_error_geometry.py`|直线、弯曲、staircase、分叉、环、断片、无深度、各向同性NA和宽度/单位|

J0独立报告、matcher根因报告、两份计划以及机器汇总/readiness/文件清单/README/CHANGELOG同步更新，未删除文件。完整路径见[文件清单](../../stereo_phase0_7_file_inventory.md)。

最终CPU：**Windows595 passed/1 skipped，80.43s；服务器596 passed/1 skipped，70.27s；均exit0**。30项早期诊断/几何组合通过后，补充NA可视化回归，最终新增31项用例。两平台重叠，不能相加。RED/GREEN保留在私有`artifacts/stereo-j1-20261110/`和`stereo-j1-reference-20261110/`：初始缺模块/API的RED仅证明接口未实现；缺instance-map被误写零ROI、错run来源、SGBM NA等回归另有失败记录。不能把任何CPU fixture当native证据。

现有服务器环境：Python3.10.12、Torch2.7.0+cu128、torch CUDA runtime12.8、NumPy1.26.4、SciPy1.15.3、OpenCV4.11.0.86、Matplotlib3.10.9。CPU测试与分析均`CUDA_VISIBLE_DEVICES=`，没有安装/升级依赖。旧捕获仍属于既有Isaac Sim4.5来源；本轮没有运行Isaac，因此没有本轮新的Isaac/GPU物理验证。

## 输入、命令与来源

重新读取全部10个实际run的`stereo_probe.pt`、40份光轴深度reference、已保存的实例ID、原producer/config/calibration/performance/completion及proof。旧rig四份缺完整Wire资格的来源仍NA，不根据后来parser补授信。6份front/roll20有原始可解析Wire mask。

最终冻结目录 `/tmp/dh116-stereo-j1-20261110-v04/snapshot`，1191文件；私有归档SHA `acaec48d8309ae0ad90a372af385ab7184b980f3ea77ba1669c9a4ef2517c893`。实际执行：

```bash
cd /tmp/dh116-stereo-j1-20261110-v04/snapshot
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /mnt/mydisk/xk/envs/dh116/bin/python \
scripts/analyze_dh116_stereo_errors.py \
--manifest docs/contracts/stereo_native_resolution_results.json \
--output /tmp/dh116-stereo-j1-20261110-v04/all10-analysis \
--threads 4 --max-cost-pixels 24
```

实际exit0，**10 completed/0 failed，57.223s**；40 batch frames、80 env-frame观测，52份真实图像/曲线/3D对齐PNG。每run四帧RGB完全相同，matcher replay按相同RGB/config/calibration缓存，每帧仍重算指标并绑定自己的身份；这些重复帧不是80个独立物理初态。全部80个env-frame的saved/replay validity mismatch=0，158个原始/来源绑定路径末尾SHA均未变。此前v01单run smoke、v02/v03完整reanalysis产物均保留，不算新native runs。

服务器CPU通过原wrapper运行：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 \
/mnt/mydisk/xk/dh116/scripts/python.sh \
/tmp/dh116-stereo-j1-20261110-v04/frozen_entry.py pytest
```

日志末尾`native_modules_imported=[]`。本机使用现有uv Python，`PYTHONPATH=src;artifacts/stage7_lifecycle_cpu_deps_v01`，执行全部`test_dh116_stereo_*.py`与原Phase0/privileged/surface/squashed-Gaussian回归，`-m pytest -q -p no:cacheprovider`。

每个`diagnostics.json`记录当前实际导入的7个实现文件SHA、包版本、原采集producer SHA、校准/原始数据SHA及resolved matcher config。机器摘要见 [J1结果](../../contracts/stereo_j1_error_diagnostics_summary.json)；逐帧所有区域指标、成本数组、原始图像保留私有，不将summary当逐像素证明。

## 定义与覆盖范围

RGB为`[B,3,H,W]`；Z/disparity/valid/ROI为`[B,H,W]`；K为3×3或batched3×3，像素中心`u+.5,v+.5`。区域包括全图、Wire、1px腐蚀边缘/内部、thin骨架1px窄带、table、palm/finger、两视图几何共同可见/左侧被遮挡proxy、纹理、距离、方向和可用局部曲率。

`reference_count=|region∩reference_valid|`；误差仅在该集合与真实保存predicted-valid/finite/positive的intersection计算。coverage分母保持reference_count；P50/P90/P95/max不跨env平均。正确3D完整率采用同像素`|ΔZ|*||K^-1[u+.5,v+.5,1]||≤5mm`，没有最近邻对齐或缩ROI提高准确率。误差为空时NA，不填0。

拒绝条件输出重叠计数与固定顺序首失败计数；它们不是独立可相加的物理原因。每区域另列saved/replay四象限，避免将CPU重放差异冒充原生失败。没有实现的depth-edge rejection保持NA，reference depth-edge仅为诊断区域。SGBM内部uniqueness无法由0/1范围标志恢复，保持NA；既有输出confidence=1不能称为概率。

右深度真实存在，右实例mask未归档。共同可见/单侧遮挡是基于右光轴Z和K/B的投影proxy，报告0.5/1/2mm容差敏感性；右Wire GT始终NA，不用投影mask冒充真值。

## 按证据排序的根因

1. **视差歧义/亚像素拟合偏差，已在固定支持集隔离。** Block9共同有效1696点中467点由Block5正确变错误，442点整数winner仍正确为43，291点offset=-.25。像素env0/frame21/(268,347)：GTd43.1491，Block5d43.1667、误差-.314mm；Block9d42.75、误差+7.218mm，仍通过LR与confidence。三点正曲率不等于真实成本可抛物线拟合。固定valid整数代换完整率46.05%→83.53%只用于归因；全部GTd仅42.784–43.218，不能据此将“禁用亚像素”作为通用修复。
2. **低纹理与窗口背景混合，解释覆盖增长。** Block5首拒left texture1515、uniqueness660，Block9降为0/106。新增valid点原5×5 median gray std=.0001057且wire占比1；9×9后std=.01114、wire占比.85185，来自边界变化而非新增真实线缆纹理。新增点仅44.10%正确。实际wire宽约7–9px，Block9跨越整条线缆。
3. **手部边缘/真实遮挡及自洽错配，主要涉及SGBM坏尾。** SGBM278个>20mm坏点中238个9×9窗含hand，41点LR=0；部分GT右深度一致但匹配错误，部分确有遮挡proxy。SGBM接受点confidence全1是范围标志；收紧LR到.1仍保留136个坏点并把完整率降至70.44%。这是局部关联与反事实证据，未暴露OpenCV内部路径cost，不能宣称每个坏点已唯一归因。
4. **极线/基线方向，是已测的受控rig因素，但没有解决精度。** front几乎水平线缆与极线平行，roll20改善支持；J2仍需跨方向、宽视差、距离和材质测试。旧rig→front同时改变构图，不能把全部收益归于某一个变量。
5. **全局ROI/场景定义解释global与Wire指标不同，不能据此改门槛。** Block9首帧env0/1约133080/132673个reference像素低纹理，均无有效输出；table占reference区域约95.5%，table有效点MAE约66mm、手部约20–29mm。Wire自身MAE仍7.978mm，故不能仅删背景ROI使其晋级。
6. **颜色分割不是这组46.05%指标的原因；骨架到token另有真实阻断。** 6个有Wire-ID run、12个首帧env观测，当前brown/runtime颜色阈值在reference有效空间precision/recall均100%，加depth-valid后recall降为实际coverage。46.05%本来以GT实例ROI计算，与颜色阈值无关。但实际调用现有`WirePointExtractor.extract`，12/12均0/128有效token，严格8邻域simple-path判据拒绝数字阶梯/断片/分叉；不能将“颜色分对”当作128点路径可用，也不能据单一金色fixture称通用分割合格。
7. **3D点关联/欧氏门槛不是主要下降来源。** Block9四帧Z≤5mm7588点，Euclidean≤5mm7484点，差104点=0.640个百分点；没有nearest-neighbour匹配。严格曲率仅12/4063点可用，4051点NA；PCA方向可用于3978/4063点，不能拿方向覆盖冒充完整三维中心线曲率。
8. **标定/参考ROI系统错误缺乏支持。** 全10run的实际桌面反投影对WorldZ=.300最大残差<4.55e-7m；误当ray distance产生约50–68mm，但现有代码使用正确光轴Z。真实Wire-ID点对同tick圆柱表面的偏差最大约.392mm，没有>2mm背景GT混入；数值符合10边形渲染表面的推断，但未归档renderer细分源码，不能声称硬件标定或精确tessellation来源获证。

因果强度各有不同：固定输入数值干预支持当前数据上的贡献，窗口/hand混入与SGBM路径仍有相关性和待验证部分。上述排序不是八个互斥可加误差比例。

## 详细证据与可视化

- [J0源码/gap matrix](2026-11-10-stereo-j0-latest-source-audit.md)
- [逐像素matcher根因及反事实](2026-11-10-stereo-j1-matcher-rootcause.md)
- 私有reference审计：`artifacts/stereo-j1-reference-20261110/reference_audit.json`，SHA `825682d4822db63934fb8e4a29148780f34118e842f7bda2ffd5fb41a544b96a`。
- 私有实际extractor结果：`segmentation_runtime_audit.json`，SHA `f656a2f5b9ef48b9a3ed9878add887434776413183e43180d18724dc0088703e`。
- 原始读回派生图位于`artifacts/stereo-j1-20261110/all10-analysis/`，每run的`figures/`保存左/右RGB、mask、视差、reference、误差、有效性、成本/纹理及三个投影面的可见Wire对齐图。

![Block9实际像素与误差](../../../artifacts/stereo-j1-20261110/all10-analysis/09_front_roll20_block9_640x480/figures/env0_frame21_pixels.png)

## Readiness与下一步

原10次native准确率仍全部false；本轮新增native次数0；没有Teacher/Student成功数据、训练Checkpoint、Actor latency或训练吞吐。选定分辨率仍null，formal BC/DAgger和deployment仍blocked；30mm/5s/0.5mm与当前5mm感知诊断门槛均未修改。528份原生产文件SHA不变，旧失败/中间分析保留，Draft PR不变且本轮不push。

J2的实际修改点、RED测试名、候选顺序和资源上限见[实施计划](../plans/2026-11-10-stereo-j2-evidence-driven-perception.md)。优先补成本/亚像素可信度与拒绝能力、验证数字骨架拓扑处理，再扩展统一matcher候选、原生ID标签分割/跟踪和depth辅助head。CPU4线程、2env、同40batch帧；单候选首帧≤8GiB/120s，可选离线GPU≤4GiB但本轮未用。不能直接用当前窄视差整数优势、颜色fixture满分或离线反事实替代J3新原生资格；J4之后Teacher训练另按任务顺序与证据门槛实施。
