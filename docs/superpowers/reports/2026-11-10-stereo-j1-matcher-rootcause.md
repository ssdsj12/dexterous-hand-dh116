# J1：Roll20 matcher 拒绝链、亚像素偏差与 SGBM 坏尾根因证据

日期：2026-11-10。范围：J0/J1 只读调查；没有修改生产代码，没有启动 Isaac、GPU、训练或新 native capture，没有登记 qualification。

## 结论与证据等级

1. **[raw 实测 + source] Block9 的覆盖提升主要来自窗口碰到 wire 边界后获得纹理及唯一性，而原先平坦的 Block5 窗口大量被拒绝。** 同一首帧两 env 共 4063 个 reference-valid wire 像素，Block5 首拒绝归因中 left texture=1515、uniqueness=660；Block9 对应为 0、106。两方法 wire 上都没有 subpixel 缺邻居或非正曲率拒绝。不能把 wire coverage 的变化归因于亚像素拒绝减少。
2. **[raw 实测 + source] 共同有效集的主要 5 mm 损失是正确整数 winner 上的错误亚像素偏移。** 共同 1696 点内 467 点由 Block5 正确变 Block9 错误，反向 98 点；467 点中 442 点整数 winner 在两方法中均为 43，291 点 Block9 offset 恰为 -0.25 px，而其真实 disparity 大多大于 43。实测邻项 SAD cost 不对称，使三点抛物线公式产生反向偏移；正曲率、LR 与 confidence 仍通过现有 gate。
3. **[raw 实测] SGBM 的少量大视差错误集中在手与线缆相邻区域，并且可以左右一致。** 278 个有效 wire 点的 |Z error|>20 mm，全部低估深度；156 点超过 50 mm。278 点中 238 点的 9×9 左窗含原生 instance-map 已解析的 hand 像素；41 个坏点 LR 恰为零。当前输出 confidence 对所有 3937 个 SGBM 有效 wire 点均为 1.0；源码没有将 SGBM cost margin 输出为 confidence。
4. **[离线 counterfactual] 改变亚像素处理值得作为 J2 假设验证，不能据此晋级或宣称通用修复。** 固定 Block9 原有效支持集、仅以整数 winner 替代亚像素深度后，正确 3D 完整率 46.05%→83.53%；完整 CPU `subpixel=false` 重处理为 83.88%。但这里所有 wire GT disparity 只有 42.784–43.218 px，天然有利于整数 43；该完整重处理全图 MAE 47.69 mm、P95 176.04 mm，仍未通过原准确率门槛。

## 数据来源、重放及 SHA 核验

原始数据只读目录：

- `/tmp/dh116-stereo-20261110T120000Z/native_roll20_block5/640x480`
- `/tmp/dh116-stereo-20261110T120000Z/native_roll20_block9/640x480`
- 既有离线输出 `/tmp/dh116-stereo-roll20_offline-20261110/offline_sgbm9_first_frame_depth.pt`。

相机 `fx=417.0321193091858 px`、baseline `0.08 m`、`fB=33.36256954473487`；calibration SHA `02fabb87c548cd982728dac5d6357ca08327258514269b163e880427b2556700`。depth 配置保持 disparity `8..192`、depth `.2..1.2 m`、LR `1 px`、texture `.001`、uniqueness `.05`、subpixel=true、image batch=1；只在标注的 counterfactual 中改变 subpixel。

**[raw 实测]** 两方法的 frame 21、22、23、24（index 0..3）内，左右 RGB、reference depth bytes、reference valid、wire mask 均与本方法首帧完全相同；方法之间也相同。因此主诊断只计算首帧 2 env，绑定到既有四帧逐值相同的输入。首帧 4063 ROI 观测与四帧 16252 ROI 观测不是不同物理初态，不能用重复帧增加独立样本量。

**[raw 实测]** CPU 仪表化逐语句重放 `_match` 及 gate，Block5/9 各两个 env 的全图 valid mask 与保存的原 native output 均 **0 mismatch**；共同有效点最大 disparity 差 `3.8147e-6 px`，最大 depth 差 `1.7881e-7 m`。SGBM CPU 完整重放与旧离线输出的 mask、disparity **逐值相同**。因此下述计数归因于真实保存的输出，而不是任意替代 matcher。

**[SHA 实测]** 两个 `stereo_probe.pt` 与八个 `depth_reference_0000..0003.pt` 全部逐一匹配：原 `native_stereo_proof_completed.json` 的 artifact SHA、上一轮 `native_readback_and_rgb_comparison.json` 的 inventory SHA、本轮分析前 SHA、本轮分析结束 SHA。两 probe SHA：

| 原文件 | SHA256 |
|---|---|
| Block5 stereo_probe.pt | `733952a7aa7495b08e98b98d3e90e38266350f79630892f27f6b90a19c793235` |
| Block9 stereo_probe.pt | `be1ba0bf5eadf82d4313e2ecc297bf016851df86a906489939591f57d1bc80d2` |

十个 probe/depth-reference 的逐文件证据见 [subpixel_counterfactual.json](../../../artifacts/stereo-j1-matcher-20261110/subpixel_counterfactual.json) 中 `original_probe_and_depth_reference_hash_verification`。此外本轮所有 **25 个输入文件**（包括八个 wire-instance artifact、producer/proof、calibration、源码、旧 SGBM 输出）末尾重哈希均未变，完整路径/SHA 在 [matcher_rootcause_summary.json](../../../artifacts/stereo-j1-matcher-20261110/matcher_rootcause_summary.json) 的 `input_sha256`。wire mask 直接复用 `native_resolved_instance_id`、`wire_inventory_complete=true`、`blocked_reasons=[]` 的 reference；没有猜测 raw ID。

## 指标定义与基线复现

固定 wire ROI = 保存的 wire instance mask ∩ reference valid ∩ finite target ∩ `.2≤Z≤1.2 m`。误差仅在该 ROI 与方法有效输出的 intersection 上统计，同时报告该方法 coverage。正确 3D 完整率分母始终是固定 ROI；点误差使用像素中心 `(u+.5,v+.5)` 与 K 的 backprojection 等价 ray norm，阈值保持 5 mm。

| 首帧、2 env，固定 ROI=4063 | joint-valid | coverage | Z MAE mm | Z RMSE mm | Z P95 mm | 正确 3D 点/ROI | 完整率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原 native Block5 | 1772 | 43.61% | 7.350 | 16.533 | 34.334 | 1340 | 32.98% |
| 原 native Block9 | 3814 | 93.87% | 7.978 | 15.116 | 21.247 | 1871 | 46.05% |
| 旧 offline SGBM9 | 3937 | 96.90% | 6.980 | 17.762 | 46.881 | 3330 | 81.96% |

首帧与四帧汇总的经验分位数可能稍有不同：重复样本改变 quantile 插值位置。例如共同集首帧 Block5 P95=26.123 mm，四帧既有报告=26.454 mm；两者数据和定义没有冲突。本文新增像素计数均明确使用首帧，避免与四帧计数混用。

## 拒绝链：coverage 为什么增加

源：[`tabletop_stereo_depth.py:50`](../../../src/go2_pvcnn/control/dh116/tabletop_stereo_depth.py#L50) 局部 gray std；71 行方形 SAD；85–96 行 top2 confidence 和抛物线拟合；149–156 行右图采样、LR、texture、uniqueness、depth gate。

以下按源码可计算 predicate 的固定顺序分配每个拒绝像素：整数无支持→亚像素缺邻居→亚像素曲率失败→左 texture→patch border→右采样无效→右 confidence 采样无效→LR→uniqueness→depth range。顺序仅用于互斥归因；原实现为布尔 conjunction，原始重叠计数也保存在 JSON，不能把重叠计数直接相加。

| 首帧固定 wire ROI=4063 | Block5 首拒数 | Block9 首拒数 |
|---|---:|---:|
| 整数无支持、亚像素缺邻居/曲率失败、border | 0 | 0 |
| 左图 low texture | 1515 | 0 |
| 右图采样无效 | 44 | 1 |
| LR residual >1 px | 72 | 142 |
| uniqueness | 660 | 106 |
| depth range（前序 gate 后） | 0 | 0 |
| 最终接受 | 1772 | 3814 |

重叠计数：[raw 实测] Block5 左 texture=1515、右采样无效=1557、LR=1298、uniqueness=2199；Block9 对应 0、1、143、149。不能将 `occlusion` 字段直接解释为物理遮挡；源码将右图无纹理/亚像素无效/越界及 LR 不一致合为该字段。

**[raw 实测]** Block9 新增的 2118 个 valid 点，在 Block5 中 std 的 median=`.000105716`、窗内 wire fraction median=1.0，说明许多 5×5 窗完全落在近乎常色线缆上；9×9 后 std median=`.0111446`、wire fraction median=`.851852`，纹理可来自掺入的边界/背景。新增点的 Block9 正确率只有 `934/2118=44.10%`。这是当前数据上的来源解释，不等于增大窗口必然提高几何精度。

**[raw 实测]** 全图亚像素拒绝与 wire 的现象不同。首帧 `.2..1.2 m` reference ROI=295488，Block9 共252385点缺亚像素邻居；其中252297点整数 winner=最小 disparity8，252047点 `best=second=0` 且 low texture，非正曲率且有双邻居的点为0。`_match` 的严格 `<` 选择最早 tie winner，再因最低 disparity 缺 minus cost 而拒绝。这主要是平坦无信息 cost 的症状，不是 wire 误差的主因，也不能靠允许边界拟合修复。

## 共同有效集：亚像素错误如何通过 gate

首帧共同有效=1696点：Block5 正确1306（77.00%），Block9 正确937（55.25%）。467点由正确变错误，98点由错误变正确，净损失369点。退化467点中442点两方法都选整数43；453点的 Block9 offset 为负，291点恰为 -0.25。该群 GT `(d-43)` median=+0.152527，但 Block9 offset median=-0.25，方向相反。仅2点右图 GT depth correspondence discrepancy>5 mm，因此不能将这整群退化归为右侧真实遮挡。

**像素证据 A [raw 实测 + source]：env0、frame21(index0)、`(y=268,x=347)`。**

- GT `Z=.773192525 m`、`d=43.149105 px`；Block5 `d=43.166668`，Z误差 -0.314 mm；Block9 `d=42.75`，Z误差 +7.218 mm（3D=7.251 mm）。SGBM `d=43`，Z误差 +2.681 mm。
- Block9 `C42=.000774630811`、`C43=.000387315406`、`C44=.001549261622`。曲率 `.001549261622>1e-8`，三点公式给 offset=-.25；LR=0、final confidence=.5，所有现有 gate 通过。
- 左窗 instance-map 是 wire 57/81、table 24/81。`d43` 的 cost 全部来自 table（wire contribution=0、background contribution=.000387315406）；`d44` 更大的 cost 也全部来自 table（.001549261622），而 `d42` cost 来自 wire（.000774630811）。这给出背景/边界掺入如何改变邻近 cost 不对称性的可检查证据。
- 右图按 GT disparity 投影、双线性采样的 rendered depth 与 GT 深度差0。此项是几何可见性 proxy，没有右图 instance mask，不能替代精确双侧 semantic/visibility truth。

**像素证据 B [raw 实测]：env1、frame21、`(y=246,x=392)`。** GT d=43.090305，Block5 d=43、Z误差+1.626 mm；Block9 d=42.75、Z误差+6.163 mm。cost 三元组同 A，但本点 cost 的 wire/background 贡献不同，说明不能把所有 -0.25 点统一声称为 table 单独主导。全 4063点中整窗 winner 与 wire-only 对 GT 最近整数的 cost 排序反转：Block5=64、Block9=214；退化467点中Block9仅21点发生该整数排序反转，进一步说明主要损失不是整数 argmin 改变。

### 离线 counterfactual 的可用结论与限制

| 处理 | 支持集 | coverage | Z MAE mm | Z P95 mm | 正确3D完整率 |
|---|---|---:|---:|---:|---:|
| 原 Block9 | 原 valid | 93.87% | 7.978 | 21.247 | 46.05% |
| 仅以整数 winner 替换 Z，保持原 valid | 固定原 valid | 93.87% | 6.269 | 22.384 | 83.53% |
| CPU完整 `subpixel=false` compute | 重新计算 LR/valid | 94.88% | 6.304 | 22.383 | 83.88% |

固定 valid 代换仅隔离 interpolation 的贡献，不是已修改算法。完整 CPU compute 仍是当前保存 RGB 的离线重处理。后者全图 coverage=6.14%、MAE=47.69 mm、RMSE=78.11 mm、P95=176.04 mm，仍失败。GT disparity 极窄且全部 nearest integer=43，禁止将本结果写成关闭 subpixel 的通用依据或已达到 native gate。

## SGBM：坏尾位置、hand 混入与虚高 confidence

源：[`tabletop_stereo_depth.py:112`](../../../src/go2_pvcnn/control/dh116/tabletop_stereo_depth.py#L112) 使用 OpenCV SGBM，灰度 uint8，window9、`P1=648`、`P2=2592`；147行 `cl=gl.float();cr=gr.float()`。这意味着外部 confidence 是合法范围指示器；所有已接受 SGBM 点得到1，不代表 OpenCV 代价的最佳/次佳差距。SAD 诊断只能归因本仓库 blockmatch 的 cost，不能冒充 OpenCV 内部 SGBM cost 或路径正则项。

| SGBM 首帧 subgroup | 点数 | env0/env1 | 预测 d 范围 px | 真实 d 范围 px | 左9×9含hand点 | LR=0 点数 |
|---|---:|---:|---|---|---:|---:|
| abs(Z error)>20 mm | 278 | 153/125 | 44–51.8125 | 42.794–43.218 | 238 | 41 |
| abs(Z error)>50 mm | 156 | 95/61 | 45.875–51.8125 | 42.794–43.218 | 136 | 8 |

278点全部是 Z 低估（signed mean=-59.404 mm），其 patch hand fraction mean=30.39%，全 wire 仅3.61%；wire fraction mean=50.24%，全 wire为71.99%。这些是测量的关联与局部混合证据；尚未暴露 OpenCV 聚合 cost/path，不能把每一个 SGBM坏点精确归因到特定 regularization 路径。

主要连通块（坐标 `bbox=(x,y,width,height)`）：

- env0 `(467,215,10,12)` 60点，d median46.375、Z误差mean -55.35mm；env1 `(461,217,8,12)` 56点，d median45.84375、mean -44.10mm。
- env0 `(447,222,8,9)` 39点；env0 `(223,307,11,7)` 38点；env1 `(241,297,10,5)` 24点（Z误差mean -118.05mm）。其余块、GT d分布完整保存在 followup JSON。

**像素证据 C [raw 实测]：env1、frame21、`(y=298,x=247)`。** GT Z=.776087761、d=42.988136；Block5 d=42.833332（Z误差+2.805mm），Block9 d=51.037483（-122.400mm），SGBM d=50.9375（-121.117mm）。SGBM右图采样 d=50.875，LR=.0625，confidence=1，因而接受错误但自洽的 correspondence。GT投影右depth差仅 .002563mm。左9×9实例为 wire48/81、table10/81、`rightHand/finger33_link`15/81、`rightHand/base_link`8/81；Block9 GT d43 的 SAD全部来自非wire，而wire-only GT代价为0。该像素的 `winner_cost_parts`/`gt_nearest_cost_parts` 可直接复算大窗混合造成的 blockmatch错配，但不能将此SAD值说成SGBM内部cost。

**像素证据 D [raw 实测]：env0、frame21、`(y=307,x=223)`。** GT Z=.774995148、d=43.048744；SGBM d=51.8125、Z误差 -131.086mm、3D误差136.212mm；右采样 d=51.6640625、LR=.1484375、confidence=1。左9×9为wire20/81、table27/81、rightHand/base_link34/81。此点GT右depth差 -129.966mm，属于几何遮挡/边界proxy显著不一致，和证据C的共同可见错配不同。

![SGBM 首帧 env0（左）/env1（右），红色为固定GT wire ROI内有效且Z误差超过20mm的点](../../../artifacts/stereo-j1-matcher-20261110/sgbm_tail_over20mm_overlay.png)

**[离线 counterfactual]** 仅收紧 LR 不能消除自洽错误：LR≤.25仍保留212/278个>20mm点，正确3D完整率81.96%→81.27%；LR≤.1保留136个坏点，完整率降到70.44%；LR=0仍保留41个>20mm点（其中8个>50mm），完整率只剩19.67%。这些对固定原支持集的阈值诊断没有改动原 qualification gate。

## 下一 J2：按证据排序的最小修改假设

以下均为待验证假设，尚未实施或晋级；保持现有 `StereoDepth` 接口与 RGB-only 输入，不新建平行 perception/PPO 路径。

1. **优先：给现有 `_match` 的亚像素拟合加入可检查的可靠性诊断，并对非抛物线/量化/混合支持研究受限 refinement 或显式 invalid。** 代码位置89–96行；回归用证据A/B，保留整数winner、C-/C0/C+、offset、curvature、LR、confidence，不把“curvature>0”误作亚像素几何准确性。必须在更宽GT disparity、其他距离/方向/材质上验证；不以当前 integer43优势直接关闭所有 subpixel。
2. **匹配支持应与wire边界/遮挡诊断联动，而不是只增大方窗。** 位置50–75行和156行；现有 std只检查局部有变化，无法区分wire纹理、table轮廓和hand边缘。用证据C、全wire214个整窗/wire-only排序反转作离线失败样例，研究 RGB 可计算的 edge-aware support/cost；GT mask仅用于评估，禁止作为 runtime matcher 输入。
3. **SGBM confidence 必须标明不可用或接入真实可计算证据，并保留独立 LR/遮挡/边界原因。** 位置147–161行；对 LR=0坏点建立回归约束。OpenCV路径cost尚未暴露；先保留诊断缺口，不能用常数1当质量概率，也不能仅收紧LR宣称去除坏尾。
4. **全图 low-texture 与 min-disparity tie 明确分类。** 位置77–95行；source支持最早tie winner并因边界缺邻居失效。区分无纹理、搜索边界、真实遮挡、uniqueness；保持不可靠像素 invalid，不能通过允许无信息亚像素输出制造 coverage。

J2验收需同一固定reference ROI、共同支持集、原全图/ROI准确率阈值及coverage并列；必须重新执行授权后的native qualification才能登记。本文不改变 selected resolution、BC/DAgger准入、Teacher/Student训练状态或原生物理成功标准。

## 可复现文件与命令

所有新增辅助输出位于本地 `artifacts/stereo-j1-matcher-20261110`，远端 `/tmp/dh116-stereo-j1-matcher-20261110`；脚本和JSON均为offline diagnostic。源是原只读 `.pt`，未保存或覆盖任何原raw。

- [matcher_rootcause.py](../../../artifacts/stereo-j1-matcher-20261110/matcher_rootcause.py)：SAD流式仪表化、原native复现、全部4063个wire像素top5 cost/GT rank/混合占比。
- [matcher_rootcause_summary.json](../../../artifacts/stereo-j1-matcher-20261110/matcher_rootcause_summary.json)：25输入SHA、逐env拒绝链、复现差异、逐组指标、坏点实例。
- [wire_pixel_records.jsonl](../../../artifacts/stereo-j1-matcher-20261110/wire_pixel_records.jsonl)：全部固定ROI像素明细。
- [followup_attribution.py](../../../artifacts/stereo-j1-matcher-20261110/followup_attribution.py) / [followup_attribution.json](../../../artifacts/stereo-j1-matcher-20261110/followup_attribution.json)：SGBM右图视差/LR、连通块、固定valid代换、实例path分解及例A–D。
- [wire_pixel_records_enriched.jsonl](../../../artifacts/stereo-j1-matcher-20261110/wire_pixel_records_enriched.jsonl)：上述逐像素附加SGBM采样及path fraction。
- [subpixel_counterfactual.py](../../../artifacts/stereo-j1-matcher-20261110/subpixel_counterfactual.py) / [subpixel_counterfactual.json](../../../artifacts/stereo-j1-matcher-20261110/subpixel_counterfactual.json)：完整CPU `subpixel=false`、全图平cost边界、原probe/ref四方SHA核验。

实际解释器 `/mnt/mydisk/xk/envs/dh116/bin/python`；`torch.set_num_threads(2)`，OpenCV线程2。三个脚本实际exit0，源哈希复核均true；无Isaac import，`CUDA_VISIBLE_DEVICES`为空，全部tensor在CPU。source depth SHA为 `36b35189d7576dde7790f22b468b1be4393e7acfbea8060dc0380073aed28008`，与本地生产文件一致。

```sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /mnt/mydisk/xk/envs/dh116/bin/python \
  /tmp/dh116-stereo-j1-matcher-20261110/matcher_rootcause.py \
  --output /tmp/dh116-stereo-j1-matcher-20261110
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /mnt/mydisk/xk/envs/dh116/bin/python \
  /tmp/dh116-stereo-j1-matcher-20261110/followup_attribution.py
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /mnt/mydisk/xk/envs/dh116/bin/python \
  /tmp/dh116-stereo-j1-matcher-20261110/subpixel_counterfactual.py
```

JSON/JSONL输出使用exclusive-create；复现时将三个脚本中的私有输出目录统一换成新的空目录（不得覆盖当前证据）。SSH调用参数为 `-o BatchMode=yes -o ConnectTimeout=8 -o ForwardAgent=no`。没有新native timing、吞吐排名或训练测量。
