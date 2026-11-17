# DH116 原生双目分辨率最终测量报告（2026-11-10）

**10次完成的原生配置采集均通过来源和原始数据复核，但10次 accuracy qualification 全部为 false，`selected_resolution=null`。目前不能选定最终训练分辨率。** 本报告汇总已完成的实验，没有启动新Isaac进程或模型训练。

## 范围与指标口径

10次采集由旧rig四种分辨率、front v3四种分辨率、front roll20的640×480 block5/9组成。每次2环境、4计量帧、2预热帧，seed116；保持原任务480Hz物理、60Hz控制及原guard/成功合同。caller仅在完整RGB/reference pair之间执行zero24控制，不在相机采样中推进物理。40个batch帧、80对双目观测不等于80个独立物理初态，更不构成每抓法100初态的任务验收。

`measurements_verified=true`表示各次冻结源码下的公共measurement复核了producer、真实外部进程exit0、artifact SHA、同步、calibration读回，并从保存的左右RGB重新计算深度；它不表示深度准确或抓取成功。独立rendered reference为**光轴Z（m）**，只供验证，不进入Stereo Actor。

- 下表MAE/RMSE/P95均为joint-valid像素的绝对Z深度误差，转换为mm；全图统计限于声明的reference深度范围。
- **Wire valid覆盖率**是Wire reference-visible ROI中有有效stereo预测的比例，不要求预测准确。
- **≤5mm正确3D点完整率**以同一固定Wire reference-visible ROI为分母；分子同时要求stereo有效，且使用K和pixel-centre backproject后的3D Euclidean误差≤0.005m。它不是Z误差≤5mm，也不是隐藏整条线缆的重建完整率。
- Wire ROI由同帧独立instance-ID、actual Wire根/可见USD渲染几何清单及stage存在读回解析，逐环境匹配。RGB颜色candidate只是有效覆盖proxy，不能代替该Wire mask或准确完整率。
- NA表示没有该来源或没有测量，不补成0。合法测量得到的0.00%则保留为实际零值。

原全局门槛没有调整：MAE≤5mm、RMSE≤10mm、有效覆盖≥50%、至少1000有效像素及2帧。P95另外报告；不能用较好的Wire局部值替代未通过的全局门槛。

## 10次原生采集：全局深度误差

编号同时用于后面的资源表和证据索引。所有行均为`measurements_verified=true`、`qualified=false`；坏值保留，不因不合格而改为NA。

|编号|实验|分辨率|MAE mm|RMSE mm|P95 mm|证据|
|---|---|---|---:|---:|---:|---|
|O1|旧rig|224×96|47.177|89.070|257.296|B|
|O2|旧rig|320×240|49.112|89.721|176.593|B|
|O3|旧rig|640×360|38.941|77.081|210.349|B|
|O4|旧rig|640×480|47.181|95.460|180.925|A|
|F1|front v3|224×96|65.656|103.809|257.474|C|
|F2|front v3|320×240|66.041|107.783|264.350|C|
|F3|front v3|640×360|58.617|103.679|261.427|C|
|F4|front v3|640×480|58.617|103.679|261.427|C|
|R5|front roll20，block5|640×480|43.744|75.889|176.401|D|
|R9|front roll20，block9|640×480|46.038|76.407|164.078|E|

以上全局MAE和RMSE均明显超过原门槛。离线重读另确认R5/R9的全局valid覆盖率仅2.62%/6.03%，也不达50%。机器摘要未提供其余八行的全局coverage，本文不从RGB candidate或Wire coverage反推。

## Wire ROI：准确率与有效覆盖分列

|编号|Wire Z-MAE mm|Wire Z-RMSE mm|Wire Z-P95 mm|Wire valid覆盖|≤5mm正确3D点完整率|有效Wire观测数|
|---|---:|---:|---:|---:|---:|---:|
|O1|NA|NA|NA|NA|NA|NA|
|O2|NA|NA|NA|NA|NA|NA|
|O3|NA|NA|NA|NA|NA|NA|
|O4|NA|NA|NA|NA|NA|NA|
|F1|40.654|52.816|124.116|10.30%|0.99%|208|
|F2|35.956|44.594|92.367|7.41%|0.00%|284|
|F3|38.078|48.449|131.468|3.98%|0.19%|684|
|F4|38.078|48.449|131.468|3.98%|0.19%|684|
|R5|7.350|16.533|34.384|43.61%|32.98%|7088|
|R9|7.978|15.116|21.341|93.87%|46.05%|15256|

旧rig四行的Wire指标保持NA：原v1 instance evidence未保存完整可用的stage/几何解析证明，不能用后来修复的parser追溯授信或补写旧文件。其RGB candidate有效覆盖proxy依次为O1 53.28%、O2 36.42%、O3 19.38%、O4 19.46%，均不等于Wire准确完整率。

front v3取得了可追溯的Wire ROI，但该视角下的稀疏/低精度匹配仍失败。roll20两组使用相同物理初态、cohort和calibration；四帧左右RGB逐值相同，reference depth字节、reference valid和已解析Wire mask也相同。每组固定Wire ROI为16252个观测。**窗口9把valid覆盖从43.61%提高到93.87%，但只有46.05%的固定ROI点满足5mm 3D误差，而不是93.87%准确。**

共同6784个有效点上，block5→block9的Z-MAE/RMSE/P95为6.716/14.340/26.454→7.973/16.628/40.852mm，≤5mm正确比例为77.00%→55.25%。较大窗口提高整体覆盖和正确点总完整率，同时使共同支持集精度变差，不能描述成全面改善。新增8472个valid观测中，44.10%满足5mm 3D误差。以上像素观测数均不是独立物理初态数。

三组cohort分开：旧rig为`248047ea9cbbd83147c940e4f37175c261506c883de3b679148bfe208d499b24`；front v3为`64c4ce7801b0c7ac379450049f33b3566db7c3e90fb4d933c581a7981169eca5`；roll20为`12a14e90fa735a56ae35d4c14526fa7c29f240d4f4a8407d725b3fa44e68ff57`。跨rig变化不能全部归因于分辨率，不能把这10行混成同一cohort作最终选型。

## 显存与吞吐的实际范围

下表A/R为**PyTorch所选设备allocator峰值allocated/reserved（MiB）**；U/F为该设备`mem_get_info`采样得到的used最大值/free最小值（GiB），不是单个Torch模型占用，也不是所有GPU/renderer的总job峰值。独立depth/instance验证render product处于激活状态，相关开销属于本次验证实验；没有Actor网络或训练activation，不能外推正式训练显存。MiB/GiB均按1024进制换算。

时间为每个双环境batch的计量均值。最后一列是从已验证时间计算的`2000/(capture_ms+depth_ms)`双目对/s；R5/R9与原报告已有同名数值一致。它**排除Actor、物理step、optimizer与训练**，只表示capture+depth路径，不能用于完整训练选型。

|编号|Torch峰值 A/R MiB|设备 U/F GiB|capture/depth ms|capture+depth 双目对/s|
|---|---:|---:|---:|---:|
|O1|18.27 / 32.00|8.455 / 15.061|27.803 / 75.694|19.324|
|O2|40.95 / 54.00|8.736 / 14.780|28.523 / 101.205|15.417|
|O3|102.76 / 142.00|9.143 / 14.373|35.248 / 192.518|8.781|
|O4|132.31 / 174.00|9.476 / 14.040|42.769 / 230.161|7.328|
|F1|18.36 / 34.00|8.426 / 15.090|27.420 / 86.054|17.625|
|F2|41.25 / 54.00|8.611 / 14.905|32.087 / 125.902|12.659|
|F3|103.63 / 144.00|9.145 / 14.371|38.866 / 188.444|8.799|
|F4|133.48 / 176.00|9.478 / 14.038|43.249 / 207.537|7.975|
|R5|133.48 / 176.00|9.603 / 13.913|45.432 / 210.705|7.808|
|R9|133.48 / 176.00|9.478 / 14.038|42.181 / 187.324|8.714|

10次均未提供Actor package：`actor_ms=NA`、包含Actor的`stereo_pairs_per_second=NA`，包含物理执行/优化的完整流水线吞吐与训练吞吐也均为NA。不能拿capture+depth吞吐或Actor=0来补这些字段。

## 单次离线SGBM9：不是第11次原生资格实验

在R9已保存的同一首帧双环境RGB上，已有SGBM9完成一次离线再处理：0预热、1次compute，没有新Isaac调用。结果scope为`offline_reprocessing_diagnostic_unqualified`，`native_qualified=false`，没有formal资格登记。其Wire指标为Z-MAE/RMSE/P95 **6.980/17.762/46.881mm**、valid覆盖96.90%、≤5mm正确3D点完整率81.96%（3937/4063个ROI点有有效预测）。与native block9相比，平均误差和正确完整率改善，但RMSE/P95尾部更差。

在共同3734点上，native9→SGBM9的Z-MAE为7.849→5.932mm，≤5mm正确3D比例49.25%→86.56%，同时Z-P95为21.247→33.052mm。改善并非完全由valid集合不同造成，但仍不能称全面达标。SGBM9首帧全局MAE/RMSE/P95为53.473/84.082/208.725mm、coverage4.97%，未通过原全局门槛。

该单次离线耗时511.614ms，scope为OpenCV CPU SGBM加Torch CUDA mask/device transfer并同步，排除render、RGB上传、import和GT计算；无预热/重复，不能与上表native流水线作方法吞吐排名，也不能据此登记原生SGBM资格或选择最终分辨率。

## 证据路径与SHA256

本报告从以下已保存材料取数，未改写原产物。公开机器摘要见 [stereo_native_resolution_results.json](../../contracts/stereo_native_resolution_results.json)。私有输入位于仓库artifact目录；原始native capture/reference/proof/completion及其冻结producer/config身份由各远端报告继续绑定，不把汇总JSON本身当物理证明。

|输入|仓库内路径|SHA256|
|---|---|---|
|10次已验证机器汇总|`artifacts/stereo-phase0-20261110/verified_results_summary.json`|`dcbfcfe13a86abc01dd810800878ed77a51d53cee60370a9faa4fcc1041faf58`|
|roll20离线报告|`artifacts/stereo-roll20-offline-20261110/report.md`|`03085a91bfb6a32dc4fbd6dc5073bc46b85f009cf29b9c8807aa33fb02c9eabc`|
|原native重读/同输入对照|`artifacts/stereo-roll20-offline-20261110/native_readback_and_rgb_comparison.json`|`0c0b5c96d5cac27baff0aa81c63a2ff2a1ef1e56669cba5ca6a889952dd736ca`|
|离线单帧SGBM指标|`artifacts/stereo-roll20-offline-20261110/offline_sgbm9_first_frame_metrics.json`|`f0b7cdc71340bb5d225cf7d2fc234c53acf9abf29bc287901e29d9043cc85397`|
|离线产物清单|`artifacts/stereo-roll20-offline-20261110/artifact_manifest.json`|`fdb3fcef9442768f5540b094b1768b802fe76b0e9c0d674290e78cf17e6a17fa`|
|未修改的全局资格政策|`configs/dh116_training/stereo_validation_v1.json`|`82a12907c342b58c5ac7e14e7bbd0315e35ffe9f219d4248fe4e0c4fc4e228d6`|

各原生报告的远端绝对路径和SHA如下。同一报告可包含多个分辨率；编号映射见全局表。

|证据|远端报告路径|SHA256|
|---|---|---|
|A|`/tmp/dh116-stereo-20261110T114000Z/native_resolution_04/resolution_benchmark.json`|`ff36f78a437762d4ee07409fe888b8b6b9e93af0db221e2dc3ee7ffe02f09768`|
|B|`/tmp/dh116-stereo-20261110T114000Z/native_resolution_05/resolution_benchmark.json`|`c15bf33c497f73e20f600ad14a78062dad9ea6dc738fd27673c71a7de02e4152`|
|C|`/tmp/dh116-stereo-20261110T115000Z/native_front_resolution_01/resolution_benchmark.json`|`eba8e2fb222f67778fe1a7f5159d86c34e0b0455000e075f8a5dd948219591e2`|
|D|`/tmp/dh116-stereo-20261110T120000Z/native_roll20_block5/resolution_benchmark.json`|`5a39cbf2b35348f213e4354de6623a4eeddff7b93a58d9bcca93d0a385aa4a94`|
|E|`/tmp/dh116-stereo-20261110T120000Z/native_roll20_block9/resolution_benchmark.json`|`5f884f2c5c49273988d4a98f5f87aa5c09d98d4037b6db7d9d716bdec0f1f129`|

roll20的原始输入/算法复算另有以下明确绑定；其余各帧reference和wire artifact SHA保留在上述native重读JSON的`input_artifacts`中，不能从另一env/另一run借用。

|对象|远端绝对路径|SHA256|
|---|---|---|
|R5原始capture|`/tmp/dh116-stereo-20261110T120000Z/native_roll20_block5/640x480/stereo_probe.pt`|`733952a7aa7495b08e98b98d3e90e38266350f79630892f27f6b90a19c793235`|
|R9原始capture|`/tmp/dh116-stereo-20261110T120000Z/native_roll20_block9/640x480/stereo_probe.pt`|`be1ba0bf5eadf82d4313e2ecc297bf016851df86a906489939591f57d1bc80d2`|
|R5实际producer参数|`/tmp/dh116-stereo-20261110T120000Z/native_roll20_block5/640x480/producer_config.json`|`97fde8b6542c5e8b48cb4ade87a0b3871bf73e395827b6f84bfc463fcbf1391b`|
|R9实际producer参数|`/tmp/dh116-stereo-20261110T120000Z/native_roll20_block9/640x480/producer_config.json`|`a601eddf554806f025797d939879799f483adfed84d0459115023b3a237a6374`|
|R9实验配置文件|`/tmp/dh116-stereo-20261110T120000Z/snapshot/configs/dh116_training/stereo_camera_front_roll20_640x480_block9_experiment_v1.json`|`5910ca3d7aa7b3211249522473cfd6c972fe6a5a8d60aec706dc6273a7d4ac60`|
|冻结depth实现|`/tmp/dh116-stereo-20261110T120000Z/snapshot/src/go2_pvcnn/control/dh116/tabletop_stereo_depth.py`|`36b35189d7576dde7790f22b468b1be4393e7acfbea8060dc0380073aed28008`|
|离线复算脚本|`/tmp/dh116-stereo-roll20_offline-20261110/analyze_roll20.py`|`bb6b06823c74d31695f70f3d05cc197321aec06975a1c234ed97ea7dd31e3a71`|

roll20两组共同calibration语义SHA为`02fabb87c548cd982728dac5d6357ca08327258514269b163e880427b2556700`，与配置文件字节SHA不是同一概念。native producer/proof、CPU兼容与失败目录的过程保真见 [主集成报告](2026-11-10-stereo-phase0-7-integration.md)；旧失败材料、旧rig evidence和冻结source均保留。

本轮没有Teacher/Student模型训练、MPC执行或历史checkpoint修改；原任务动力学、物理/控制频率、guard与30mm/5s/0.5mm正式成功标准没有改变。相机rig的构图/roll变化为独立实验配置，不覆盖旧rig。**来源复核通过10/10，精度资格通过0/10；Teacher/Student任务成功与完整部署吞吐未测，不能写成0/100或已验收。** 后续须先解决深度错误/离群与可见几何精度，当前不启动正式BC/DAgger/长PPO，也不选最终训练分辨率。
