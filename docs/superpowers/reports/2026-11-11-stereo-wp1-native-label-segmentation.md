# WP1 原生实例监督分割与实测线缆点采样

基线 `ae307a55917519138b6da5179e8254ce2e9a7c8f`。本轮完成实际代码通路、既有原生RGB的离线复放和CPU少样本诊断拟合。新分割不能替换默认颜色候选路径，当前没有新增原生成功证据。所有原始数据、旧模型、失败模型均保留；没有push或merge。

## 实际改动

- `tabletop_stereo_wire_segmentation.py` 的 `StereoWireSegmentation` 对左右RGB运行共享卷积权重，在原分辨率产生 visible_wire、occluded_wire、unknown_instance 三种logit。`native_instance_targets` 从同frame/time/calibration的真实native instance ID和prim路径重新生成标签，逐侧检查inventory与reference mask。颜色阈值不能作为监督标签。现有历史仅有左侧ID，右侧标签与occlusion标签保持未记录；unknown表示未解析实例，不能解释为已校准的概率或隐藏线缆真值。
- `segmentation_loss` 保留weighted BCE诊断对照，新增独立balanced BCE+masked Dice候选。没有来源的Head有效Mask全false、梯度为零。`train_dh116_stereo_wire_segmentation.py` 实际读取SHA绑定的native捕获、保留原始分辨率坐标、执行少样本诊断拟合、保存模型/Adam/来源。不能授予泛化、双侧资格或专家成功资格。
- `configured_segmentation_depth` 将学得RGB mask接入现有 `StereoDepth.compute`、双向匹配和 `TrackedWirePointExtractor.extract`。NativeCamera与独立RGB runner使用同一包装器；必须显式 `wire_segmentation.diagnostic_only=true` 并绑定真实checkpoint SHA。`processing_contract`保存该SHA及方法。未通过的分割不能悄悄进入旧checkpoint。
- `SensorBatch/StereoHistoryBatch.point_provenance` 传播 0未知、1图像中心线、2实测表面。新的 `point_source_encoding=true` 经 `make_stereo_actor`、`StereoStateEstimator._spatial`、`PalmPointEncoder` 使用独立source分支和共享双手权重，写入checkpoint网络配置并校验。旧模式默认false，不改变旧模型参数拓扑。新分支将未知来源Mask掉；任何来源均不授予Cosserat材料节点身份。旧原始模型仍用其冻结源码，严格源码SHA合同不被绕过。
- RGB可见片段在立体深度缺失时仍可维护 `image_track_id`。此ID使用二维端点/切向/曲率/弧长及历史预测，仅表示传感器图像关联。原 `track_id` 仍要求完整有效三维描述；新二维ID不能用于材料节点或速度监督资格。Reset、坏时间戳、超时及歧义时保守清空或unknown，不连接隐藏区。
- `observed_surface_capacity` 新opt-in配置从32改为96，保持总128槽，32槽保留中心线缺深度NaN，96槽只能来自同可见分量中真实深度有效且唯一的观测像素，不插值。默认仍为原32。独立配置 `stereo_camera_roll20_surface96_diagnostic_v1.json` 与 `stereo_student_hybrid_source_diagnostic_v1.json`。
- `audit_wire_frame`继续保存wire ROI与左有效深度、LR一致、范围、3D有效点的交集，以及可加总首拒绝、重叠拒绝。增加光轴Z与欧氏点误差P99，欧氏RMSE单独列出。`replay_dh116_wire_sampling.py` 对已SHA绑定dense深度比较不同采样配额，不声称更新matcher。
- `ValidationDepthReference` 原生reader之前已读到双侧tiled instance tensor但丢弃右侧。本轮保留右侧raw ID/map，在右侧optical-Z有效区用实际USD库存独立解析，保存 `right_wire_instance_resolution` 与右mask。左右来自相同render产品读回，沿用前后时钟检查，不额外render或推进物理。CPU验证原始右ID归档与同帧拒绝，新原生采集仍需授权。

## 同cohort实际采样对照

两套原始RGB分别来自2026-11-10的Front640和Roll20 Block5，各2环境×4帧。同一物理初态重复帧不计作独立验证。capture SHA分别为 `2f0ee494d1339f7d68d661eb81af23d8578843b935ae21863802c9c1bc60bdec` 与 `733952a7aa7495b08e98b98d3e90e38266350f79630892f27f6b90a19c793235`。匹配/参考只读同一套数据，未进行新native采集。下列每行的4帧结果一致，原逐帧全部保留。

完整率分母为固定native wire有效参考像素，Front二/三2128/2165、Roll20二/三2009/2054；正确性用同像素欧氏3D误差≤5mm。不能将光轴Z误差当作欧氏3D误差。

|相机/抓法|实测surface配额|有效点 / 正确点|正确3D完整率|Z MAE / RMSE / P95 / P99 mm|欧氏3D MAE / RMSE / P95 / P99 mm|
|---|---:|---:|---:|---|---|
|front / 二指|32|15 / 6|0.2820%|24.986 / 35.826 / 68.805 / 68.805|25.825 / 37.029 / 71.122 / 71.145|
|front / 三指|32|14 / 0|0.0000%|17.878 / 19.604 / 37.256 / 37.256|18.672 / 20.556 / 39.389 / 39.404|
|front / 二指|96|15 / 6|0.2820%|24.986 / 35.826 / 68.805 / 68.805|25.825 / 37.029 / 71.122 / 71.145|
|front / 三指|96|14 / 0|0.0000%|17.878 / 19.604 / 37.256 / 37.256|18.672 / 20.556 / 39.389 / 39.404|
|roll20 / 二指|32|32 / 31|1.5431%|2.397 / 4.302 / 3.184 / 16.043|2.431 / 4.335 / 3.220 / 16.133|
|roll20 / 三指|32|32 / 30|1.4606%|2.846 / 5.729 / 11.921 / 21.797|2.892 / 5.809 / 12.034 / 22.151|
|roll20 / 二指|96|96 / 91|4.5296%|2.708 / 4.949 / 6.057 / 21.789|2.748 / 5.015 / 6.176 / 21.981|
|roll20 / 三指|96|96 / 89|4.3330%|3.286 / 5.869 / 20.602 / 21.794|3.345 / 5.977 / 20.710 / 22.138|

96配额提高已测量点的采样覆盖，没有改善dense匹配本身，误差尾部仍不合格。Roll20 dense接受654/702、正确614/661，与前次匹配相同，理论128 token的像素完整率上限约6.37%/6.23%，因此不能将1.5%全部归因为matcher。前视原blockmatch仍是4/9点，表中为轮廓匹配后的15/14点；增加surface配额不能恢复缺失深度。

## 原生标签诊断拟合与失败证据

对原始16个env-frame做CPU拟合，仅native左ID标签有来源。8个时间帧来自2套相机捕获，每套仅2环境，未分配独立validation/test。正负图像裁块由native标签位置采样，未产生右侧伪标签。80-update weighted BCE模型保存为 `50fb2cc0e9f07793dac392d798715a97a570291cd35a8def4c0044395cef886b`，左侧IoU为0。继续独立200-update对照证明输入、真实稀少正例及梯度可学习，不能把80轮失败直接解释成标签错误。

|诊断拟合|Front左分割IoU 二/三|Roll20左分割IoU 二/三|右视图监督|资格|
|---|---|---|---|---|
|weighted BCE 80|0 / 0|0 / 0|缺失|不通过|
|weighted BCE 200|0.584 / 0.698|0.535 / 0.646|缺失|仅同cohort拟合|
|balanced BCE+Dice 200|0.585 / 0.699|0.625 / 0.740|缺失|仅同cohort拟合|

weighted200 checkpoint SHA `a4973744225382a5ce06cd678e7a219332bd7a4a98796784d2af7c6ccd963429`，balanced200为 `7c9c3a60945e17a8363144d6ceebbd9cb18d676240582d2ac1f0428dc9333920`。80版在同cohort depth重放产生大量错误前景并耗时，运行两帧后主动停止；原像素图和log保留，不补写完整运行成功记录。

在各cohort首帧两个环境上将200版真实输出接入原双向轮廓匹配：weighted版Front为0/3点、正确0/0，Roll20为0/0；balanced版Front0/0，Roll20为0/1点且该1点正确。候选联合unknown屏蔽使visible预测显著收缩，左右mask未可靠对应。上述是实际运行失败，未用标签直接替代预测mask，未绕过LR/唯一性/深度范围条件；分割IoU改善未转化为可靠双目几何，不能部署。该短深度对照仅首帧，绝非完整新原生cohort。

## 验证与保留证据

先写失败测试，实际复现缺模块、缺mask路径、缺来源字段、缺P99、缺96配额、缺右ID归档及不平衡loss模式，然后实现。执行针对segmentation/wire/tracking/palms/frame storage/PPO/preprocessing/inference/error diagnostics/reference的CPU回归；最终145 passed（32.45s），输出见 `artifacts/stereo-wp-20261111/wp1/regression_final2.log`。原生启动次数0，CPU模型诊断拟合3次，没有正式Student训练。

服务器只读输入来自 `/tmp/dh116-stereo-20261110T115000Z`、`/tmp/dh116-stereo-20261110T120000Z`、`/tmp/dh116-stereo-wire-reliability-v02`。新输出独立放在 `/tmp/dh116-wp1-segmentation-20261111` 与 `...-v02`，每次保存新目录，不覆盖输入或80版失败结果。执行Python为 `/mnt/mydisk/xk/envs/dh116/bin/python`，`CUDA_VISIBLE_DEVICES=`，CPU2线程。完整命令和机器可读数据SHA在私有 `artifacts/stereo-wp-20261111/wp1/`，汇总 `docs/contracts/stereo_wp1_evidence_20261111.json`。

## 下一次有界原生采集门槛

需要先取得用户授权，再用Roll20 surface96诊断配置，二指/三指各4个独立物理初态、共8初态、每初态4个控制步，同时记录左右RGB、左右native instance IDs、optical-Z、371D、24D和全部reset/frame/tick/SHA。不使用未合格learned masks，不把新数据当成功专家。实测后才能复核双侧监督、遮挡边界与unknown校准。真正隐藏线缆的occlusion正标签仍需独立可验证的原生可见性来源，不能从缺深度直接伪造。
