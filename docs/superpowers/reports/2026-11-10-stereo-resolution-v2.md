# Stereo resolution v2：主精度640×480与subpixel/资源合同

> 本文是该软件阶段的历史快照，以下native=0指当时开发冻结时点。后续10次实际原生采集、失败精度与未选型结论已记录在[最终原生报告](2026-11-10-stereo-native-resolution-final.md)和[集成报告](2026-11-10-stereo-phase0-7-integration.md)，不能将旧NA表当作当前测试结果。

状态：新增多分辨率配置、真正重建K/resize K两个独立接口、streaming subpixel匹配与可配置图像处理batch；CPU软件验证已运行，**native runs=0**。640×480是新的主精度默认配置，不是已通过精度或吞吐资格的选型结论。旧224×96配置字节完整保留；没有修改原控制、480Hz physics/60Hz control、reward或成功判据，没有安装依赖、启动Isaac/GPU、训练或提交Git。

## 分辨率与几何

新增配置：

- `configs/dh116_training/stereo_camera_320x240_v2.json`
- `configs/dh116_training/stereo_camera_640x360_v2.json`
- `configs/dh116_training/stereo_camera_640x480_v2.json`（主精度默认alias）
- `configs/dh116_training/stereo_resolution_experiment_v1.json`

旧`stereo_camera_v1.json`仍为224×96，不重写旧K、配置、checkpoint。新profiles保留同一左右T_world与80mm baseline；按75°水平FOV重author native cameras：

```text
fx=fy=W/(2*tan(75°/2)), cx=W/2, cy=H/2
```

320宽的fx约208.516px，640宽约417.032px；VGA近0.2m的视差约166.8px，不能继续用旧max64。320新profile采用8..112，640两profile采用8..192，覆盖指定0.2..1.2m范围并留边界余量。`disparity_budget()`可按新fx、baseline、near/far计算建议预算。

`rerender_calibration(c,W,H,horizontal_fov_degrees=75)`创建新相机K，保持square pixel及T/B；纵横比改变会改变垂直FOV。`resize_calibration(c,W,H)`只给实际图像resize换算校准，严格使用`K'=diag(sx,sy,1)K`，不重新解释FOV。二者不能混用；像素坐标保持u+.5/v+.5约定。测试覆盖K的resize round trip，以及真实bilinear图像resize后视差4→8、fx60→120、深度1.2m不变。

支持224×96、320×240、640×360、640×480及合理自定义；`validate_resolution`默认资源guard为2048×1536，显式allow_resource_override可超过，不把224当语义上限。左右图像尺寸必须严格相同且等于实际calibration；不偷偷resize/downsample。resource guard不是任何显卡可承受该负载的保证。

## Subpixel与有效性

`StereoDepth(subpixel=True, image_processing_batch_size=...)`对左右方向分别在最佳integer cost及d±1 cost上作抛物线拟合：

```text
delta=0.5*(C(d-1)-C(d+1))/(C(d-1)-2*C(d)+C(d+1))
```

只在三点cost有限且曲率为正时使用，offset限制±0.5；不可拟合、低纹理、uniqueness不足、LR不一致或越depth范围均mask无效。LR使用浮点x-d位置插值right disparity/confidence，并要求实际参与插值的两个相邻点都valid，不能round回integer。两方向都启用相同subpixel逻辑。OpenCV SGBM已有fixed-point结果继续除16，未截断为整数。

PureTorch matcher按一个disparity plane流式遍历，维护best/second与胜者±1邻域，不建立`N×D×H×W`完整cost volume。原RGB/remap masks在**每个candidate cost**前约束完整patch，防止无效填充像素通过相邻cost影响subpixel曲线。该缺口有RED→GREEN：仅改变mask=false处RGB曾改变一个valid像素；加入candidate patch masks后，valid与有效disparity完全不变。

旧constructor默认`subpixel=False`保留整数行为，供旧路径与明确ablation；新accuracy profiles显式True。旧224文件不改，因此若按原文件直接运行，它仍是整数baseline。进行受控分辨率比较时应在内存显式对224施加同一subpixel/block/uniqueness设置并保存实际depth config；integer ablation单独记录，不能把算法与分辨率的联合变化全部归因于分辨率。

## Batch与资源

Depth的image_processing_batch_size分批后拼回，与一次处理全batch的结果/masks一致；disparity始终逐plane streaming。`compute()`保留原五个[N,H,W] Tensor结果，资源诊断在`.last_resource_metadata`，由Camera packet的`depth_resource_metadata`带出。诊断明确full cost volume未分配、当前candidate plane=1、固定cost-state planes=6，不把这当GPU实测峰值。

640×480单个FP32 plane约1.17MiB；完整192-plane单张cost tensor约225MiB，shift/difference/pool还会额外分配。当前streaming只保留固定数量planes及工作张量，但RGB、render product、网络、allocator/cache与算子workspace仍占内存，实际GPU allocated/reserved/free必须另测，当前全NA。

Native reader仍创建并同步render真实2N views，不降低物理/control频率。创建render product前检查全分辨率RGB readback的资源下界，默认1GiB预算，可显式override或减少num_envs。估算仅RGBA8+float32 RGB，不含renderer/assets/transient overhead；没有OOM后偷偷降分辨率。资源拒绝CPU测试证明在导入Isaac前发生。

StereoEncoder / StereoStateEstimator / StereoStudentActor新增统一`image_processing_batch_size`。None保持原单次cat；正整数对共享GN-ResNet18/project按图像分块，训练启用nonreentrant checkpoint保留grad并减少高分辨率activation保留，eval直接分块。没有image resize；GN/dropout0保证确定性。CPU测试对比chunk1与全batch的features及backbone参数gradient。新network config候选batch为2，训练num_envs/BC batch/rollout由独立CLI配置，互不冒充。

## 数据驱动比较与NA

`compare_resolution_results(reports, thresholds=None)`只消费公共`measure_stereo_evidence(proof)`经实际source/sync/calibration/原始RGB重新计算的native测量。真实误差未过资格阈值时仍显示actual metrics、qualified=False；没有native证明时输出NA，不读取自报summary替代。

GPU/timing仅使用公共gate验证后的native CUDA performance artifact；普通report里填GPU=0的CPUfixture不能成为实测。throughput按真实batch与capture+depth+actor总耗时计算。缺actor timing时throughput为NA；RGB候选点完整率proxy不替代真实wire point-cloud completeness。

仅在actual same cohort、qualified、明确MAE/RMSE/P95/真实wire completeness/sample count/吞吐门槛全部满足且GPU指标齐备时选择；按实测RMSE、P95、吞吐排序，不预选低分辨率。配置的default_accuracy_profile=640×480与selected_resolution=None明确分离。

|分辨率|Native MAE/RMSE/P95|真实wire点云完整率|GPU allocated/reserved/free|capture/depth/actor与throughput|
|---|---|---|---|---|
|224×96|NA|NA|NA|NA|
|320×240|NA|NA|NA|NA|
|640×360|NA|NA|NA|NA|
|640×480|NA|NA|NA|NA|

## 本次CPU证据

实现前resolution测试RED为13 failed（缺新helper/subpixel参数）；encoder batch另单独RED为缺构造参数。新fractional synthetic为4.35px、60px焦距、80mm baseline，仅在1792个共同有效像素作算术对照：integer disparity MAE约0.350px，subpixel约0.208px；depth MAE约0.0966m→0.0556m。该低fx synthetic不代表真实线缆精度，不能外推640 native精度。

新测试还覆盖：全部预期分辨率/自定义、rerender/resize区别、T/B不变、legacy配置SHA、近距disparity预算、image batch一致、实际pool输入不存在D维batch、float LR双邻mask、mask填充值不影响cost曲线、encoder gradient、native factory实际读取匹配depth config、GPU/性能NA、受控mock的same-cohort阈值选择。mock只是比较算术，不创建native proof。

最终完整CPU命令、case数与原文件保护摘要在收尾结果中记录；本阶段native=0，未取得精度、完整率、吞吐或最终选型资格。


## 最终收尾结果

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_resolution.py tests/test_dh116_stereo_geometry.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_collection.py tests/test_dh116_stereo_dagger.py
```

本次最终102 passed in 9.74s，exit0，无warnings（resolution20、既有Phase2 geometry36、Phase3 estimator27、Phase4 collection11、Phase5 DAgger8）。旧depth.compute结果增加dict曾触发兼容测试，已把metadata移到属性/Camera packet后保留原五Tensor合同，旧36项全部通过。

逐字节复核Phase0 action manifest中的30个原action/control/training源文件，全部SHA不变。旧224配置SHA256：`f5c53f5c9d3042ee0ce97453a3420dda3ecb7417469d7362be20604b0171d494`，测试固定该值防覆盖。没有声称本次重跑完整Isaac或实测GPU峰值；native count仍为0。

本次文件：修改3个新Phase2模块camera/calibration/depth，补充2个新Phase3模块encoder/state_estimator与新network config；新增resolution helper、20项tests、3个camera profiles、experiment config及本报告。所有这些是新隔离pipeline，不修改基线原物理控制/模型/配置。

源文件与配置SHA256（不含本文，避免自引用）：

|path|SHA256|
|---|---|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_camera.py`|`10bd3a95725bc30d100a9f8eafac4938b0b0477e78454753d94ebd36edb7ef1c`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_calibration.py`|`e0db2dfb0e0a9737e0821b015de96888a32377d0ac0c80ad927c01b6511d2664`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_depth.py`|`36b35189d7576dde7790f22b468b1be4393e7acfbea8060dc0380073aed28008`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_resolution.py`|`cefac066fc212017716a4aea96369c150e686e3fb3fbdeed815bc645027ff15a`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_encoder.py`|`37d6da0274902fa0763c230642f7c22f8eaac7547ad9d78fc1ff1d5aa91fdc3e`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_state_estimator.py`|`9b02a18f8a34196a8c7108f273de7cc24f83cacc3a11497eaf62930ff907709b`|
|`configs/dh116_training/stereo_camera_320x240_v2.json`|`b8748a59df4a15546d443d067b322e6acc1b4da40cb8ae58aefc214b89e1291a`|
|`configs/dh116_training/stereo_camera_640x360_v2.json`|`183fde8cdc100755e1396c6005ad118c00e15653cc1843278fe47eaffae627c0`|
|`configs/dh116_training/stereo_camera_640x480_v2.json`|`3b8a12ae01f762bafdc8a3de7f8d8576ef7920f68b9fc6d1acd2fbbc92dd4d68`|
|`configs/dh116_training/stereo_resolution_experiment_v1.json`|`a323e9204425e2858124decf35e20a202e53f908d98d49214ad1c9d1add24d5f`|
|`configs/dh116_training/stereo_network_v1.json`|`ec0e7356658474275aa4735e1698be00aafad8c20d1b70284cab76e5f0c8e193`|
|`tests/test_dh116_stereo_resolution.py`|`2ecc1c271fcfba8b1bc0bb35fd610a12eacaf7fe0f8a4fe4e097a0fcfcd1c595`|


## 后续VGA lens数值读回与最终冻结

经RED模拟float32 SDK焦距/aperture roundtrip，K-only新增2*float32.eps相对舍入包络，仍保留1e-5px绝对项；T仍atol1e-5/rtol0，+2e-5m拒绝。packet另输出calibration_readback_residuals；不改物理guard/成功门槛。comparison额外透传公共gate已验证的gpu_device_used_bytes，与Torch allocated/reserved明确区分，无来源仍None。最新联合cold/分辨率/旧核心115 CPU tests通过，见cold-trials附录；native仍0。
