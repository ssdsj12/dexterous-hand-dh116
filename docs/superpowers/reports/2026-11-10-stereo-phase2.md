# Phase 2：隔离双目RGB、深度与坐标变换

本阶段软件实现与Teacher原生资格分离：用户允许隔离后续开发，Teacher gate仍pending。没有启动Isaac/GPU、训练或采集新原生数据；本报告不授予Stereo准确度、力学来源、Teacher或Student晋级资格。原528个生产/配置文件不修改，只新增独立模块与显式opt-in配置。

## 设计选择与接口

采用一个tiled RGB-only render product内的2×N真实独立相机，而不是移动同一个相机依次左右渲染。两台眼睛的姿态只差左optical +X方向80mm刚性基线；每个环境复制同一rig，外参仅加caller origins。读相机时允许 `sim.render()`，不调用物理推进或world.refresh/task truth。ReferenceTime与同产品IsaacReadTimes在读前后必须一致，caller pre-action timestamp也必须一致。

深度默认采用可执行的纯Torch已rectified SAD block matcher。可选StereoSGBM与一般畸变rectification延迟导入OpenCV，本机没有cv2，明确报missing dependency而不安装、不用渲染深度补位。另一种“仅提供SGBM接口”的方案会使本机无法运行；学习深度方案需要额外训练/数据资格，本阶段不引入。纯Torch分支是可检查baseline，不代表它达到线缆任务精度。

5个新增模块各自职责如下：

| 文件 | 接口与职责 |
|---|---|
| `tabletop_stereo_calibration.py` | frozen `StereoCalibration(K_left,K_right,T_world_left,T_world_right,width,height,baseline_m,distortion_left/right,rectified,version)`；`from_dict/from_json/to_dict`、语义`.sha256`、严格 `validate_readback(...,origins=None)`；optical轴+X右/+Y下/+Z前，K使用pixel-centre u+.5/v+.5 |
| `tabletop_stereo_depth.py` | `StereoDepth(method='blockmatch'|'sgbm',min_disparity=1,max_disparity=64,block_size=5,min_depth_m=.2,max_depth_m=1.2,lr_tolerance_px=1,texture_threshold=.001,uniqueness_threshold=.05)`；`.compute(left_rgb,right_rgb,c,*,left_valid=None,right_valid=None)` |
| `tabletop_stereo_geometry.py` | `backproject(depth,K,valid)`→camera XYZ；`transform_points(points,T)`→world；`world_to_palms(points,actual_palm_pose[B,2,7])`→左右实际腕/掌link坐标；`order_polyline(points,valid,adjacency)`明确关联拓扑后按弧长排序 |
| `tabletop_stereo_camera.py` | `StereoCamera(c,reader,depth_estimator=None,sim=None,origins=None)`；`.from_native(sim,origins,config_path,device='cuda')`lazy构造真实2N Camera prim和RGB-only CameraView；`.sample(identity=...)`输出当前RGB pair+由RGB估计的depth与时序/标定身份 |
| `tabletop_stereo_validation.py` | RGB shape/range、pre-action identity、pair timing验证；`depth_metrics(estimate,reference,valid,reference_valid)`仅用于有单独来源资格的评估，缺有效样本返回NA（None） |

`compute` 输入左右 `[B,3,H,W]` float RGB0..1，返回 `depth_m/disparity_px/valid/occlusion/confidence`，全部 `[B,H,W]`。valid为bool；无效depth/disparity保留NaN，confidence置0仅作为无效诊断，与valid一起使用。禁止depth/GT参数。左右双向匹配、LR consistency、完整patch边界、低纹理、uniqueness、near/far和可选remap mask均参与有效性。

非rectified输入必须先显式调用 `rectify_rgb(left,right,c,return_masks=True)`，得到 `(rect_left,rect_right,new_calibration,left_valid,right_valid)`；后续depth与world几何统一用new_calibration。不能在depth内部悄悄改K/T。OpenCV使用整数pixel centre，函数将K主点减.5再rectify，并在新K加回.5。remap出界mask必须继续传入depth；RGB border fill不能作为真实观测。当前仅水平正baseline pipeline，垂直rig不静默重解释。

camera identity支持单环境dict或长度B的list[dict]；每行caller提供run_uuid、env_id、episode_uid、reset_id、policy_tick、physics_tick_start/end、simulation_timestamp。camera复制并追加frame_id/calibration_sha256，不改caller；现有冲突声明会拒绝。physics range指pre-action state刚完成区间，后续执行区间另记。输出也有 `frame_id/simulation_time/K_left/K_right/T_world_left/T_world_right/native_evidence/render_timing`。CPU注入reader即使声称native也不会取得native_evidence；没有原生测试证据。

`order_polyline`只接受caller由可见线缆分割/skeleton association提供的无分支、无环、连通simple-path邻接图，返回points/valid/order/arc_length_m/association_valid。折返路径沿图连接走，不沿刚体主轴投影排序；遮挡断片、branch或loop拒绝，padding保持NaN/false，不声称已恢复真实完整中心线。当前模块不提供通过仿真节点建立关联的捷径，也尚未实现像素分割/skeleton提取。

## 224×96、75°与80mm基线

新增 `configs/dh116_training/stereo_camera_v1.json` 默认关闭新实验。水平75°得到 `fx=224/(2*tan(37.5°))≈145.96px`。80mm基线使0.5–0.8m距离视差约23.35–14.60px；0.2–1.2m深度范围约58.38–9.73px，落在1..64搜索预算。16mm直径在目标范围约4.67–2.92px，仅说明几何量级，不能保证细线覆盖、遮挡鲁棒性或深度准确度。

整数视差baseline在0.5/0.8m处每1px约对应21.4/54.8mm深度变化。它不足以直接证明0.5mm正式滑移精度；需要亚像素估计、时序/几何验证、真实标定与独立评估。合成整数位移恢复精确，不能外推实物/原生误差。80mm是按工作距离与视差预算选择的候选，不宣称最优。

## CPU验证、RED与当前缺口

先写测试，第一次因新calibration模块不存在按预期collection ERROR（exit1）。初版实现25项通过后，针对per-env identity stamp、刚体变换拒绝、显式rectification新calibration与declared-SHA新增RED，观察5 failed/26 passed，再只修改新增模块；最后remap-valid mask接口先RED（unexpected keyword），随后实现。失败证据保留于工具执行记录，本报告没有把失败次数改写为成功。

最终fresh命令：

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_geometry.py
```

结果：最后新增严格env/reset/tick整数、非空episode与去重identity检查，先观察4项RED再修复；**36 passed**。连同11项state审计和1项旧source freeze，fresh组合 **48 passed，exit0，4.24s**。覆盖合成已知视差的真实matcher、LR/遮挡/纹理/深度range、缺OpenCV报错、标定baseline/姿态/K/readback/SHA、World/palm变换、folded path及无效拓扑、batch identity、fresh/time/读回/禁止rendered depth、remap mask传播、CPU不得冒充native、导入无Isaac、camera源码不调用physics step。

原生adapter目前只按已存在且只读的 `_NativeTiledReader`/CameraView 4.5接口实现，**没有在Isaac执行**。它与独立USD/运行时标定、矩形tile布局、真实native RGB格式/通道、成像覆盖、时间同步都仍需bounded native readback验收。cv2实际rectification/SGBM路径未执行（本机缺依赖），没有将missing-dependency测试称为数值验证。无新的depth MAE/RMSE或原生成功结论；Teacher gate、Stereo promotion、正式DAgger仍pending。

实现参照[OpenCV官方calibration/rectification文档](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)核对pinhole、stereoRectify与initUndistortRectifyMap；SGBM fixed-point disparity按官方约定除16。该API核对不替代本机未执行路径或真实native验收。
# 集成补充：入口与可见线缆

Root集成新增 `tabletop_stereo_camera_cli.py`、`tabletop_stereo_perception.py`、probe/evaluate两个CLI与对应CPU测试。RGB颜色候选经delete-only thinning、非分叉连通拓扑和实际视差反投影产生128个可见弧长点；遮挡不补全，端点跨帧一致性只作候选身份，未取得原生身份资格。此规则不是任意线缆通用分割模型。

组合命令：`python -B -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_geometry.py tests/test_dh116_stereo_camera_cli.py tests/test_dh116_stereo_perception.py`（PYTHONPATH=src，CUDA关闭，现有CPU环境）。实际结果 **43 passed in4.78s，exit0**。测试为CPU synthetic/reference算术；原生采集0次。`python scripts/probe_dh116_stereo_camera.py --check-config`仅验证配置，不启动Isaac。

