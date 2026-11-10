# Phase 3：Stereo Transformer estimator 设计与验证记录

当前状态：Phase0封存497e290后新增隔离的五模块与两配置，完成27项CPU测试。网络从scratch初始化、无pretrained权重；尚未训练或原生验收，不授任何Teacher/Student/native promotion。

## 约束与设计选择

保留旧3844D RGB-D Actor、371D privileged Critic、24D controller及既有checkpoint字节不变。新增五个独立模块与两份版本配置，运行时依赖仅PyTorch/标准库，不导入torchvision、Isaac或旧训练loop。不安装依赖。

比较三种实现路径：共享ResNet18后只做池化MLP计算较省，但缺少所需双目token交互和显式时序；全分辨率图像/点云统一Transformer保持大量空间信息，但默认224×96计算开销明显增加。选择共享ResNet18 `[2,2,2,2]` BasicBlock、左右cross-attention、depth/point/proprio分组空间tokens，再做四帧时序Transformer。该路径满足规范并允许CPU32×32图像验证。

默认embedding128、4 heads、2 temporal layers、4 history；主干默认64/128/256/512通道。GroupNorm替代BatchNorm以支持小图像和batch1训练，BasicBlock拓扑和ResNet18主路径层数不变。不会加载互联网预训练权重。测试缩小embedding至32/64，保持真实八个BasicBlock。

## 文件与接口

|新增文件|责任|
|---|---|
|`tabletop_stereo_observation.py`|严格SensorBatch、StereoHistoryBatch、四帧StereoHistory；纯部署输入，mask及reset边界|
|`tabletop_stereo_encoder.py`|纯torchResNet18 BasicBlock、共享左右backbone、双向stereo cross-attention、空间/视图位置编码|
|`tabletop_stereo_temporal.py`|带明确time encoding的4head2layerTransformer；正确处理全invalid history|
|`tabletop_stereo_state_estimator.py`|depth/points/proprio空间融合、多头估计、仅sensor-history输入的StereoStudentActor|
|`tabletop_stereo_losses.py`|读取Phase0逐维合同、masked SmoothL1/BCE、SO(3) geodesic、force逐来源资格限制|
|`stereo_network_v1.json`|架构版本、真实ResNet18、时序超参数、canonical actionSHA|
|`stereo_state_estimator_v1.json`|head尺寸、Phase0state合同路径、损失有效性策略和候选权重|

`SensorBatch`字段固定为：`left_rgb/right_rgb float32[B,3,H,W]`、`depth_m/depth_confidence float32[B,H,W]`、`depth_valid bool[B,H,W]`、`points_world float32[B,128,3]`、`point_valid bool[B,128]`、`proprio float32[B,62]`、`dt/image_age float32[B]`、`image_valid bool[B,2]`。dataclass构造和`from_mapping()`拒绝任何额外truth/phase/Teacher字段。identity列表由collector独立校验，不混入模型输入。

RGB范围0..1、proprio/timing严格finite，dt>0，image_age≥0；每手wxyz非零。无效depth/point允许NaN，但只有mask为false时可接受，encoder以`torch.where`清除并携带false mask，不生成GT补值。proprio顺序为q12/qd12/左右palm xyz+wxyz14/previous actually-issued24。

`StereoHistory.update(sensor, reset_mask=None)`先清指定env所有历史mask，再append当前真实frame，返回`StereoHistoryBatch`，其每个sensor字段为`[B,4,...]`，附`history_valid bool[B,4]`。返回值拥有自己的数据；不会因后续update/输入原地修改而改变。两个dataclass支持`.to(device)`、`.clone()`、`.detach()`、`.index_select(indices)`，history另有`cat(list)`供PPO批处理。

`StereoStateEstimator.forward(history)`只接受`StereoHistoryBatch`，返回：

```text
latent             [B,D]
privileged_state   [B,371]
geometry           [B,128,3]
velocity           [B,128,3]
deformation        [B,128]
contact_logits     [B,2,3]
force_proxy        [B,2,3]
slip_logits        [B,2]
slip_velocity      [B,2]
```

`StereoStudentActor`持有estimator和`log_std[24]`。`forward(history)`/`latent_mean(history)`输出unboundedmu24；`deterministic(history)=tanh(mu)`。动作分布数学复用existing `tabletop_squashed_gaussian.py`，不新建另一套Gaussian。构造时强制canonical action-only SHA一致。Actor使用estimator latent，辅助371Dhead的训练目标不通过参数进入forward。

## 空间、时间、遮挡与身份

共享backbone提取左右空间tokens；明确二维空间位置及视图编码，双向cross-attention使用image_valid key masks。全缺失视图不对全mask softmax，采用安全零token路径，并保留可部署proprio输入。该mask只代表视图级缺失，不声称已恢复真实逐点左右遮挡关系；逐点可见性由Phase2有效性/置信度侧信息表达。

depth/confidence/validity经显式空间网格token编码；points输入是统一world框架，附worldframe/点序编码，并保留pointmask。proprio有独立token及已知frame身份，不把不同frame坐标直接相加。空间融合后形成每帧embedding，再加入固定可重现time encoding进入Transformer。padding历史不会影响输出，所有invalid历史输出finite零latent。

128点的顺序定义为collector提供的弧长顺序/跨帧一致身份；网络保留该顺序，不能凭无拓扑点云自行声称已完成排序或跨帧重识别。没有真实同身份/同顺序证据时，geometry temporal监督必须mask，而不是拿相邻数组行充当同一物理点。

## Loss合同

`StereoEstimationLoss`从`docs/contracts/stereo_privileged_state_contract.json`读取371行与`rotation_groups`。强制排除`deployment_allowed=false`、`student_source=unobservable`、`loss_type=none`、`supervision_status=NA`。conditional来源必须逐样本逐维sideband mask；`validity['privileged_state'][B,371]`还需与`validity['state_sources']`中每维`required_validity_keys`相交，缺键默认false。finite不等于来源有效。原始四元数组使用归一化后的SO(3) geodesic，q/-q等价，排除其各维普通Huber项；圆柱object orientation还须显式symmetry可辨识mask。

`forward(predictions, targets, validity, force_qualification=None)`返回各head loss、total及validity coverage。geometry/velocity/deformation、slip速度采用SmoothL1，contact/slip概率采用BCE-with-logits。masked reduction先`torch.where`清除无效NaN再计算，分母由mask×weight构成；全invalid为可微零，无NaN、无无效标签梯度。

force_proxy固定解释为每手thumb/index/middle三digit accepted-pad normal代理N；没有硬件触觉输入，不是world XYZ向量。除同shape validity外还需要逐元素`source_valid`、`contact_valid`、`attributable`三个布尔张量，shape严格`[B,2,3]`。另须具备`source_sha256`、`units='N'`、`coordinate_frame=accepted_pad_normal`、`partner_attribution=verified_object_partner`、`per_digit_source[2][3]`、`sign_convention=compressive_positive_on_object`及`evidence_scope='native_qualified'`来源metadata；缺一项或`evidence_scope='cpu_fixture'`必须block并返回coverage0/明确原因。不接受单个全局bool；来源未审查则零监督coverage，不能把零填充值当有效力标签。371D中的force相关字段也遵守Phase0条件mask，不从finite自动取得资格。纯CPU schema/mask算术测试不会赋予任何实际native资格。

默认损失权重只是工程候选，未经独立holdout校准，不声称最优。诊断任务internal标签不进入Actor恢复目标。

## RED→GREEN与验收计划

- [x] SensorBatch字段、shape/dtype/device、finite、非法privileged键、invalid-depth/points策略。
- [x] reset/history清理、样本复制、to/detach/index_select/cat；append当前真实frame后返回。
- [x] 真ResNet18八个BasicBlock、左右共享权重、4head2layer时序及明确pos/time编码。
- [x] 指定heads维度、CPUfinite、改变有效左右图像会改变编码、改变invalid点不会改变结果。
- [x] allinvalid视图/深度/点/历史的finite和padding隔离，非finite有效输入拒绝。
- [x] canonical actionSHA拒绝、Actor仅history输入、deterministic=tanh(mu)、existingGaussian logprob可用。
- [x] unavailable/internal371维强制exclude、state宽度拒绝、所有invalid损失无NaN/零梯度。
- [x] q/-q geodesic不变，force逐来源mask，BCE/SmoothL1参与反传。
- [x] 运行CPUtests与源文件SHA保护；记录命令/exit/测试数。

尚未进行训练、原生评估、误差指标holdout或task success验收。CPUforward/backward只证明软件接口与算术，不证明状态估计精度。

已执行RED：`PYTHONPATH=src; CUDA_VISIBLE_DEVICES=''; PYTHONDONTWRITEBYTECODE=1`，既有CPU解释器执行`-m pytest -q -p no:cacheprovider tests/test_dh116_stereo_estimator.py::test_phase3_modules_start_as_explicit_red`，`1 failed`、exit1，明确因新模块尚未创建。没有在Phase0封存前写入src。


## 最终验证与交付

新增9文件：前述5个src模块、`configs/dh116_training/stereo_network_v1.json`、`configs/dh116_training/stereo_state_estimator_v1.json`、`tests/test_dh116_stereo_estimator.py`以及本报告。原3844D视觉/371D特权/24D控制源码不变。

完整RED：实现前执行整个新测试文件，`20 failed in 1.70s`、exit1，失败原因均为明确缺少新模块。GREEN命令：

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_estimator.py
```

最终结果：`27 passed in 4.58s`，exit0，无warnings。覆盖真实双目共享backbone、eight BasicBlocks、缺失view/points/history、显式时序位置、batch1 train无BatchNorm/无dropout、backbone实际梯度、现有saved-latentGaussian接入、全部heads尺寸、严格typed输入、action/state SHA拒绝、371D逐来源mask、q/-q group geodesic、SmoothL1/BCE反传、force证据缺失/CPU fixture拒绝及masked MAE/RMSE/P95的NA。

`git diff --check` exit0；`git diff --name-only -- src scripts configs`无输出（所有Phase3内容为新增文件）。独立逐字节复核Phase0 action manifest中的30个action/control/training源文件SHA，全部不变。完整旧source528项的基线由root Phase0封存保护；本报告不把30项检查夸称528项重验。

`StereoHistoryBatch.as_dict()`仅返回12个Tensor（11个sensor字段+history_valid），可由`StereoHistoryBatch(**tensor_tree)`重建，用于独立PPO树存储。Actor不会读取critic/Teacher参数。`per_dimension_metrics()`为通用masked指标，原始四元数分量误差不能替代独立SO(3)误差；无有效标签的指标返回NaN表示NA，loss本身全invalid为可微零。

边界：仅CPU软件验证；未运行Isaac、GPU、训练epoch、checkpoint写入或真实采集。未证明点序重识别、逐点遮挡关系、估计精度、原生力代理资格、Teacher/native任务成功。force默认invalid；配置权重未holdout校准。Teacher、stereo、state estimator、DAgger、最终Student仍须各自独立原生资格验收。

新增文件SHA256（不含本文，避免自引用）：

|path|SHA256|
|---|---|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_encoder.py`|`37d6da0274902fa0763c230642f7c22f8eaac7547ad9d78fc1ff1d5aa91fdc3e`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_temporal.py`|`32829d4df8ba56f959223048274fcf72f32ce390d6072e4c6111fc2d208454a2`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_state_estimator.py`|`9b02a18f8a34196a8c7108f273de7cc24f83cacc3a11497eaf62930ff907709b`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_losses.py`|`2d390c7f61f87ea27e1aca26a637b86128f38e671c4e037c46b7c9e9e230af26`|
|`src/go2_pvcnn/control/dh116/tabletop_stereo_observation.py`|`169062c10a622ddc851bf7360e9c6df0b1917c26ca9fd76b82ed85582a17fa08`|
|`configs/dh116_training/stereo_network_v1.json`|`ec0e7356658474275aa4735e1698be00aafad8c20d1b70284cab76e5f0c8e193`|
|`configs/dh116_training/stereo_state_estimator_v1.json`|`fbf5a53378c4de0016c08a88e5fa7ea739f9bb96db12fcae6eef2c952dbc5b48`|
|`tests/test_dh116_stereo_estimator.py`|`0a8d427b851bbcbb11ffe05d6eb18e9337b34088247da6ebd1b4a2cd8488dea1`|

未修改的关键源码SHA详见`docs/contracts/stereo_action_contract.json`和Phase0 source manifest；本次已按前者30项逐字节复核。


## Peer review后的force资格加固

Phase4/5期间peer review发现仅metadata的`native_qualified`字符串和64hex不能授予物理来源资格。先加RED断言，观察metadata-only错误开启force loss（0.5）；随后增加lazy公共gate `tabletop_stereo_promotion.verify_force_qualification(binding)`，真实proof/producer/native rawartifact路径及字节SHA须通过重读验证，否则force coverage始终0并返回blocked原因。推理Actor不导入该训练gate。

CPU loss梯度正例仅monkeypatch proof verifier以隔离mask算术，不创建或宣称native proof。真实CPUfixture和metadata-only两条路径仍blocked。此前SHA表已更新至加固后源码/测试；本次依旧没有运行任何native力学资格实验。
