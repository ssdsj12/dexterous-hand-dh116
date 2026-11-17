# Stereo J2 Evidence-driven Perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Track work with the checkboxes below; this document does not authorize J3 native execution.

**Goal:** 在现有 Phase 0–7 路径上实现可诊断的 matcher 候选、native-ID 监督分割、可拒绝的图跟踪及 depth auxiliary head，并用 CPU 测试与已存同 cohort 数据筛选 J3 候选。

**Architecture:** 保留 `StereoDepth` 默认 SAD/blockmatch 与旧配置的行为，候选通过独立配置显式选择；统一结果合同供已有 perception/diagnostics 消费。分割、图跟踪和辅助深度接入既有 encoder/loss/inference，不创建第二套 Phase、PPO 或 privileged runtime。

**Tech Stack:** 已安装 Python、PyTorch、NumPy、OpenCV、pytest；从零初始化已有 Torch 模块；不新增依赖、不下载模型或权重。

**Status:** 仅实施计划；以下任务均未因本文件而实现、测试、训练或通过 native qualification；J2尚未执行；本轮不push。

## 证据、范围与修改地图

- J1依据：[matcher根因报告](../reports/2026-11-10-stereo-j1-matcher-rootcause.md)及其 SHA 清单；执行前重读最新 J1 全10run报告、readiness、工作区diff、branch/HEAD与保护文件清单。
- 已测：Block9共同1696点中467点从正确变错误，其中442点整数winner仍为43、291点offset=-.25；只做固定valid整数代换不能作为通用修复，GT d仅42.784–43.218。
- 已测：Block5低texture首拒1515点，Block9为0；wire亚像素拒绝均0；SGBM278个>20mm坏点中41个LR=0，已接受confidence恒1。
- 已存10run×4 batch frames=40 batch frames，每batch含2env，共80 stereo-pair观测；重复RGB、同initial、同cohort的观测不计作独立初态。仅严格相同RGB/calibration/reference支持paired比较。
- 拟扩展：`src/go2_pvcnn/control/dh116/tabletop_stereo_depth.py`、`tabletop_stereo_perception.py`、`tabletop_stereo_encoder.py`、`tabletop_stereo_losses.py`、`tabletop_stereo_inference.py`、`tabletop_stereo_supervision_dataset.py`、既有`tabletop_stereo_error_diagnostics.py`；每个扩展先核对保护清单，受保护baseline原内容不覆盖，必要时用可选新adapter挂接。
- 拟新增：同目录`tabletop_stereo_matcher_candidates.py`、`tabletop_stereo_wire_segmentation.py`、`tabletop_stereo_wire_tracking.py`、`tabletop_stereo_depth_auxiliary.py`；分别负责现有接口下的cost候选、像素概率head、图/关联、辅助depth head。
- 拟新增配置：`configs/dh116_training/stereo_j2_{sad_reliable,census,zncc,gradient,coarse_to_fine,perception}_v1.json`；每候选仅一个明确配置和source/config SHA，不覆盖旧实验配置。
- 拟新增测试：`tests/test_dh116_stereo_j2_{fixtures,matcher_contract,subpixel,costs,segmentation,tracking,depth_auxiliary,offline_comparison}.py`，共用`tests/stereo_j2_fixtures.py`生成解析fixture。

## 共用执行规则与合成真值矩阵

- [ ] 每任务先新增具体测试并运行得到语义正确的RED，再实施最小改动得到GREEN；保存完整命令、exit code、测试名/数、版本和日志SHA到独立`artifacts/stereo-j2-<task>-<UTC>/`。禁止用ImportError或收集失败冒充有效RED。
- [ ] 每任务只运行对应测试和受影响旧回归；最终运行全部stereo回归。保留用户未提交修改，不reset/stash；只在实现获授权后执行本计划，不因计划自动训练或提交。
- [ ] 解析fixture由连续函数直接采样双视图，`I_R(u,v)=I_L(u+d,v)`，不得反复插值已经离散的图像制造GT；`d=fB/Z`，边界与真正无对应区域单独mask。
- [ ] 全因子数值测试覆盖fractional disparity `{.1,.25,.5,.75}`×distance `{.25,.5,.8,1.1}m`×wire direction `{0,20,45,90}deg`；调整基准整数/校准保持解析Z与fraction同时成立，所有K/T与单位显式保存。
- [ ] 每组再分别覆盖texture `{constant,ramp,periodic,seeded_noise}`、wire宽度 `{1,3,5,9}px`、无遮挡/左单侧/右单侧/hand-like前景遮挡、亮度偏移、uint8量化；先单因素，不直接全组合爆炸。
- [ ] 退化fixture预期显式invalid/unknown及reason；可观测fixture同时检查disparity误差、valid覆盖与depth误差，不能通过全invalid通过数值测试。

## Task 1：统一 matcher 诊断合同，默认baseline不变

**Files:** `tabletop_stereo_depth.py`、`tabletop_stereo_error_diagnostics.py`；测试`tests/test_dh116_stereo_j2_matcher_contract.py`。

- [ ] RED：`test_default_sad_outputs_unchanged`、`test_matcher_result_shapes_units_and_aliases`、`test_sgbm_confidence_is_not_cost_probability`、`test_invalid_disparity_never_clamped_into_depth`、`test_reference_argument_rejected_by_runtime`。
- [ ] 固定合同；`B,H,W`平面、float disparity_px/depth_m/confidence、bool validity/occlusion；无效depth/disparity为NaN，confidence不伪造：
```python
result = dict(disparity_px=d, depth_m=z, valid_mask=valid, confidence=confidence,
              occlusion_mask=occlusion, diagnostics=diagnostics)
# 保留旧消费者所需 valid/occlusion aliases；默认输出数值不变。
diagnostics = dict(capabilities=dict(confidence_kind='cost_margin_or_unavailable',
    confidence_is_probability=False, true_occlusion_available=False), reasons=reasons)
```
- [ ] capabilities按实际backend填值：SAD为top2相对cost margin；SGBM未暴露margin时标`unavailable`，旧数值1只能标`range_validity_compatibility`；新质量gate拒绝将其当真实置信度。不可用诊断为null+原因，不填0。
- [ ] 拆开left/right texture、no-support、search-boundary、subpixel、uniqueness、LR mismatch、right-sample-invalid、depth-range、depth-edge；保留重叠predicate和固定顺序first-reject计数；`occlusion_mask`若仅proxy必须声明。
- [ ] GREEN命令：`pytest tests/test_dh116_stereo_j2_matcher_contract.py tests/test_dh116_stereo_perception.py tests/test_dh116_stereo_opencv.py -q`；预期全部通过且默认SAD fixture逐值不变。

## Task 2：cost曲线与亚像素可靠性，先解释后改变

**Files:** `tabletop_stereo_depth.py`、`tabletop_stereo_matcher_candidates.py`；测试`tests/test_dh116_stereo_j2_subpixel.py`。

- [ ] RED：`test_fractional_shift_matrix_preserves_observable_coverage`、`test_boundary_missing_neighbor_is_explicit`、`test_nonconvex_and_flat_cost_invalid`、`test_separated_equal_minima_ambiguous`、`test_uint8_step_cost_not_assumed_parabolic`、`test_j1_quarter_pixel_wrong_sign_regression`。
- [ ] fixture断言使用Task共用矩阵；另固定J1像素A的`C42/C43/C44=.000774630811/.000387315406/.001549261622`，断言旧公式offset=-.25且positive-curvature不能代表准确，不能把GT43.149当runtime输入。
- [ ] streaming记录`C-/C0/C+`、整数winner、offset、curvature、搜索/支持边界、近邻与非相邻第二峰margin、局部极小数、量化台阶计数；仅采样ROI cost曲线，禁止VGA全量`[B,D,H,W]`volume。
- [ ] 单因素候选依次为：拟合前可靠性拒绝；混合边界支持下的受限offset；独立的非抛物线局部cost拟合。每次只变一个因素并独立SHA，保留SAD baseline及`subpixel=false`诊断对照，后者不作为默认通用解。
- [ ] 实施公式以可靠性mask先控制合法候选：`z=fB/d`仅对finite且`d>0`计算；不可靠输出invalid并记录原因；depth/confidence/validity同步。正曲率之外明确检查多峰、非凸、边界和量化；不凭GT调单像素阈值。
- [ ] GREEN：`pytest tests/test_dh116_stereo_j2_subpixel.py tests/test_dh116_stereo_j2_matcher_contract.py -q`；解析可观测小图的disparity MAE≤.1px、P95≤.25px且固定observable ROI coverage≥90%；退化样例必须拒绝，门槛只作CPU单元测试，不替代native gate。

## Task 3：Census、ZNCC、gradient与coarse-to-fine候选

**Files:** `tabletop_stereo_matcher_candidates.py`、独立`stereo_j2_*_v1.json`；测试`tests/test_dh116_stereo_j2_costs.py`。

- [ ] RED：`test_census_hamming_monotonic_intensity_invariance`、`test_zncc_constant_window_invalid`、`test_gradient_cost_linear_brightness_offset`、`test_pyramid_disparity_and_K_scale_together`、`test_coarse_wrong_peak_can_escape_local_band`、`test_all_costs_share_validity_contract`。
- [ ] Census固定5×5邻域bit比较+Hamming；ZNCC固定9×9、方差≤1e-8时invalid；gradient固定中心差分后L1。各候选复用同一range、LR、subpixel可靠性与结果adapter，先分别替换cost，不叠加调参。
- [ ] coarse-to-fine单独配置为`.25→.5→1`，每层同步缩放K和disparity，最终原分辨率refine；局部±2px若无可靠候选须全range重试或invalid，不能永久锁住错误粗峰。原图输出尺寸、坐标与scale trace必须保存。
- [ ] GREEN：`pytest tests/test_dh116_stereo_j2_costs.py tests/test_dh116_stereo_j2_subpixel.py -q`；沿用相同解析矩阵与数值/coverage要求；每候选单独保存失败，不合并成“新matcher已合格”。
- [ ] RAFT/IGEV/HITNet等学习型stereo留在依赖/license/权重来源评审清单；当前不下载、不安装、不实现替代Phase。

## Task 4：native-ID标签与RGB分割head

**Files:** `tabletop_stereo_wire_segmentation.py`、`tabletop_stereo_supervision_dataset.py`、`tabletop_stereo_encoder.py`、`tabletop_stereo_losses.py`；测试`tests/test_dh116_stereo_j2_segmentation.py`。

- [ ] RED：`test_native_wire_label_requires_stage_resolved_binding`、`test_color_threshold_cannot_be_training_truth`、`test_missing_right_instance_labels_remain_unknown`、`test_segmentation_probability_shape_and_masked_bce`、`test_unknown_pixels_have_zero_gradient`、`test_rgb_only_segmentation_forward_rejects_reference`。
- [ ] 读取native wire instance-ID/stage解析binding、raw SHA、frame/env/calibration；缺ID映射/右侧label/完整inventory时对应supervision mask=false并列原因，不从颜色或左图投影伪造右GT。几何nodes只留标签端。
- [ ] `StereoEncoder`增加显式feature-grid访问，旧`forward`默认仍返回原tokens/mask；像素head复用共享RGB features，不复制backbone。输出`probability/mask/confidence/visibility_unknown`，未训练head标`untrained`，禁止报告semantic accuracy已提升。
- [ ] 监督复用`masked_bce(logits,native_wire_mask,source_valid & image_valid)`；未知/遮挡像素不补label，GT不进入runtime。按cohort/物理initial划分，重复帧不得跨train/validation泄漏。
- [ ] GREEN：`pytest tests/test_dh116_stereo_j2_segmentation.py tests/test_dh116_stereo_estimator.py -q`；只用CPU小图一次forward/backward核验shape、finite梯度、mask，不启动学习曲线或监督训练。

## Task 5：simple-path图、弧长与跨帧关联

**Files:** `tabletop_stereo_wire_tracking.py`、`tabletop_stereo_perception.py`；测试`tests/test_dh116_stereo_j2_tracking.py`。

- [ ] RED：`test_simple_path_requires_two_endpoints`、`test_branch_loop_and_disconnected_fragments_rejected`、`test_occluded_gap_not_interpolated`、`test_tracking_uses_tangent_curvature_arc_and_stereo`、`test_reset_and_frame_gap_invalidate_identity`、`test_unknown_depth_does_not_create_3d_token`。
- [ ] 复用既有thinning建立8邻接图，记录components/degrees/cycles；仅单component、2端点、其余degree2的simple path做弧长排序。branch/loop/disconnected分别给reason，不任意剪枝或连接隐藏段。
- [ ] 输出每点`confidence,visibility,track_id,arc_length,frame_id,env_id,reset_id`；关联联合弧长、切向、曲率、stereo consistency与历史预测，匹配不唯一时unknown，不只最近距离。保留现有128-token合同和显式point_valid。
- [ ] GREEN：`pytest tests/test_dh116_stereo_j2_tracking.py tests/test_dh116_stereo_perception.py -q`；解析3D曲线、反向排序、交叉近邻与中途遮挡序列验证身份拒绝；没有时间变化的旧四帧不能充当native tracking qualification。

## Task 6：独立depth auxiliary head与source-masked loss

**Files:** `tabletop_stereo_depth_auxiliary.py`、`tabletop_stereo_encoder.py`、`tabletop_stereo_losses.py`、`tabletop_stereo_inference.py`；测试`tests/test_dh116_stereo_j2_depth_auxiliary.py`。

- [ ] RED：`test_depth_aux_rgb_feature_only`、`test_depth_aux_never_overwrites_matcher_depth`、`test_depth_aux_masked_loss_excludes_unqualified_reference`、`test_empty_reference_is_na_metric_and_zero_loss`、`test_depth_aux_prediction_nan_on_valid_target_fails`、`test_actor_na_latency_stays_null`。
- [ ] 用共享stereo feature-grid经轻量Torch decoder输出`depth_pred_m/validity/confidence`，可预测inverse depth但显式换算米；matcher输出与learned输出独立命名，禁止隐式fallback或把两者最佳像素拼为单方法。
- [ ] `StereoEstimationLoss`扩展独立命名`depth_auxiliary`，复用`masked_smooth_l1`；mask=`reference_source_valid & alignment_valid & finite_target & range_valid & image_valid`，不得用模型低confidence缩小监督mask逃避困难像素；有效标签处非finite预测应失败。
- [ ] GT rendered depth只在dataset/loss/eval端；runtime签名拒绝reference、wire nodes与privileged state。Actor仍按已有source合同处理NA，不补零、不开放force/contact监督。
- [ ] GREEN：`pytest tests/test_dh116_stereo_j2_depth_auxiliary.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_inference.py -q`；128×96、2env一次CPU backward，输出未训练状态；报告全图/固定wire ROI误差与coverage但不称已训练精度。

## Task 7：同cohort离线对照、资源与J3交接

**Files:** 复用既有J1分析CLI/`tabletop_stereo_error_diagnostics.py`；测试`tests/test_dh116_stereo_j2_offline_comparison.py`；独立`artifacts/stereo-j2-offline-<UTC>/`与J2报告。

- [ ] RED：`test_comparison_requires_same_rgb_calibration_and_reference_sha`、`test_fixed_roi_and_common_support_both_reported`、`test_duplicate_frames_not_independent_trials`、`test_missing_labels_do_not_shrink_denominator_silently`、`test_cpu_diagnostic_cannot_promote_native`。
- [ ] 每配置先首帧2env筛查，再最多完整40 batch frames一次；不同rig/rgb只分层报告，不造paired对照。保存每frame/env所有reference/predicted/intersection count、coverage、MAE/RMSE/P50/P90/P95/max、≤5mm正确3D完整率、拒绝链与cost/GT disparity直方图。
- [ ] 固定reference全图/wire/中心窄带/边缘/table/hand/双侧可见proxy/单侧遮挡proxy分区；分别列每方法intersection误差及两方法common support误差，空/未证实区域用NA+缺失路径/SHA，不选小valid集掩盖误差。
- [ ] CPU预算：仅现机已安装环境，`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`且Torch/OpenCV各≤4线程；同时只一个候选，2env、image chunk1；首次单frame≤120s且RSS≤8GiB，超限终止该候选并保留失败；整40batch≤30min/候选，最多5个cost/refinement候选，不做自动grid search。
- [ ] 可选GPU3仅离线且另起进程：首次单frame先确认GPU3无占用，进程peak allocated/reserved均≤4GiB、≤60s；超限仅停止本候选。不得启动Isaac/Kit，不与native或训练并跑；CPU为默认，当前计划本身不启动GPU。
- [ ] 不装包、不升级环境、不下载权重；受缺依赖/license/权重许可阻断时保存明确原因。记录RSS、耗时、设备分配/保留/空闲显存；没有Actor package时actor latency、完整sensor-to-action latency、policy throughput一律NA，不能用capture/matcher吞吐代替。
- [ ] GREEN：`pytest tests/test_dh116_stereo_j2_offline_comparison.py -q`后运行`pytest tests/test_dh116_stereo*.py tests/test_stereo*.py -q`；只在全部相关CPU检查通过后整理J2候选矩阵，失败配置保留原始日志/输出/SHA。
- [ ] 原准确率limits、≤5mm正确3D metric、coverage/min-frame/throughput gate与promotion证据链不改；`selected_resolution`、正式BC/DAgger准入、Teacher/Student训练状态均不因离线改进自动改变。readiness分开写implemented/CPU/native_capture/accuracy/physical/deployment。
- [ ] J3仅交付申请：最多1个候选×1个rig×2env×4测量帧、单Isaac进程，附resolved config/source SHA、估计显存/时限、相同cohort计划与失败保留目录；在J1/J2核查通过后单独申请，不在J2直接运行。物理480Hz/控制60Hz、材料/接触/执行器及Lift≥30mm、Hold≥5s、Slip≤.5mm等原生门槛保持。

## 交付检查

- [ ] 提供变更文件/理由、实际RED/GREEN日志、resolved config、公式/shape/单位、所有input/source/config/output SHA及资源峰值；只将已验证状态标真，未训练head和未native测量候选明确pending。
- [ ] J2结论逐项标“source事实/CPU实测/已存native离线实测/假设”；保存J1像素A–D与新增反例的前后结果，不删除坏尾或失败run。没有资格证据时不宣称J2算法合格、策略已训练或物理任务成功。
