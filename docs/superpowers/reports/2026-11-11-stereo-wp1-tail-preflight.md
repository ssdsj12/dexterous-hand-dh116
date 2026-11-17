# WP1：Roll20 surface96 逐点尾部预检

本次只读重放已有原生 RGB、左右 optical-Z reference 与已保存 stereo 深度，不新增原生采集，不启动 Isaac，不改变默认配置、匹配器、物理参数或 5 mm 欧氏误差标准。基线提交为 `ae307a55917519138b6da5179e8254ce2e9a7c8f`。

## 预检结论

surface96 的像素来源真实性通过本 cohort 审计：4 帧 × 2 环境 × 96 点，共 768 个记录；每个 env/frame 内源像素唯一，全部来自实际可见 RGB 候选、有有限正深度的像素，重新反投影后 XYZ 一致。每个 observed_surface 点均通过 RGB 连通区域内最多 3 步的可达检查，源像素可归入一条可见路径；不会从隐藏间隔生成点。它们是可见表面采样，不是 Cosserat 材料节点。128 总 token 中剩余无效中心线 token 仍无效，没有以插值凑满。

此检查从 raw RGB mask 重算，但复用运行时 `_paths_from_mask/_ribbon_pixels`，**不是独立拓扑算法证明**。历史 sampling 文件没有保存 `component_id/track_id`，无法验证历史存储的片段/metric track ID；保存的 `image_track_id` 全部为 -1，保持 unknown，不能授予跨帧跟踪资格。审计结果明确写 `not_recorded/all_unknown`。新增可选字段核验会拒绝错误 component ID 或一个有效 track ID 跨两个可见片段；以后的 sampling replay 保存这些字段，旧文件保持不变。

三维精度资格没有通过：每帧二指 91/96、三指 89/96 在 5 mm 欧氏误差以内；正确完整率的分母仍为该帧真实可见 wire reference 的 2009/2054 个像素，分别为 **4.529617% / 4.333009%**。不能把 91/96、89/96 的样本精度当完整率；也不能把本轮 48 条尾部记录当 48 次独立实验。这里仅有两个物理初态，12 个 env/pixel 位置各重复在 4 帧出现。

主要直接证据是 **整数 silhouette 对应错误**：10 个位置使用 42 px，而 reference 对应为 43.118～43.187 px；另外 2 个位置使用 44 px，而 reference 为 43.212～43.216 px。这产生 14.101～22.390 mm 欧氏误差。所有这些点均被现有 left-depth/LR/range 门控接受，记录的已知 rejection flag 均为 false；`true_occlusion` 未提供，保持 null。

不能把尾部统一归咎于右手遮挡：12 个位置的右 optical-Z 可用，在左 GT 投影处与左 GT 的最大差仅 **0.000136291 mm**，保存的右遮挡 proxy 均为 false。左侧 5×5 窗口只有 3/12 个位置包含 finger 像素；这是窗口重叠证据，不是因果证明。旧数据没有右 instance ID，因此不能进一步确认右匹配窗口内物体身份，也不能把 RGB 颜色覆盖当原生遮挡真值。未重新获取完整匹配代价面，不宣称已唯一证明代价函数为何选中这些错误整数候选。

## 实际帧与身份

run UUID `db63bfc2-6edd-485f-a07c-0ff964b19fdb`；env 0 的 episode 为 `...:0:1`、二指，env 1 为 `...:1:1`、三指；两者 reset_id 都为 1。

| frame_id | simulation timestamp (s) | policy tick | physics tick | 二指正确/有效/参考 | 三指正确/有效/参考 |
|---|---:|---:|---:|---|---|
|21|0.07083333702757955|2|16|91 / 96 / 2009|89 / 96 / 2054|
|22|0.08750000456348062|3|24|91 / 96 / 2009|89 / 96 / 2054|
|23|0.10416667209938169|4|32|91 / 96 / 2009|89 / 96 / 2054|
|24|0.12083333963528275|5|40|91 / 96 / 2009|89 / 96 / 2054|

## 全部尾部像素

下表每行均在 frame 21/22/23/24 出现；对应同一 env/episode。完整 48 条记录保存在 `docs/contracts/stereo_wp1_tail_preflight_20261111.json` 的 `tail_records`，包含每帧原始身份、参考文件 SHA、depth/sample map SHA、源片段与每个重叠拒绝标志。

坐标为 0-based `(y,x)`；右匹配横坐标为 `x-disparity`，不把它取整替代真实坐标。Z 为光轴深度；3D error 为同一左像素反投影的欧氏距离。Lw/Lf/Rw 分别是 5×5 窗口内左 native wire、左 native finger、预测右窗口 RGB wire 候选像素数（右 RGB 不是真值）。

|env|左 pixel (y,x)|slot|预测 / GT disparity px|预测 / GT 右 x|预测 / GT Z mm|3D error mm|Lw/Lf/Rw|
|---|---|---:|---|---|---|---:|---|
|0|(237,418)|46|42 / 43.119882|376 / 374.880118|794.346869 / 773.716629|21.198242|23/0/23|
|0|(257,376)|65|42 / 43.184441|334 / 332.815559|794.346869 / 772.559941|22.004971|23/0/23|
|0|(265,341)|79|42 / 43.118241|299 / 297.881759|794.346869 / 773.746073|20.666580|23/0/23|
|0|(285,299)|98|42 / 43.186487|257 / 255.813513|794.346869 / 772.523344|21.979228|23/0/23|
|0|(301,249)|125|44 / 43.215819|205 / 205.784181|758.240223 / 771.999002|14.100742|21/4/21|
|1|(229,440)|37|42 / 43.120357|398 / 396.879643|794.346869 / 773.708105|21.489345|23/1/23|
|1|(241,420)|46|42 / 43.183285|378 / 376.816715|794.346869 / 772.580624|22.389506|23/0/23|
|1|(249,385)|59|42 / 43.119187|343 / 341.880813|794.346869 / 773.729086|20.875823|23/0/23|
|1|(261,365)|68|42 / 43.184738|323 / 321.815262|794.346869 / 772.554636|21.950325|23/0/23|
|1|(269,330)|81|42 / 43.118018|288 / 286.881982|794.346869 / 773.750067|20.654781|23/0/23|
|1|(293,277)|104|42 / 43.187070|235 / 233.812930|794.346869 / 772.512913|22.125065|23/0/23|
|1|(306,240)|126|44 / 43.211916|196 / 196.788084|758.240223 / 772.068739|14.249200|20/5/20|

全部 768 个采样记录的左 native wire 判断为 true。本次没有用 GT 来重新挑选点或抛弃尾部；所有尾部保留。GT 仅用于离线误差审计，不进入 Actor。

## 亚像素合同核查

原始 native capture 的 depth 配置是 `blockmatch, subpixel=true`；本次采样使用的是既有 `wire_silhouette` 离线重放深度，其实际配置为 `subpixel=false`。`analyze_dh116_wire_depth.py` 显式设置该值；`StereoDepth` 拒绝 `wire_silhouette + subpixel=true`；`match_wire()` 不在 silhouette 分支执行亚像素细化。运行元数据没有虚报该重放已使用亚像素。

本轮不改这个算法，不改默认配置。下一步可以独立评估 RGB 灰度/纹理与 silhouette 约束结合的亚像素候选，但必须保留本次失败地图作同 cohort 对照，不能以 GT 指定视差或仅删除尾部提高分数。

## 可复核产物与 SHA

服务器原始 proof：`/tmp/dh116-stereo-20261110T120000Z/native_roll20_block5/640x480/native_stereo_proof_completed.json`。

|对象|SHA256|
|---|---|
|原始 capture|`733952a7aa7495b08e98b98d3e90e38266350f79630892f27f6b90a19c793235`|
|proof|`fda6bb67d0d9a141baa2352a7b9b8a4cc4cd7f3e90a7aae8ec411f1269de4172`|
|calibration|`02fabb87c548cd982728dac5d6357ca08327258514269b163e880427b2556700`|
|已保存深度 replay summary|`18ca39435a9984dafa2dc962d5578c5c1547287c0fa7a5a4f36bc165df2a3cbb`|
|surface96 sampling summary|`57ce8dd2579ec1cd2f04a416c0ec9aa9a321f13cc54e082488456fb65bf768bd`|
|768 条采样血缘 JSONL|`4cf31d704c05511f47e8c81e4e2b558f4aa9c33e8b680a450495862e4b49efdf`|
|48 条尾部 JSONL|`24f9179ee576b7f91f68ca09b63e9bf570c1e3ca9f316edd86dc6b469ed27c53`|
|审计 CLI 源码|`fa5a31517e573753674753953a082d0f2129f49aa758366b224359c544bb1386`|
|审计 module 源码|`8bffba5a2a08da68746ee93e7d565e119d602abb25b46b6f3c3c4edb8a6da2c0`|

最终新输出为 `/tmp/dh116-wp1-tail-preflight-20261111-v03/output/`，本地 summary/tails 镜像为 `artifacts/stereo-wp-20261111/wp1/tail_preflight_v03/`；相同 SHA 的完整 768 条 lineage 另保存在本地 `tail_preflight_v02/`。v01 遇到原始 `true_occlusion=None` 时发生审计脚本异常；失败输出保留在原独立目录和本地 `tail_preflight_v02/failed_v01.log`。v02 保留 unknown；v03 再增加可选存储身份核验，不覆盖前两版。最终机器报告 SHA256 为 `00c9865a1625c7604e60851d917dadc9f5421c20416375f07396fd5932116e88`。

## 代码与测试

新增 `tabletop_stereo_wire_tail_audit.py`：独立测量像素来源核验、固定参考分母、欧氏尾部误差、真实右坐标及保留未知拒绝源。

新增 `scripts/audit_dh116_wire_tails.py`：逐级核验 proof/capture/calibration/reference/depth/sample SHA；输出完整逐点身份和尾部证据。不会调用 Isaac 或改变模型。

新增 8 个测试，涵盖重复源像素、XYZ 与测量不一致、跨片段/隐藏间隔拒绝、亚像素右坐标、光轴误差不能替代欧氏 5mm 门槛、固定 2009/2054 分母、未知通道保留 null、错误 component ID 和跨片段 track alias 拒绝。先观察缺模块/缺函数及 wrong-component 未拒绝的失败，再实现；最终新旧四组回归 **55 passed in 3.13s**。日志 `artifacts/stereo-wp-20261111/wp1/tail_preflight_v02/regression_components.log`，新增 component 测试 RED 日志 `component_red.log`。

```powershell
$env:PYTHONPATH='src;artifacts/stage7_lifecycle_cpu_deps_v01'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:OMP_NUM_THREADS='4'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_wire_tail_audit.py tests/test_dh116_stereo_wire_depth.py tests/test_dh116_stereo_runtime_wire_tracking.py tests/test_dh116_stereo_wire_segmentation.py
```

离线执行命令（SSH 使用既有只读证据，输出独立目录）：

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src OMP_NUM_THREADS=2 /mnt/mydisk/xk/envs/dh116/bin/python scripts/audit_dh116_wire_tails.py --proof /tmp/dh116-stereo-20261110T120000Z/native_roll20_block5/640x480/native_stereo_proof_completed.json --depth-replay /tmp/dh116-stereo-wire-reliability-v02/roll20_ribbon3_result --sample-replay /tmp/dh116-wp1-segmentation-20261111-v02/roll20_capacity_rmse --output /tmp/dh116-wp1-tail-preflight-20261111-v03/output
```

CPU 审计通过只说明当前采样没有重复、隐藏插值和来源伪装；不等同于新的原生采集、双目深度精度通过、接触测量资格或任何 Teacher/Student 抓取成功。
