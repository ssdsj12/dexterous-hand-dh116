# TASK 2–3　143D 状态审计和离线监督

基线为 `085f6b8d96891b7c975c6382412e629d18e896b3`。逐字段记录按真实 `TaskStateAdapter.field_indices` 顺序生成，包含原 371D 索引、单位、坐标系、归一化、Actor 输入来源、候选标签来源、有效性条件、最少历史帧、未知不确定度和实际监督样本数。字段的 schema 准入与当前数据是否有标签分别记录。[143D 逐字段审计](../../contracts/stereo_task_state_143_field_audit.json)

|来源|候选坐标数|本轮证据|
|---|---:|---|
|关节位置、关节速度|24|SDK 读回在现有运行时只开放前 24 项的训练时标签核验。它们在 Actor 中为直读项。关节位置还需显式归一化上下限。|
|实际提交动作历史|24|由传感历史直读上一动作。|
|Geometry|63|历史 Teacher 数据的核验数为 0。新的有界采集中，相对中心 XYZ 三个字段各通过 6 个来源核验，其余仍缺标签。|
|Velocity|6|需至少两帧有效历史及同钟速度来源，符合条件的 Teacher 样本数为 0。|
|Contact|24|20 项二分类 logits 和 4 项连续回归，符合条件的 Teacher 样本数为 0。|
|Slip|2|需有效物理跟踪窗口及历史，符合条件的 Teacher 样本数为 0。|

Deformation 与 Force Proxy 是额外预测头。当前刚性任务没有真实柔性节点来源，Force Proxy 的 native raw force proof 也未通过。本轮没有开放这两类监督。Estimator 停止输出常数 1 作为预测不确定度。Adapter 报告中的缺失值为 NA，Actor 的数值编码保留原有有效性通道与有限中性值。编码中的 1 表示未知不确定度。

三种 v3 Actor 配置分别运行了 CPU VGA 合成传感测试。三个模式均产生有限的 `[1,24]` 动作。状态模式和 hybrid 在同一个构造的两帧样本上各有 123 个可通过配置及观测门控的字段。其余 20 项为未提供归一化限位的 12 项关节位置、4 项未通过对称性条件的四元数，以及 4 项未开放的反馈/滑移字段。该合成测试验证通路和掩码，字段通过数仅适用于该输入。

## 真实数据检查

- Teacher 确定性均值收集文件 `3648ac9e…` 和原生物理账本 `cbef466e…` 与原核验 SHA 一致。实测 8 个 Teacher 采样时刻、2 个实际初态、0 条持久化 RGB history、0 条逐字段标签 mask。离线入口读取了实际文件并拒绝入训。
- 历史 front VGA `2f0ee494…` 与同运行账本 `de09f674…` 在 run/env/episode/reset/physics step/timestamp 上匹配全部 8 帧。每帧有原生 rigid pose/velocity 和 SDK 深度/线缆实例参考。全部样本是零动作探针，无 Teacher checkpoint，且只有 2 个初态。8 帧的 contact diagnostic 标志为真，力源资格和 formal tracker 资格均为假。与 Teacher 采集的初态哈希相同不能用于跨运行拼接。
- 远端项目和历史实验临时目录搜索到旧单目监督文件，未发现可直接使用的历史 `stereo_teacher_candidates.pt`。
- 在最终冻结源码上，以 guarded Teacher 更新后的独立 checkpoint 和 Roll20 `wire_silhouette_ribbon3` 配置实际执行 `collect_dh116_stereo_teacher.py --candidate-only --state-reference --num-envs 2 --steps 4 --enable-experiment --execute-native`。进程退出码为 0，保存 8 条候选、2 条回合头和 64 条 480Hz 原生样本。每帧两种抓法均有 32 个实际输入点。候选 SHA 为 `4da4f234db2ff77c505d100366bcf30b5c9b2d007f48cfd5b132c1e5a0f8cd70`，物理账本 SHA 为 `da81d600f58864b059cde81131353ce6384be5e84f53bc6446a94bb7b00c6929`。
- 在服务器同路径用 `train_dh116_stereo_state.py` 和上述 SHA 重新读取候选与原始账本。相对中心索引 61、62、63 各有 6 个来源合格的标签，全部来自 policy tick 1–3。首帧缺同钟前一物理读回，线速度缺两帧共享的已验证 track，其他头也没有合格标签。2 个物理初态无法提供三路独立划分，入口保存 `blocked_insufficient_independent_groups` 后退出码为 2，未生成模型。各头的实际训练 loss、MAE、RMSE、P95、梯度和校准结果均为 NA。

## 离线入口及验证

`scripts/train_dh116_stereo_state.py` 消费现有 stereo Teacher candidate 格式和原始 480Hz 账本。它核对文件 SHA、实测初态、动作所有者、每个控制步的连续物理 tick、时间边界、实测相机帧和标定。候选收集运行时在动作前把已采集的相机包绑定到原生记录，复用该包，无额外渲染。

候选收集可显式开启 `--candidate-only --state-reference`。它归档同帧原生 SDK 光轴深度和 Wire 实例 ID，引用的 SHA 随相机帧写入动作前身份。离线入训重读并解析实际实例 ID，只认可同时通过 RGB 深度有效性、Wire 可见性和 5 mm 光轴深度误差的已采样点。刚体相对中心 61:64 还须能用同一运行的同钟物理读回重算并匹配。线速度 68:71 额外要求相邻两帧至少共享一个实际合格的 Wire track ID。对称圆柱的角速度保持关闭。标准候选里只有编码器标签时，离线入口只对满足上述条件的坐标扩大监督掩码。原生光轴深度和实例 ID 保存在监督证据中，Actor 继续使用 RGB 推导的输入。

其他视觉源仍需相应的原生标签核验，Force Proxy 复用现有独立 raw proof。不同初态和源 episode 的连通组不可跨 train/validation/test。优化依序为 Geometry、Velocity、Deformation、Contact、Slip、Force Proxy，只使用三个独立划分均有合格数据的阶段。各阶段记录优化 loss 和梯度，独立划分记录 MAE/RMSE/P95/coverage。规范化坐标逐维报告，旋转误差以 rad、力误差以 N 计。校准使用验证组最大残差，独立验证组不足时半径保持 NA。

CPU RED 首次复现了 6 项缺失能力或不正确语义，随后另有未校准常数、不完整原生帧标识和缺少候选帧绑定的 RED。CPU 合成训练用例实际更新了 Geometry 输出行并核对禁用 Head 权重、独立划分和 NA 校准。实际候选采集只用于核验来源及拒绝原因，未启动状态模型训练或正式 BC/DAgger。

补齐独立参考采集和离线入训后，最终相关测试 **91 passed / 20.76s**。新增 RED 覆盖编码器候选在真实视野条件满足时的几何标签准入，以及参考深度不一致或 Wire mask 缺失时的拒绝。命令如下。

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_state_supervision.py tests/test_dh116_stereo_runtime_task_state.py tests/test_dh116_stereo_runtime_actor_modes.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_training_cli.py tests/test_dh116_stereo_runtime_teacher_identity.py
```

原生回放证据、RED/GREEN 日志与 VGA 三模式结果保存在私有 `artifacts/stereo-reliability-20261111/state/`。
