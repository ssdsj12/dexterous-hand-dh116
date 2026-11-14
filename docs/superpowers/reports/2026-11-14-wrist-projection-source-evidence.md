# 原手腕投影输入与真实失败重算（2026-11-14）

## 修改及边界

新增 `tabletop_wrist_projection_evidence.py`，独立opt-in，不改变默认控制、物理或阈值：
- `capture_geometry`同一pre-action SDK/rod时刻再次只读拷贝左右23-link位姿、34个原碰撞包络、
  body/link/prim身份和base索引，核对实际reader身份未变，不推进物理。
- `record_projection`调用原投影函数，保存全部原输入，输出目标保持逐位相等。
- `validate_record`由原始link和本地顶点重新计算净空、最低shape和base位姿，再重放原投影，
  核对safe_z、速度/运输限制reason、标志及root/angles。
- `ProjectionEvidenceMixin`在原owner中局部安装，正常与异常退出均恢复原函数；
  原生handle引用不进入账本，新捕获帧在finally中清理。
- `RetainedGeometryMixin`只在实际raw已成功读取后，保留目标检查异常的raw与失败原因；
  不把失败packet设为passed，读取本身失败仍不能制造数据。

新增 `audit_dh116_left_pinch_preprojection.py`，对已消耗失败实验SDK175→176作只读CPU重放。
验证同一env/episode/reset/rod与连续SDK边界，不拼接不同环境的峰值。

## 失败因果链已精确重建

来源是上一包的真实失败账本，不是新采集。由实际SDK175 link位姿和锁定USD的原碰撞包络，
重算SDK176的原控制器所有投影输出，最大数值差为 **0**。

|侧|最低原碰撞包络|pre净空mm|safe_z偏移mm|请求Z偏移mm|实际Z偏移mm|
|---|---|---:|---:|---:|---:|
|左|finger13_link|2.153580850|-1.180440967|-1.181250000|-1.180440967|
|右|finger22_link|2.690211975|-0.963420917|-0.393750000|-0.393750000|

左实际base Z=0.41968244314193726m，reset seed Z=0.42080930325897004m。
原公式 `safe_z = measured_z - seed_z - (clearance - 0.0021)` 与记录完全一致。
输入common lift=0、allowed/live=true来自冻结owner代码常量，报告明确区分它们与实测字段。
没有修改这些值来制造结果。

之前“静态路径通过”不足以证明移动手腕下不会触发原safe_z。
下一独立实验必须明确采用原投影后的真实目标，并继续保留全部原几何/关节/速度/深度保护，
不能把旧无投影轨迹直接改判为成功。

## CPU验证

35通过、0失败、0跳过，229.76秒：
- 真实Python owner＋明确合成SDK buffer的232测量步完整路径；
- 原参数读取在SDK11失败后的9条测量前缀、terminal留存与正常关闭调用；
- 逐步projection输入重算、不更改原目标；
- 修改clock、episode、shape、pose、base、clearance、request、seed、flags、source及输出均拒绝；
- 真失败账本的源几何重算6项。
这些不计作新Isaac原生物理验证。

测试先保存缺失实现失败日志。开发中一次owner重放因测试期间源文件指纹改变而被拒绝，
保留该日志；最终冻结文件后重新运行完整35项并通过，未绕过指纹校验。
全套最终日志SHA：
`557081617766d0d1cf94f09d174e86c9a2dca51af0307f65d41607e004cbf93f`。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_wrist_projection_evidence.py tests/test_wrist_projection_evidence_owner.py tests/test_dh116_flexible_wrist_projection.py tests/test_left_pinch_preprojection.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES='' bash scripts/python.sh scripts/audit_dh116_left_pinch_preprojection.py --output <fresh-private-output.json>
```

## 证据与晋级

- 原native receipt SHA：`25a75775ec43e03f6c722ae395df8584ac149e38c87e18ac77aba52f6d8d745d`
- 重算JSON SHA：`9bd10e284f623d2af1526eebf0f88e4772a7be0d8e51e8e4659f64f99990f610`
- raw SHA及来源路径由只读脚本锁定，原始文件不上传GitHub。
- 新物理步0，PPO0，WireSeg训练0。原native176步/exit1仍失败，不因本CPU包通过而晋级。
- 下一包：独立“原投影控制”合同、跨步连续性与完整owner回归；冻结后才形成新原生配方。
  不重跑旧配方，单实例，0PPO，≤240步/180秒。
- Pinch尚无双侧有效承托；Tripod、Power的闭合几何阻断保持各自状态；Teacher准入仍blocked。
  528保护文件、历史Checkpoint与30mm/5s/0.5mm标准不变，PR#2 Draft。
