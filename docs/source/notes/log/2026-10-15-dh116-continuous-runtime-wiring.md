# DH116 连续抓取运行时接线

阶段D4，父节点 [DH116抓握](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty，Inline/core0。按 [held-transport Task6](../../docs/superpowers/plans/2026-10-15-dh116-held-transport.md) 实施，不热更pilot8，不提交/推送。

## 运行时合同变化

`verify_fixed_dual_panda_dh116.py`新增显式`--grasp-sequence`，与独立`--closure-pilot`互斥。启动前强制要求SHA固定prepose/approach、点参考、contact manifest、重力、桌面wire、反力、协调控制、100µs以内步长、四路显式载荷和足够运行时间。

连续模式从.318m自由wire和预位姿开始；原pilot保持.308m切触诊断。`grasp_mode`决定测量/保护可用性，`closure_active`仅在实际阶段close/hold/lift允许目标指腹接触；此前全部pad仍受无接触源保护和prepublish门限制。总驱动预算、速度、全臂/非目标碰撞、指腹前区/穿透和反力门保留。

每步真实PhysX读回点/法向对比张指接近目标；实际base_link内固定材料锚点、实际rod节点高度和当前rod半径计算两站抬升/最低线表面。ID只在开始从固定目标区域选取，不随接触漂移重分配。阶段观察器消费这些实测值，不使用命令位姿代替证据。

连续闭指/保持/抬升使用`coordinated_bimanual_increment`一次共同限速，在任一手目标发布前计算两侧增量。只有实测保持通过才给共同向上速度；任一证据或保护失败停止。结束必须finish和全部物理/点J门，不能靠步数到期成功。独立pilot始终否认抓取资格；最终探针失败时所有摘要抓取成功字段强制False。接近通过记录保留在阶段切换时，不用抬升后的位姿重新比较旧接近目标。

## 验证

- 新合同先RED：缺少validator；增加后专项发现2个旧AST只允许pilot的断言，更新为显式互斥grasp_mode合同。
- 最终探针失败清除嵌套成功字段：RED 1失败/38通过，补齐失败收尾。
- 完整DH116/moving-contact：**420 passed in50.40s**，`/tmp/dh116-sequence-runtime-full.log`，单核/数值库单线程。
- 真实CLI用`--grasp-sequence --closure-pilot`拒绝，退出1且`/tmp/dh116-invalid-sequence-must-not-create`不存在；未启动第二个Isaac。
- Inline自查：实际测量在sim.step/arm.update后，命令共同缩放在下发前，无wire绑定，无阈值放宽；这不是独立代理审查。

## 未验证

新连续模式尚未实际运行，420tests主要是纯核心和接线合同，不证明动态接近/保持/抬升。pilot8仍用启动前快照（无新模式），PID75598/session59075；最近真实流式记录3236步四路有效、载荷.374/.333/.280/.349N，最长四路.2108s、最大滑移.245643mm，未保持。

下一步：跟进pilot8稳定性与原保护结果，再启动新的连续GPU试验，独立保留证据、源码和pin。新approach/prepose见[准备记录](2026-10-15-dh116-new-prepose-approach.md)。完整成功及3重复仍OPEN。
