# Pinch 非零动作：坐标、Mimic 与 SDK 跟踪误差分离
日期2026-11-13；基线 `0779cf8d300faccce9ae7de1ca9f7cd7b98bd41a`。

## 结论与适用范围
已有16个实际480Hz端点（8零动作+8非零动作）显示，实际命名关节重建的原USD FK与真实link pose非常接近；目标跟踪滞后明显更大。但目标全部到位的指腹SDF仍有正间隙，**本区间的无接触不是单纯FK坐标映射错误或Mimic偏差，也不能仅用目标滞后解释**。
这是同一轨迹的CPU对照，不是新的原生实验、闭环实验或全局根因证明。没有改变动作、初态、控制器、资产、参数或关节保护。未拟合任何经验关节系数。

## 实际代码
新增 `scripts/audit_dh116_pinch_dynamic_fk.py`：
- `full_named_poses` 复用原 `HandPointFK.edges` 的左右独立USD joint frames，输入全部11个实测关节，保留实际被动角；不展开替换、不夹紧、不重新建模。
- 同一实际base_link pose、同一杆节点、同一半径，比较 actual_full11 / actual_active_mimic / sdk_active_target_mimic。后两者使用原coupling。逐帧核对自有遍历与原 `HandPointFK.poses` 完全一致。
- 第三个模型只替换六个手指active目标，仍以实测base pose为根，专门隔离手指跟踪差异；不是假设整个手腕到达目标。
- `evaluate_exchange` 核验原实测几何包、原半径来源、动作所在policy、实际SDK已接收的target与submission；保存完整17DOF q/qd/targets及各关节SI单位。腕部平移是m，其余是rad，不将17维都标为rad。
- 保存每个实际env/episode/reset/policy/physics/rod tick/timestamp、17个source link位置和旋转误差、四个必要pad的实际与模型SDF、实际Mimic残差和真实pair数量。
- `run` 只接受已固定SHA的完整18总步历史receipt和16测量行。新物理步/optimizer均0。CLI输出独占、fsync，不覆盖旧证据，不导入Isaac/ropesim/warp。

## 实际对照
每手每模型272个link端点（17原source link ×16物理端点）。下表为位置误差，单位µm；它不是物体Lift，也不是线缆接触深度或硬件测量精度。

| 手 | 模型 | P50 | P90 | P95 | 最大 | 最大对应link/physics step |
|---|---|---:|---:|---:|---:|---|
| 左 | 实际全部11关节 | 0.027859 | 0.045233 | 0.068549 | 0.068549 | finger43_link /3 |
| 左 | 实际active+Mimic | 0.027859 | 0.045233 | 0.068549 | 0.068549 | finger43_link /3 |
| 左 | SDK active target+Mimic | 0.031836 | 3.673819 | 8.834761 | 13.428401 | finger14_link /18 |
| 右 | 实际全部11关节 | 0.024455 | 0.046098 | 0.047702 | 0.077378 | finger13_link /14 |
| 右 | 实际active+Mimic | 0.024455 | 0.046098 | 0.047702 | 0.077378 | finger13_link /14 |
| 右 | SDK active target+Mimic | 0.027834 | 2.186769 | 5.798625 | 11.588912 | finger22_link /18 |

实际11关节模型最大旋转误差左1.120538e-6rad、右9.367184e-7rad。实际Mimic残差最大左7.638645e-8rad、右8.313370e-8rad。这里保留FP32原生读回误差，不宣称未来运动有同一误差上界。所有峰值在JSON中有完整身份，左右峰值可能属于不同link/时刻，不能拼成一条轨迹。

SDK18同一端点，指腹 sampled-node SDF间隙（mm）：

| 手/指腹 | 实际位姿 | 实际11关节FK | SDK目标FK |
|---|---:|---:|---:|
| 左拇指 | 1.603887 | 1.603848 | 1.584885 |
| 左食指 | 2.040942 | 2.040891 | 2.014487 |
| 右拇指 | 1.861802 | 1.861795 | 1.851297 |
| 右食指 | 2.439360 | 2.439323 | 2.410284 |

四个SDK目标的几何间隙仍为正。原生hand–rod pair仍0。SDF不是精确连续距离、不是实际法向载荷；不由几何预测授予接触资格。
该区间动作只有1/2/7/8索引为0.1/0.1/0.05/0.1，无手腕指令。下一步需重新设计可达初态与闭合路径，并独立验证，不能直接扩大右手动作或放松桌面/关节保护。

## 测试、失败及证据
新增18项CPU测试；先保存缺模块失败 `evidence/pinch-dynamic-fk-red-0r7efa/pytest.log`。
最终相关回归 **83 passed，0 failed，53.61s**。CLI自然exit0，无native imports。
覆盖完整被动角、原FK等价、刚体变换不变性、原左右源、18/16步身份、已提交动作/target核验、七类篡改拒绝、峰值身份及拒绝覆盖。

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_pinch_dynamic_fk.py tests/test_dh116_r0_measured_approach.py tests/test_dh116_native_parameter_r0_replay.py tests/test_dh116_measured_closure_geometry.py tests/test_dh116_dual_pinch_closure_readback.py
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/audit_dh116_pinch_dynamic_fk.py --output /new/output/comparison.json
```

日志/JSON：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/pinch-dynamic-fk-regression-GbTDji/`。
comparison SHA `cb87da4ba3da91d82aff235861d1e40671028ae116c949f2a74bddf6faa84267`。
父receipt SHA `f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901`。
原physical.jsonl SHA `1981d68067f6554e63a27b190b9eb55f8663501ebdc9daca4700c73929e4ba2a`。
公开报告不携带原始数组/模型。代码未修改任何原controller、FK、USD、物理或动作合同。

Teacher仍0/3；新增完整episode0、成功0、PPO0。保持R2关闭，PR #2 Draft。继续实际接触可达性、稳定性测量来源、完整R0/Reset和剩余特权状态资格；相机/数据缺口不阻断无相机准备。
