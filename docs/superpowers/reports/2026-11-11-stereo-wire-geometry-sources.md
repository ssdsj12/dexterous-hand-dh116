# 双目线缆点源与时序身份复放

基线为 `98ba1fb`。新的 `WirePointGeometry` 接收原有 `TrackedWirePointExtractor` 的实测输出，保存可校验的逐点几何数据。每槽位记录世界坐标 XYZ、来源、左图可见性、分量、米制弧长、深度有效性、置信度及轨迹 ID。像素弧长、图像轨迹 ID、深度分段 ID 和采样像素另存，单位和作用域各自明确。

| 来源编码 | 物理含义 | 弧长和身份 |
|---|---|---|
| `centerline=1` | 左 RGB 可见骨架样本，深度可以缺失 | 2D 弧长和图像分量 ID 可保留。连续实测 3D 段有局部米制弧长，深度洞处为 NA。只有合格的完整分量描述子可以得到 3D 分量轨迹假设。 |
| `observed_surface=2` | 同一已观测 RGB 分量附近的独立有效深度像素 | 米制/像素弧长为 NA，`track_id=-1`。可以报告所依附图像分量的轨迹重叠，不能赋予表面点材料身份。 |
| `unknown=0` | 新提取器中的空槽；历史 Actor 输入保留既有未标来源语义 | 新几何记录无 XYZ、轨迹、弧长或深度有效性。 |

2D 骨架在深度查找前形成。RGB 断开的可见片段分别标记。深度洞内的左图骨架保持可见，XYZ 为 NaN，深度掩码为 false。每段的米制弧长从零重新计算，没有补点或跨洞连线。源组件时间匹配无法给出材料点速度、真实曲率或 Cosserat 节点标签。

`sensor_fields()` 只使用当前 `SensorBatch` 已有的 `points_world`、`point_valid`、`point_confidence` 和 `point_provenance`。现有掌坐标估计器的 opt-in 来源编码可直接消费这些字段。额外几何元数据用于证据与监督准入，保留历史 Actor 输入和训练/部署模型形状。原跟踪器、观测类型和估计器源码是旧处理或部署合同的 SHA 输入，此工作包保持其字节不变。

`NativeStereoEnvironment.snapshot()` 在使用已验证的跟踪提取器并有参考帧时，把逐点 sidecar 复制进 `extra_labels.state_wire_geometry`，随候选保存。既有采样像素和轨迹字段仍可读取。引用数据和逐点身份均未加入 Actor 张量。该采集路径通过 CPU fixture 验证，新的原生采集尚待批准。

## 已执行的 CPU 复放

用服务器现有解释器复放历史 Front 与 Roll20 各四帧、每帧两环境。读取已完成的原始采集 proof、参考 SHA 及上一轮逐帧对应归档。独立回读检查左右 RGB、帧身份、原生标定、每帧归档 SHA。用归档正视差及原标定反算有效 Z 并重新跟踪。每方法逐帧验证当前采样像素和原归档逐点相等。光轴深度参考只参与评分。

下表为首帧，深度误差为选中且匹配原生左 wire ID 的光轴 Z，单位 mm。固定分母仍为同抓法原生左 wire 区域中的有效参考像素。表中各来源分别报告其对这个分母的贡献。暂无独立真实 3D 中心线分母。三维正确点沿用同像素欧氏误差 ≤5 mm。

| 数据/抓法 | 方法 | 中心线有效/可见槽 | 中心线正确/固定分母 | 中心线 MAE/RMSE/P95 | 表面有效槽 | 表面正确/固定分母 | 表面 MAE/RMSE/P95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Front 二指 | 原色轮廓 | 2/32 | 0/2128 | 50.853/53.929/67.010 | 13 | 6/2128 | 21.007/32.148/65.257 |
| Front 三指 | 原色轮廓 | 0/32 | 0/2165 | NA | 14 | 0/2165 | 17.878/19.604/37.256 |
| Front 二指 | 全窗口软引导 | 4/32 | 0/2128 | 46.914/48.944/65.827 | 96 | 9/2128 | 49.677/72.615/132.704 |
| Front 三指 | 全窗口软引导 | 3/32 | 0/2165 | 41.294/41.554/46.699 | 60 | 5/2165 | 31.721/55.852/79.948 |
| Roll20 二指 | 原色轮廓 | 0/32 | 0/2009 | NA | 96 | 91/2009 | 2.708/4.949/6.057 |
| Roll20 三指 | 原色轮廓 | 0/32 | 0/2054 | NA | 96 | 89/2054 | 3.286/5.869/20.602 |
| Roll20 二指 | 全窗口软引导 | 4/32 | 0/2009 | 66.837/75.954/116.656 | 96 | 67/2009 | 7.231/13.786/22.273 |
| Roll20 三指 | 全窗口软引导 | 4/32 | 0/2054 | 41.259/43.405/59.499 | 96 | 65/2054 | 7.326/16.377/20.901 |

四帧三次相邻转移的图像分量轨迹重叠数量保持一致：

| 数据 | 方法 | 中心线图像分量重叠，二指/三指 | 表面关联图像分量重叠，二指/三指 | 3D 中心线分量重叠 |
|---|---|---:|---:|---:|
| Front | 原色轮廓 | 6/4 | 2/2 | 0 |
| Front | 全窗口软引导 | 16/5 | 3/2 | 0 |
| Roll20 | 原色轮廓 | 6/4 | 2/2 | 0 |
| Roll20 | 全窗口软引导 | 13/5 | 3/3 | 0 |

所有方法的首帧及其后观测都没有可用的 3D 中心线分量轨迹。图像重叠是传感器关联诊断，缺少真实逐点身份且样本运动有限，不能据此开放 Velocity 或 Deformation。表面材料身份保持 NA，unknown 没有可评分的实测槽位。每个有测量的 Roll20 正确点都属于 `observed_surface`，不能按中心线或材料节点计数。

命令：

```text
python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_wire_geometry.py tests/test_dh116_stereo_wire_depth.py tests/test_dh116_stereo_runtime_wire_tracking.py tests/test_dh116_stereo_perception.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_runtime_actor_modes.py tests/test_dh116_stereo_runtime_task_state.py
python scripts/audit_dh116_wire_geometry.py --proof <native-completion-proof> --correspondence-summary <prior-replay-summary> --correspondence-sha256 <prior-summary-sha256> --output <new-private-directory>
```

96 项 CPU 测试通过，5.08 秒，含新增候选旁路测试。两次历史复放均正常退出。Front/Roll20 新复放摘要 SHA256 分别为 `cc61184f8e051494e8cc7f6077376aa039eb6a507b6d9bdc3274edc160ef5f3c` 和 `07b5c2b0d385ac81be82daa9e5ec01b2aa26121c8c394c69ea0dfad489f8f085`。逐帧几何和日志保存在独立证据目录。本轮无新原生采集、Isaac 或模型训练。下一步是在获批同步采集后复核传感器轨迹与同身份连续帧，并保持材料节点及力学来源单独资格。
