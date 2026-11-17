# 握式 Teacher 源几何预检

握式按整手包握进行本次几何筛选，暂用拟议名称 `power_wrap`。每手要求拇指与另外四个独立远端指腹均在原 CAD 近接触和法向范围内，拇指与至少三个合格指腹相对。双手原碰撞净空及现有两端支撑场景也须通过。将来如获得合格初态，包握身份需在原生轨迹上单独由实测指腹接触验证，沿用原 371D、24D、30 mm 抬升、5 s 保持、0.5 mm 最大滑移及外部支撑条件。握式几何筛选阈值独立于这些成功条件。

当前 Teacher 的 `STYLES`、`GRASP_STYLES`、重置与配对批次仅注册二指 `pinch_thumb_index` 和三指 `tripod_thumb_index_middle`。`--full-hand-contact` 打开现有 17 个链接的接触读取，包含五指和掌心分组。原生成功跟踪器检查双手每手对向有效指腹、抬升及固定锚稳定性。全手读取和原生成功布尔值本身无法识别包握形态。

## 已归档候选

使用本轮原 T5 采集的实际 `tensor_contract.json`，SHA256 `eda0b7e316a493997192ffec9abd8b5648c826a265d817774dd9117c734ec551`。现有 `solid_cylinder` 长 0.5 m、直径 16 mm，轴为 Y，双端支撑抬高 60 mm，中心 Z 为 0.368 m。通过候选中每手的实际源几何距离、法向及对向点积重新计算，保留五指各自的结果。

| 来源与 ID | 左手合格指腹 | 右手合格指腹 | 几何场景 |
|---|---|---|---|
| effective pad triad 158、190、149 | 拇、食、中 | 拇、食、中 | 与 T5 相同杆中心和双端支撑 |
| effective pad triad 153 | 拇、食、中 | 拇、无名、小 | 与 T5 相同杆中心和双端支撑 |
| 历史 multifinger 28、16、30 | 拇、食、中 | 拇、食、无名 | 旧杆中心 0.308 m |
| 历史 multifinger 17 | 拇、食、中 | 拇、食、中 | 旧杆中心 0.308 m |

0/8 符合双手包握。匹配 T5 场景的左手，最小无名指、小指几何间距分别约 14.372、32.170 mm。右手 153 的无名指、小指间距约 0.245、0.338 mm，食指和中指未通过。既有 center 组 4 对均只含拇指和食指。历史 pinch 和 grip_lift 候选只保存每手两个接触区的距离，无法验证五指。旧 multifinger 报告的 4×6 秒原生短测为 0 成功。

两份来源 SHA 分别为 `8ba12d5b8765a582c0f6d6d339ef9ab92b2aa8cb3e07531702632db3e3edc4f9`、`52fdd94cdcdcae136da73662c2ff9849d39dbbf7311334328486391653212492`。本次输出 SHA 分别为 `bc19fde5067597604163a2ff73aeed8fb3a81ed2736d0a612a439d5f09811193`、`234eb791df392d0b78f3902d86b156b2255d22b74cfbc0a39eb8c5ba74ef29a7`。

## 有界 CPU 搜索

在私有目录独立调用现有源 USD FK、实际指腹 CAD 区域采样、圆柱距离和端支撑碰撞函数。旧生成器的目标是最优任意三指，独立试验改用全五指距离及拇指相对方向目标。双手各从 4 个已有三指种子及其小扰动出发，共 8 个起点，48 次优化并以原密度手、杆、指腹采样做最后重算。保持 17 个碰撞体、原手、原关节限制、原圆柱和支撑，脚本不写原生成器或既有候选。17.25 秒完成，两侧均无通过单手筛选的候选。在本次无穿透样本里，最小的最大五指间距左手约 9.42 mm，右手约 4.38 mm。该预算只说明这些起点和搜索没有找到合格包握。

私有搜索脚本 SHA `4d73e9527cedff2aa5274c66417c5b6187a0711d9b604c8c0e4900108136f50b`，输出 SHA `95dac9d1b207f30aa226b95826ea19bc83fef99898413b266415b8f8b28ad1d7`。原始证据保存在 Ubuntu 私有目录。

## 复现与下一步

```bash
bash scripts/python.sh scripts/audit_dh116_teacher_wrap_geometry.py \
  --source "$archived_geometry" --source-sha256 "$archived_geometry_sha256" \
  --native-contract "$observed_t5_contract" --native-contract-sha256 "$t5_contract_sha256" \
  --output "$fresh_report"
python -m pytest -q -p no:cacheprovider tests/test_dh116_teacher_wrap_preflight.py \
  tests/test_surface_success.py tests/test_multifinger_monitor.py tests/test_dh116_stereo_trials.py
```

Ubuntu CPU 回归 234 通过、3 跳过，6 个新测试覆盖五指距离、拇指对向和畸形输入。核对 528 个保护文件与 2 个历史 Checkpoint。本次原生包握运行次数为 0，二指和三指路径沿用现有版本。下一步在同一圆柱和支撑场景继续寻找独立合格几何，复核完整初态和双手碰撞后，再为握式新增独立原生队列、实测身份验证和 PPO Teacher 报告。当前阻断为缺少双手五指合格初态，评估和训练尚未开放。
