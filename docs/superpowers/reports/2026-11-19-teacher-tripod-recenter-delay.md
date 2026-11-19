# Tripod 双手姿态微调与历史延迟对照

参考端点绑定原双手源几何和同一根 41 节点杆。新增独立双侧候选配置，腕位改变量上限 1 mm、腕旋转 0.04 rad、主动关节 0.02 rad。使用原 24D 编译器、原投影和净空规则，在 20 个控制步与 160 个物理参考步上重算连续几何证书。右拇指近节仍是主要限制。

Antigravity 建议在理想跟踪证书之外，分别将实测左 Tripod 的腕部和手指滞后放到各自的目标序列。使用冻结回执与 `physical.jsonl` 得到的上界为腕 Z 6 tick，拇指屈曲、食指、中指分别 11、11、16 tick。右侧借用该组滞后做压力对照。第一版在仅手指滞后的左侧参考 tick 162 触发 `source:finger22_link`，低于原 2.100 mm 保护线 0.914 µm。保留失败证据。第二版在该候选上只抬左腕 15 µm，重新运行同一检查。

| 侧别 | 连续桌面余量下界 µm | 连续杆余量下界 µm | 终点最大指腹间隙 mm | 两种延迟对照采样数 |
| --- | ---: | ---: | ---: | --- |
| 左 | 2.64 | 13.84 | 0.484692 | 167 / 177，均未触发 |
| 右 | 495.63 | 4.33 | 0.489087 | 167 / 177，均未触发 |

两侧指腹方向点积均通过原判据。右杆下界相对原双侧参考从 0.145 µm 增至 4.33 µm。历史延迟对照覆盖两个离散组合，右侧没有独立实测跟踪延迟。双手连续的混合滞后、自碰撞、实际反力、离架和有效 Hold 仍待验证。

```bash
python -B -m scripts.audit_dh116_tripod_bilateral_closure \
    --endpoint "$PAIR_ENDPOINT" \
    --endpoint-sha256 16fcae86fcdf1a10f914d99d2cbc487dca1a44b359cf7584259cf7adb4a6f146 \
    --recenter configs/dh116_execution/tripod-bilateral-recenter-v1.json \
    --recenter-sha256 98e68553189cf7c34b22ee0d68e184ccdb9e1ab27251c8f9027cbe75ccd76000 \
    --screen-recorded-delays --output "$NEW_AUDIT"
```

42 项 CPU 回归通过。第一版反例结果 SHA `959df5557e1605335be80652610578ed2b1386195010102778aae306a2be816f`，候选最终证据 SHA `303223056094d6b5f7c59ff4d281fc506f242d929d577c9c84458e12c726be93`。Isaac 新物理步 0，PPO 更新 0。
