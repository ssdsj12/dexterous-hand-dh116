# Power 掌面接近和五指包握分阶段检查

Antigravity 核对了两个现有入口。腕部候选的初筛复用了旧闭合关节，且后退范围只有 0 和 4 mm。当前新增两个互斥的 CPU 审计模式。第一种在固定的原腕位和一个独立腕位上，用原主动限位重解闭合，增加单独的拇指外展采样。第二种固定材料工位和腕朝向，使用完全张开的原主动关节下界，沿原掌心向内方向扫描 0–24 mm。先用掌面几何定位 0.5 mm 内的候选，再检查全部 22 个凸包、原桌面保护线、支架、杆、拇指与其余手指净空。候选通过后才尝试五指闭合。

- 左手固定腕位重解后，拇指间隙由 15.477 降到 6.631 mm，无名指仍为 17.253 mm。向内接近的五个近掌候选均碰到同一支架（索引 1），由 source `finger13_link`、source 和 cooked `finger14_link` 触发。
- 右手材料腕位让食指间隙从 16.972 降到 13.400 mm，无名指从 5.724 降到 1.768 mm，掌面间隙增加到 15.390 mm。独立张手向内对照找到 5 个几何合格的近掌终点。最佳近平面候选向内运动 14.750 mm，掌面间隙上界 0.140 mm。随后尝试闭合，右中指最小候选间隙上界仍有 35.833 mm，无法同时满足握式要求。
- 上述定位支持分侧求解。左侧优先验证避开支架的材料工位。右侧优先在已能接近掌面的腕位改变局部俯仰，核查五指是否能围合。两路仍需连续安全路径、locked cooked 掌面以及双侧接触反力资格。

```bash
python -B -m scripts.audit_dh116_power_material_reach \
    --closure-evidence "$POWER_CLOSURE" \
    --closure-sha256 80d9fa11b6ea28b3c86f61e18b7b276071d8845c1511fab2ab4509267b158fb1 \
    --resolve-closure --output "$WRIST_AUDIT"
python -B -m scripts.audit_dh116_power_material_reach \
    --closure-evidence "$POWER_CLOSURE" \
    --closure-sha256 80d9fa11b6ea28b3c86f61e18b7b276071d8845c1511fab2ab4509267b158fb1 \
    --palm-approach-only --output "$PALM_AUDIT"
```

16 项 CPU 回归通过。腕位闭合证据 SHA `592370deec72f00323f739691da60d411b92a0e756294d0cb7ed8ace8449cb43`，单轴掌面证据 SHA `4798f7bb8655d9b5865f802ced9008f66215df53e4db2152d9421bf6e5bfb749`。Isaac 新物理步 0，PPO 更新 0。OakInk 的物体局部姿态表述用于设计对照。官方 Shape 归档已定位，下载链路未返回可验 SHA 的样本，此轮训练使用的 OakInk 样本数为 0。 [OakInk 数据格式](https://github.com/oakink/OakInk/blob/main/docs/datasets.md) [固定归档](https://huggingface.co/datasets/oakink/OakInk-v1/blob/3eadff1d4a144aab44683ee24b4c81cee7ecc5b1/oakink_shape_v2.zip)
