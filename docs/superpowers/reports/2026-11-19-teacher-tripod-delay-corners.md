# Tripod 双手混合延迟角点验证

Antigravity 提出了枚举四路延迟角点、先筛混合组合、缓存相同几何状态的最小改动。复核 `recorded_lags` 和 `runtime_reference` 后，新增可选的 `--screen-delay-corners`。沿原 160 个 SDK 参考步，在腕 Z 的 0/6 tick 和三路手指的 0/11、0/11、0/16 tick 之间组成 16 个角点。各角点独立读 SDK 参考姿态，直到轨迹和延迟尾部均结束。复用原全部 22 凸包和阈值。先测混合组合，共用相同 pose/q 的几何结果。旧双对照模式继续可用。

加入了可证伪的回归：单拇指延迟使某一步杆净空下界为负，该反例在旧的两个组合中均未触发，扩大到 16 组合后才能定位。

| 侧别 | 完整角点 | 计入的整数 tick 检查 | 最小采样净空余量 | 限制形状 |
| --- | ---: | ---: | ---: | --- |
| 左 | 16/16 | 2776 | 7.306 µm | 桌面 `source:finger22_link` |
| 右 | 16/16 | 2776 | 11.041 µm | 杆 `source:finger13_link` |

两侧最小样本出现在参考 tick 160。右侧借用了左侧已记录的延迟范围。通过的离散角点包含理想轨迹，同候选另有理想情况下的连续路径证书。混合延迟盒内部、不同 Reset、同手自碰撞以及有效对向接触反力仍待认证；保持 `continuous_delay_box_proved=false`。Isaac 新步数 0，PPO 更新 0。

```bash
python -B -m scripts.audit_dh116_tripod_bilateral_closure \
    --endpoint "$PAIR_ENDPOINT" \
    --endpoint-sha256 16fcae86fcdf1a10f914d99d2cbc487dca1a44b359cf7584259cf7adb4a6f146 \
    --recenter configs/dh116_execution/tripod-bilateral-recenter-v1.json \
    --recenter-sha256 98e68553189cf7c34b22ee0d68e184ccdb9e1ab27251c8f9027cbe75ccd76000 \
    --screen-delay-corners --output "$AUDIT"
```

43 项相关 CPU 回归通过，分析产物 SHA `d9c90067555f321312225d51ef1d15196d804f1a1e398ef9a1f81c1382c63fda`。下一项用同一参考端点验证亚步延迟与被动关节残差，准备双侧逐指负载的原生对照。
