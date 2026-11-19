# Tripod 连续跟踪延迟的条件证书

Antigravity 建议按凸包只有一路主动关节变化的性质来降低域的维数。追问后核对了延迟方向。腕相位相对手指相位的偏移属于 `[-D_wrist, D_finger]`，手指已知相位下的最晚腕相位由手指延迟上界决定。每个活动链使用相位与相对偏移的矩形域，腕相位裁剪在该轨迹的有效区间。全部点对应连续纯延迟。仅腕运动影响的凸包用一维域。

使用当前控制器的 161 个实际参考结点逐段线性插值。通过真实主动及被动耦合链计算各凸包单步位移上界，按位移半径扣减桌面、杆和双手分离净空。每组证书独立复算全部子矩形，检查覆盖和互斥。原 2.100 mm 桌面保护线保留。

| 侧别 | 独立复查覆盖箱数 | 证书最小余量 |
| --- | ---: | ---: |
| 左 | 81 | 0.118 µm |
| 右 | 86 | 0.051 µm |

连续包络覆盖固定 41 节点中心线、理想耦合与左右共用的历史延迟边界。右侧仍需独立实测延迟，证书最小余量很小。原生追踪偏差、被动联动残差、同手自碰撞和接触反力仍待验。

```bash
python -B -m scripts.audit_dh116_tripod_bilateral_closure \
  --endpoint "$PAIR_ENDPOINT" --endpoint-sha256 16fcae86fcdf1a10f914d99d2cbc487dca1a44b359cf7584259cf7adb4a6f146 \
  --recenter configs/dh116_execution/tripod-bilateral-recenter-v1.json \
  --recenter-sha256 98e68553189cf7c34b22ee0d68e184ccdb9e1ab27251c8f9027cbe75ccd76000 \
  --certify-delay-band --output "$AUDIT"
```

44 项 CPU 回归通过。完整产物 SHA `fa6910beac936c05d809b878ff4763d7f47e64736ae592df7d302ff43de9814c`。Isaac 新物理步和 PPO 更新均为 0。
