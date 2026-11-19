# Pinch 初始净空与接触位置复核

基线为 `717c52965b883f3151335a1b9f865ac0917d46f6`。上一轮 Pinch PPO smoke 在 SDK 12 读到 `left:source:finger22_link` 净空 2.097674 mm，低于 2.100 mm。该轮有一个完整控制步和一次部分步进，未执行 PPO 更新。

## 同源 CPU 证据

| 候选 | 左连续净空 mm | 左最大指腹终点误差 mm | 结论 |
| --- | ---: | ---: | --- |
| 左侧增加 0.050 mm 高度余量 | 2.151095 | 0.071819 | 终点误差大于 0.050 mm |
| 同高度加左腕世界 X 0.055 mm 和拇指 seed 0.001 rad | 2.151095 | 0.023015 | 静态余量、指腹终点和原连续路径检查通过 |

右侧独立原始连续净空 2.112018 mm，双手完整扫掠 Y 轴间隔 295.679175 mm。固定原自由杆及关节耦合，仅重新测量独立 Reset 候选。无额外余量时，输出逐字节复现旧配对证据 SHA `996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37`。

左腕仅升高的拒绝记录 SHA `d104bdc39b3750e7980df5774b397d81769a44db4e261a43d10df1ed4aeb7a46`，进程退出 3 且没有 Reset seed。25 个固定高度的关节微调都未通过指腹终点检查。增加 X 重定位后入选候选的记录 SHA `eb60404a0f9a889edfd377e1ef59d989e0ca1498ec45e7e636810678a9cad063`，Reset seed SHA `8cb5a370228e62b699675af28babeba45126cc3a395086b287af6bc9dec00a50`，原生准入为 false。

```bash
python -B scripts/prepare_dh116_paired_pinch_reset.py --right-result "$RIGHT_CPU_EVIDENCE" \
  --left-extra-clearance-m 0.00005 --left-root-x-m 0.000055 --left-thumb-seed-rad 0.001 \
  --output "$NEW_CPU_PROPOSAL"
```

`build_fresh_pinch_owner` 将合同中的完整 Reset seed 传给双手 builder。builder 在读取冻结 pair 后、创建设施或推进 SDK 前校验完全一致。旧正式记录与旧 pair 的 seed 源相同，未来替换须由独立自然退出的原生 Reset 回执重新绑定。

## 验证

- Ubuntu CPU 候选回归 5 项通过。
- builder、Teacher session 和实际 provider CPU 联合回归 62 项通过，日志 SHA `d967ab68906dd7e3ec42ebc5c4a70d33646e3cf48650dd196345f3283c8c9ec0`。错误的 seed 注入会在初始化前被拒绝。
- 本包有 0 个 Isaac 原生运行、0 次 PPO 更新。下一项是把独立新 pair 与原生 readback、numeric Reset、外层自然退出回执绑定，然后依原几何、反作用力和 30 mm / 5 s / 0.5 mm 判据验证。
