# Power 物体局部横截面的腕位可达性

按 OakInk 的物体局部抓取坐标思路，在同一根双端支撑杆的左 33 / 右 8 材料工位，保持已测得的五指闭合姿态，分别改变绕杆滚转、横截面第二轴偏移和掌面后退量。每侧有 5×5×2 共 50 个候选；先复算全手对线缆的保守下界，再对短名单复算桌面、端支架、拇指与其他四指的分离和原凸包近点。桌面最小净空仍为 2.1 mm。 [OakInk 数据说明](https://github.com/oakink/OakInk/blob/main/docs/datasets.md)

| 侧别 | 原最大精确间隙上界 mm | 新最佳 mm | 观察 |
| --- | ---: | ---: | --- |
| 左 | 17.253457 | 17.253457 | 两个较好候选的向下横移均被桌面修正抵消；无名指仍限制上界 |
| 右 | 16.972087 | 15.390277 | 掌面后退 4 mm 得到约 1.582 mm 改善；多种滚转给出同一量级 |

另做锁定原腕角、只试拇指外展与横移的同杆对照。原腕位的拇指外展调整未降低两手的最大间隙，多个取值触发了保守拇指自碰筛查。这个有限对照只测试已选手型，接下来应比较不同滚转及抓取工位与指节重求解的组合。旧单一开指姿态排序无法回答闭合后掌面和五指是否同时够到线缆。两侧被测最佳仍高于 0.5 mm。实际掌面需锁定的 cooked 碰撞来源及真实承载反力，当前仍未资格化。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -m scripts.audit_dh116_power_material_reach \
  --closure-evidence "$PRIOR_CLOSURE_AUDIT" \
  --closure-sha256 80d9fa11b6ea28b3c86f61e18b7b276071d8845c1511fab2ab4509267b158fb1 \
  --output "$NEW_POWER_REACH_AUDIT"
```

实际结果 SHA `c874fe70ba18db98b0272dbf83abf25ddd7ee998acef6997f727c35fa483b48f`。14 项 CPU 回归通过，0 Isaac 实际步，0 PPO 更新。
