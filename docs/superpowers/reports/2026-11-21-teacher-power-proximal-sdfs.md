# Power 原始小指近端 SDF 诊断

从两侧相互独立的 PhysX `finger52_link` 原始碰撞体生成 DeformX SDF。输入绑定已发布的同钟 Tensor View 射线回执和各手原始 Cooked Hull，保留闭合三角扇。左右各有 36 条来自指定原刚体和碰撞体的射线。两种分辨率使用同一物理位置比较。

| 网格间距 | 左表面 P95 / 最大绝对误差 | 右表面 P95 / 最大绝对误差 | 距表面 25 / 100 / 500 µm 的符号错误，每侧各 128 样本 |
| --- | --- | --- | --- |
| 1.0 mm | 0.2451 / 0.4220 mm | 0.2452 / 0.4219 mm | 两侧均 6 / 3 / 0 |
| 0.5 mm | 0.1222 / 0.1880 mm | 0.1226 / 0.1887 mm | 两侧均 0 / 0 / 0 |

表面统计各使用 194 个采样点。0.5 mm 网格为 78×86×139，后续接入使用此候选。误差是已测采样误差，后续路径须再验证真实间隙及入接触时的反力。

两次 RTX 4090 独立烘焙及 GPU 网格注册见证分别约 6.03、6.24 秒，外层自然退出码均 0，未推进 Isaac 或杆求解器。此处的 2 个网格仅注册到独立形状检查用的杆。Power 原生任务注册、接触、抓取和 Teacher 准入均待验证。未修改原物理配置。

## 复核

`python -m scripts.bake_dh116_power_proximal_sdfs --capture CAPTURE --ropesim-package-root BACKEND --output OUTPUT --spacing-m .0005`

`python -m pytest -q tests/test_dh116_deformx_pad_sdf.py tests/test_dh116_power_left_palm_sdf.py tests/test_dh116_power_right_palm_sdf.py tests/test_dh116_power_palm_raycast.py tests/test_dh116_work_packages.py`

70 项 CPU 回归通过。原始捕获、GPU 回执、SDF SHA 和原始部件映射见对应 readiness。下一步在原生 Power 任务中将两侧近端碰撞体分别接入，保留原掌面和远端接触门槛。
