# F2 DeformX standalone CUDA 数值验证

冻结上游后端 `96b8514c7c55c724f1abba1fbf6ed6642b7db72b`，扩展 SHA256 `c47942c4b15b3dfedc47212e71f8193825c0347df3c1d73fb76b648f7bdedb53`。Ubuntu 原 Python 3.10.12 和 Torch 2.7.0+cu128 运行。物理 GPU 3 读回 RTX 4090、计算能力 8.9。外部 `ldd` 记录显示扩展的 `libcudart.so.12` 解析到隔离 SDK。独立进程外层退出码 0。

## 原生 CUDA 数值测试

选取上游 API、静态、动态、平面接触、SDF mesh 接触、CUDA 零拷贝和逐点力读取用例。**36 项通过，1 项预期失败，0 项非预期失败**，用时 18.07 秒。已知高刚度管材 `EI=10 N·m²` 在该上游测试中最大应变达 632%，保留其严格 xfail。上游静态试例的两个悬臂挠度与理论比为 0.924 和 0.928。摆动周期 1.3832 s，对比离散链理论 1.3354 s。

独立运行在 21 个自由材料节点和静态 SDF 平台上读取每帧 World wrench。90 个 60 Hz 帧、每帧 100 个杆子步。此处验证求解器自身的力、速度和力矩输出。

| 指标 | 实测 |
|---|---|
| 线缆重力 | 1.14776997 N |
| 最后时刻 mesh 所受法向力 | −1.14717400 N |
| 最后时刻 rod 所受法向力 | +1.14717376 N |
| 最后时刻作用反作用最大分量残差 | 2.3842e−7 N |
| 平台原点处反作用力矩 Y | +0.57361054 N·m |
| 线缆平均垂向位移 | −13.65007 mm |
| 最终节点最大速率 | 6.1633e−5 m/s |
| 终态离散中心线最大曲率 | 0 m⁻¹，平面静置 |
| Z 动量收支残差 | 0.00102327 kg·m/s，约为重力冲量绝对值的 0.0594% |
| 独立数值记录墙钟 | 1.13834 s |

动量残差按实测节点质量和速度差减去重力冲量、再减去逐帧反向 mesh 接触冲量计算。此验证的运行范围为 standalone 求解器和静态 SDF mesh。真实 PhysX 刚体的运动反作用、双手同杆抓取和任何 Teacher 成功仍需原生证据。该试例的材料参数来自上游标定场景。

## 复现与记录

```bash
python=/mnt/mydisk/xk/envs/dh116/bin/python
export CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$build/python" LD_LIBRARY_PATH="$sdk/lib64"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
"$python" -m pytest -q -s -p no:cacheprovider \
  "$backend/tests/test_api.py" "$backend/tests/test_statics.py" "$backend/tests/test_dynamics.py" \
  "$backend/tests/test_contact.py::test_plane_rest_support_equals_weight" \
  "$backend/tests/test_contact.py::test_plane_incline_coulomb" \
  "$backend/tests/test_mesh_contact.py::test_box_floor_matches_plane" \
  "$backend/tests/test_mesh_motion.py::test_resting_rope_wrench_equals_weight" \
  "$backend/tests/test_batch.py::test_zero_copy_views" \
  "$backend/tests/test_mesh_broadphase.py::test_force_views_match_readouts"
```

私有证据目录 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-f2-20261112-Xli6B9` 保留命令日志、动态库映射、退出码和逐帧测量脚本。

| 文件 | SHA256 |
|---|---|
| standalone 测试日志 | `841bdaf74ad89a94f9a45fafda0af191e5d82c48d31c3d25e95b9a2680b1ec48` |
| 独立数值结果 | `85bd4b464c4ce27eff5453766adf520d6efabf53087f155c509b7b140dbfaf18` |
| 数值测量脚本 | `0bd03f4ae7b25077426e3e61097e90e74f76bf24deb08561c43abbe3a6c5ed1a` |
