# R07：非中心 USD 相机投影修复

实施位置：Ubuntu 工作树 `/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/repair-r07-camera-20261115`，分支 `codex/repair-r07-camera-20261115`，基线 `d1fa6630f76420177a93cda7c24de25d97bf20af`。改动 calibration、camera、新测试、既有 resolution 测试的错误静态 SHA pin 和本报告五个文件。

## 问题和修复

旧 author 将主点偏移的水平和垂直符号都写反，readback 又使用相反的错误逆式，因此同 K 回读能自洽，却不能验证实际投影。RED 中直接执行实际 `_NativeRGBPairReader.__init__`，将其创建的真实 USD prim 交给 `UsdGeom.Camera.GetCamera` 和 `Gf.Frustum.ComputeProjectionMatrix`：640×480、fx=fy=400、目标主点 (350,260) 的光轴实际投影为约 (290.000002,219.999999)。

新增纯函数 `usd_aperture_offsets` 与 `usd_aperture_intrinsics`，由实际 author 和 readback 使用：

- 水平 offset = (width/2 − cx) × focal/fx；垂直 offset = (cy − height/2) × focal/fy。
- 逆式 cx = width/2 − horizontal_offset × fx/focal；cy = height/2 + vertical_offset × fy/focal。
- K 必须是有限、无 skew、正 fx/fy、合法齐次最后行的 3×3 矩阵；尺寸必须是正整数；focal 和 aperture 必须为有限正值，offset 必须有限。focal、aperture、offset 使用共同 USD lens 单位。
- 保留 SDK focal getter 乘 10、focalLength=24、分辨率、物理米制位姿、optical↔USD 轴变换、环境 origin 和数字相机槽排序。

`StereoCalibration` 字段、序列化和 digest 未改。默认中心 offset 仍精确等于 0；旧默认 calibration digest 固定为 `0d96cdb45d30a4ed36f5d2f0a8d263a296d33122a3fda2f67536b8f0025f538f`。左右相机可在原有 `rectified=False` 规则下使用独立 K；未放宽 rectified rig 的共同 K 限制。

## 测试范围

新文件 `tests/test_stereo_usd_principal_point.py` 共 50 项。覆盖 224×96、320×240、640×360、640×480，左右独立 K、中心和非中心主点、从 640×480 正确缩放 K、非平凡光学旋转、两组环境 origin、逆序返回的相机槽、中心默认输出和 digest、非法 K/尺寸/focal/aperture/offset、错误 SDK focal 单位拒绝，以及独立子进程的无 Isaac 导入保护。

实际 author/readback 测试仅替代 Isaac 扩展管理、CameraView getter 和 annotator 边界；USD stage、USD Camera、Gf 相机和视图/投影矩阵均是真实 CPU 实现。测试投影多个 optical 点到世界，再用独立 Gf 投影比较目标 K，避免只验证两个转换函数互逆。

共同测试前缀：

~~~bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -B -m pytest -q
~~~

| 运行 | 实际结果 | exit | 日志 SHA256 |
| --- | --- | --- | --- |
| RED：新测试，修改生产代码前 | 44 failed，6 passed；其中四个真实 author 非中心投影失败，其余为缺少 helper | 1 | `f23985e361516243da869ea32787b7fbc88ec055182a20e32eb1ad121e164ab2` |
| GREEN：新测试 | 50 passed | 0 | `e4806254ab4ae0821ef1e05b2dd3c7073ea6429a3e64cc24d6e522cb1a0ecddc` |
| 批准套件：新测试 + geometry、resolution_benchmark、rig_experiments、runtime_contract | 135 passed，1 skipped | 0 | `9d222d7f882d232dcacaed466d6ebde986c35d1a215e399e952ce88ea2f274fe` |
| 扩展旧回归：批准的四个旧文件 + resolution、native_bootstrap、camera_cli、shutdown_validation | 157 passed，1 skipped，1 failed（下述基线缺陷） | 1 | `3067041dac048fda268d27f1ac7d1a4057974af4ccdea1c6d6820870ea327769` |
| 独立基线复现错误 pin 单项（修正前） | 1 failed | 1 | `7990e0221e1c2c293778fe7168bc926bce1d8f917601fb177a82e4ece756caed` |
| 最终完整套件：新测试 + 扩展八个旧文件，修正错误静态 pin 后 | 208 passed，1 skipped | 0 | `08ef7fc1128511a9f93d541780e675b1b5c006baeda5812e7f675d56dc5f9b2b` |

批准套件的 skip 为既有 OpenCV 缺失分支测试：当前环境已安装 cv2，该分支按既有设计跳过。以上套件有重叠，计数不能相加作为独立测试总数。

完整实际命令、exit 文件、日志及 SHA 清单位于 Git 外：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/r07-usd-camera-projection-20261115/`。
对应名称为 `red`、`green`、`approved-suite`、`regression`、`baseline-failure`、`final-suite`，各有 `.command`、`.exit`、`.log`；汇总为 `manifest.json`。

## 扩展回归发现的既有失败

`tests/test_dh116_stereo_resolution.py::test_new_profiles_keep_old224_config_and_cover_fx_baseline_near_range` 第 76 行对旧配置字节 SHA 的断言已在基线失配：

- 实际文件：`configs/dh116_training/stereo_camera_v1.json`。
- 修正前 test 期待 SHA256：`f5c53f5c9d3042ee0ce97453a3420dda3ecb7417469d7362be20604b0171d494`。
- 工作树与基线实际 SHA256 均为 `897e842d3133b0fe3e7040abe1255c3c213c6a146f95dd73d10f3389a9ad6ddc`。
- 原 config 最后改动提交 `4e964ff14b39eee7b2aef2d5cd51d61e0eb80d55`；test 与该 SHA 断言最后改动提交 `964dd33a00d9b47c2b3d7a48c3ddb67ccd946862`。

通过 `git archive d1fa6630f76420177a93cda7c24de25d97bf20af` 向证据目录 `baseline-d1fa` 导出完整 src/configs、原 Python launcher 和原 test，在该目录独立运行原单项仍产生完全相同的失败。修正 pin 前的工作树与导出 test 文件 SHA 也相同（`f29e0ed0b6db00f0aea2e8c78379e7a1b633209223a78fee02ebf9c9df0f2a0d`）。因此该项确认为基线缺陷，非 R07 新回归。原 RED 和独立基线失败证据完整保留。来源核对日志 `baseline-provenance.log` 的 SHA256 为 `5da9aeba723e2ff57cf95087d2f702c7eea6443a654b15ad159932ef59c0b673`。

进一步直接读取 config 创建提交、test 创建提交和本包基线的 Git blob 原始字节：三者均为 1994 bytes，SHA 均为 `897e842d3133b0fe3e7040abe1255c3c213c6a146f95dd73d10f3389a9ad6ddc`；`git log --follow` 仅有 config 创建提交。因此错误静态 pin 从 test 创建时就不匹配，配置本身没有变化。该独立来源核对日志 `config-origin.log` SHA256 为 `1afe54f803811e439aaa88e1dcc45396765dde2115b30ef776cf2ccdc0cb8f79`。

经 root 扩展本包范围授权，仅将既有 resolution test 的 `LEGACY_224_SHA` 静态常量修正为原创建字节的真实 SHA，并注释原创建提交。保留实际断言，没有动态自算期望值或跳过；原 config 不改。修正后含新 50 项和扩展八个旧文件的完整套件为 208 passed、1 skipped，exit 0，无剩余失败。

## 保护和资格限制

修改前后分别调用真实 `scripts.run_dh116_work_packages.Repository(...).verify_protection()`；528 个受保护源码/脚本/配置与两个历史 Checkpoint 均保持 SHA 不变。中间后检 `protection-after.log` SHA256 为 `eebeb227ea317d3b615aa1df449813e9027b199ec339d73451be27a853b3b6c0`；五文件最终范围再次核验通过，结果记录在 `protection-final.log` 与 manifest。`git diff --check` exit 0。

本包 native 启动 0 次、GPU 使用 0 次、训练/optimizer update 0 次。旧 native proof、历史 receipt/结果与模型均保留原身份；没有重贴旧资格。CPU Gf 投影与 SDK getter 替代仅证明软件数学、接线和数值回读。修改后的非中心 producer 仍需要新的独立 native calibration accuracy proof；同 K readback 不能授予 native accuracy、Teacher 或 Student 资格。
