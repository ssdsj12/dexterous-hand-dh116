# 右手式对捏：持续承力与实际提升复核

本次在 `right_pinch_pregrasp_v1.json` 上继续验证，无固定目标力、无 PPO 更新。结论是：四指持续承力有所改善，但力仍明显波动；实际提升失败，合格教师仍为 0。

## 对照与实测

| 试验 | 唯一主要变量 | 结果 |
|---|---|---|
| C，静止保持 6 s | 相对此前 B 取消手腕上升 | 两只拇指全程 0 N；前 2.5 s 原始字段与 B 逐值一致。问题在提升前已经存在 |
| D/E，原生读回 | 只增加 collider/manifold/cooked 数据读取 | 两侧 thumb14 都启用，rest offset 为 0；存在 manifold，但间隙为正且力为 0 |
| F，静止保持 6 s | 以 native separation 代替源顶点 SDF 作为退让反馈 | 拇指开始承力，但四指同时 >0.05 N 最长仅 2.08 ms；右拇指尾 1 s 为 1.537±2.634 N |
| G200，静止保持 6 s | 相对 F 仅将 mimic naturalFrequency 400 改为 200 s⁻¹ | 四指持续承力仅 4.17 ms，未解决问题，不采用此候选 |
| H，静止保持 4 s | 相对 F 仅以最近 8 个原始步的最小值判断持续接触 | 四指同时 >0.05 N 连续 3.0896 s，>0.1 N 连续 2.4667 s；仍有明显力波动 |
| I，实际提升 | 使用规范入口的 native/minimum 选项，恢复原独立上升计划 | 3459 个原始物理步，实际双腕均上移 46.9208 mm；圆杆底面最高仅离桌 0.006880 mm，随后全手失联；reset 0，成功 0 |

I 原计划 10 s；确认上升后持续失联、圆杆仍留在桌面后，通过 `stop_requested` 保存记录，在 7.20625 s 结束。它不是安全保护停止。请求腕升高约 47.0208 mm，实际约 46.9208 mm，说明本轮指令已执行。不能将手腕高度当作物体提升成功。

独立 CPU 源几何审计耗时 19.1 s，覆盖全部 3459 帧：左/右最大手入杆 1.43255/1.45079 mm、最大杆入手 1.18053/1.28829 mm，采样自碰均 0，最小桌面净空 0.47214/0.45922 mm。穿入峰值均在 `finger23_link`，分别出现在 3.12708/2.90208 s。末 1 s 全手失联，穿入与采样自碰均为 0，桌面净空约 40 mm。检查使用所有源 hull 顶点、生产表面采样和完整凸包平面；杆表面及自碰仍是采样检查，不是连续全表面碰撞证明。本轮实际提升和源几何要求均未通过。

H 最后 1 s 的逐原始步法向力：

| 指头 | 均值±标准差 N | 变异系数 |
|---|---:|---:|
| 左拇指 | 0.2885±0.1352 | 46.9% |
| 左食指 | 0.6166±0.0231 | 3.7% |
| 右拇指 | 4.1583±2.7990 | 67.3% |
| 右食指 | 6.1515±0.7275 | 11.8% |

持续接触不等于力稳定。I 末期四指力都为 0，必须标作失联，不能把零标准差当作稳定承力。

## 已确认的问题及尚未解释的差异

旧控制用 `mean(last8)` 判定接触，单次冲击会使指头已脱开时仍保持原指令。F 右拇指尾部原判据始终 hold；同一记录离线改用 `min(last8)` 则持续 seek。H 的单变量实测确认持续接触时间增加，但没有消除力波动。

源 hull 与 PhysX cooked hull 不完全一致：thumb14 左/右源顶点 658/657，cooked 各 36；源顶点超出 cooked 平面的最大距离约 0.716/0.723 mm。**这不能单独解释全部原生间隙差异。** E 同一帧的密集表面采样：左 source/cooked SDF 约 −0.820/−0.775 mm，native separation 却为 +0.276 mm；右约 −0.849/−0.734/+0.139 mm。native shape 变换、实际碰撞表面与离线几何的完整等价性仍未确认，未擅自给 separation 减去 1 mm，也未修改 cooking。

独立检查排除了源网格 body-local 提取、子层级 scale、wxyz/SDF 广播的误用；源 hull 独立提取误差约 1.6e-16 m，SDF 与 SciPy 局部坐标计算误差约 2.1e-17 m。没有证据将 force 波动归因为关节角振荡、400 cycles/s 混叠或实际电机扭矩不足。当前 Isaac 4.5/PhysX 106.5.7 支持所用 mimic compliance 属性，400 s⁻¹ 不能直接当作 400 cycles/s。

## 实现与边界

`run_dh116_pinch_contact_seek.py` 增加显式诊断参数：

- `--depth-feedback source|native`，默认 `source`，保留旧行为。
- `--contact-statistic mean|minimum`，默认 `mean`；minimum 初始化的未填满历史按 0 处理。
- `--hold-only`，仅取消计划上升，不新增 lift 准入门槛。

native 模式逐 480 Hz 读取有效 manifold slice，取 `max(0,-separation)`；源顶点深度仍以 60 Hz 另存。缺 manifold 只是未知，不是整指无穿透的证明。0.8 mm 是退让触发阈值，**不是保证不会超出的硬上限**；I 实际观测峰值约 1.216 mm，源顶点深度约 1.440 mm。所有成功记录仍需源表面全过程审查。

I trace 保存实际选择的 min8 输入、用于决策的 native 深度、源深度和 native contact 是否观测到。reset 后首次实际物理步前遮蔽旧 manifold。H 是先行包装试验，其 `contact_seek_trace.feedback_digit_load_n` 仍记录旧 mean，实际控制在包装内覆盖为 minimum；该字段不能直接用作 H 的真实决策输入，需从原始力重建。原文件保留不改写。

原 0.2 Nm 电机限制、0.3 rad/s 主动限速、角度限位、质量、重力、桌面与材料保留；G200 只是单独的联动参数对照，I 仍用 400 s⁻¹。没有给杆外力、添加附着、传送物体或启动训练。μ=5 与柔顺垫 K/C 仍是既有仿真假设，未经硬件标定。新选项只用于显式小批次诊断，没有接入普通 PPO。

## 验证与证据

新增 manifold slice 测试先红后绿，最终 `112 passed in 1.53s`；入口 `py_compile` 及 `git diff --check` 通过。独立静态复核未发现 I 启动阻断。

独立 CPU 决策重建检查 I 的全部 3459 步：从原始逐指力重建 min8，与 trace 的反馈最大误差 0 N；使用记录的决策深度重放 `contact_seek`，关节目标最大误差 0 rad。H 与 I 的前 2.5 s，19 个公共原始字段逐值一致，确认规范入口复现了 H 的提升前行为。该检查不证明碰撞表面等价，也不把静止承力当作提升成功。

服务器记录均在 `/root/hand/runs/`，分别为 `pinch_force_hold_20261030_c`、`pinch_native_manifold_20261030_d`、`pinch_cooked_geometry_20261030_e`、`pinch_native_depth_hold_20261030_f`、`pinch_native_depth_hold_20261030_g200`、`pinch_persistent_hold_20261030_h`、`pinch_persistent_lift_20261030_i`。I 原始状态、控制 trace、力和源/原生深度、配置、审计及图保留；运行前冻结 179 个源码/配置/测试文件于 `runs/deployment/surface_teacher_20261030/pinch_force_validation_i_sources`。

复现 I 使用新输出目录：

```bash
DH116_NUM_THREADS=4 bash scripts/python.sh scripts/run_dh116_pinch_contact_seek.py --candidates configs/dh116_training/right_pinch_pregrasp_v1.json --output runs/new_persistent_lift --seconds 10 --wall-seconds 600 --maximum-poses 1 --reference seed --closure-margins 0 --thumb-spreads 0 --depth-feedback native --contact-statistic minimum
```

仅复核静止承力时加 `--hold-only`。精确回放需要采用记录中的选项与 reset/采样相位，不能改用默认参数。尚未完成稳定双手夹持及离桌提升；后续重点是实际接触表面的等价性、夹持方向及提升期间持续约束物体的能力。
