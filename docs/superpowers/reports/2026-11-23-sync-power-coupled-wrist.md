# Power 左腕和中指、无名指耦合重拟合

从发布的 963 窗口路径终点起步，使用历史实测 SDK 6504、rod tick 78024 的同一条 41 节点杆。先独立重验双手原有局部接触几何。右手保持掌面 9、拇指 4、无名指 3 的原目标。左手保留掌面 29、拇指 37 和食指 36，分别尝试加入中指 35、无名指 33 以及同时加入两指。

左腕用节点 29 的 SDF 掌面法向和杆的局部切向构造正交坐标。法向、切向和横向平移各允许 ±6 mm，绕三轴的旋转向量分量分别允许 ±0.10、±0.18 和 ±0.10 rad，原控制器的全局腕工作区同时有效。拇指外展、原指垫和新增指垫的主动关节独立拟合，受原耦合关节限位和种子附近 ±0.25 rad 约束。每组尝试五个种子，覆盖静止腕、两个相反的切向滑移与绕杆滚转组合、两个相反的法向与横向组合。对后两组先用最小二乘找接近可行的种子，再执行有界约束拟合。锁定 SDF 和原 source、cooked 凸包护栏分别重新读取并验收。

15 次拟合均未通过精确端点验收。联合零扰动种子保留了左掌 29、拇指 37、食指 36 和中指 35 的独占站位。原拇指与食指的法向余弦为 −0.9848。无名指 33 仍有 +3.487 µm 的 SDF 间隙，而 cooked 无名指包络的杆间隙上界为 −0.461116 mm，至少超出 −0.150 mm 穿透限值 0.311 mm。将该无名指恢复到原关节目标后，独立复核仍发现 cooked 中指超出穿透限值至少 0.287 mm，桌面净空 2.087751 mm 低于 2.100 mm 护栏。正向横向种子将无名指的 cooked 上界改善至 −0.180715 mm，但掌面和食指失去原站位，无名指本身仍有 +0.317 mm 间隙。反向横向种子触发了 source 指节 finger13_link。

没有实际候选通过路径门槛。离线工具要求前向 60 Hz 24D 命令和倒序取反的逆向命令均与同一组目标增量一致，再检查连续 source、cooked 包络与 480 Hz 样本上的独占材料站和原拇指、食指对向。回归用例覆盖了双手命令中的右手保持、逆向速率、原指垫法向失配，以及近点算法返回无新方向时保留原凸包证书方向的处理。搜索结论只覆盖这些有界拟合。更换材料站、其他左腕姿态和右腕重新定位仍需另行验证。新增 Isaac 步数和 PPO 更新均为 0。

## 复现与证据

在 Ubuntu DH116 Python 环境中加入原 Isaac 安装的 USD Python 路径和该目录的 bin 动态库路径。以下命令使用相同线程设置和历史证据文件。

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B \
  -m tools.audit_dh116_power_coupled_wrist \
  --palm-receipt "$EVIDENCE/ag2404-power-palm-witness-native/outer.json" \
  --continuous-audit "$EVIDENCE/ag_power_thumb_route_continuous_6504_v4.json" \
  --partial-audit "$EVIDENCE/ag_power_partial_opposition_6504_v1.json" \
  --bilateral-detour "$EVIDENCE/power-bilateral-detour-archive-v1.json" \
  --fixed-wrist-screen "$EVIDENCE/sync-power-round2-qualified-screen.DImGBP/result.json" \
  --output "$NEW_RESULT"
~~~

最终回放退出码为 2，结果文件为 sync-power-round3-coupled.wWYiEv/result.json，SHA-256 为 58b88020c23ecb30acfe22e25ef3cb6230cc7ea42244c609e866de68d2765a24。辅助恢复单指的几何复核日志为 sync-power-round3-reset-one.FgcGWx/screen.log，SHA-256 为 f21edf75b7eb2a26293761ff40e0ee73ede7a3233d8ea5ba7d0c581ca309531f。

96 项 CPU 回归通过，其中新增 5 项。日志为 sync-power-round3-cpu.0Ou8Hk/pytest.log，SHA-256 为 ec9f22eb3894baf052152d23db619de10b8f03d39953fc69680302dd6be4dbc6。工具、测试和报告均在冻结输入清单之外。
