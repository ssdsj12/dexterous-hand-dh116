# Power 右掌恢复净空与固定腕指垫进近

从历史实测 SDK 6504 的杆和 R6 已认证的第 5 个控制窗口末端出发。入口保留右掌节点 9 和拇指节点 4 的几何预压，但无名指节点 3 已经开离 +0.607832 mm。本轮保持左手原目标，冻结右腕转角，只对右腕平移和右拇指主动目标做小幅协同修正。随后分别测试固定腕无名指重接触和固定腕小指闭合。

## 净空恢复与拇指对照

| 同一右掌节点 9 | R6 截止状态 | 两个重新就位窗口后 |
| --- | ---: | ---: |
| 登记 SDF 最小节点间隙 | −221.937 µm | −50.001 µm |
| 原 cooked 三角扇最小节点间隙 | −244.193 µm | −72.892 µm |
| source 凸包相对整杆胶囊的间隙上界 | −2.405093 mm | −2.249951 mm |

两个 60 Hz 窗口保持右掌 9 和拇指 4 的独占几何接触，原 cooked 掌面逐 480 Hz 样本也通过原 250 µm 深度护栏。右腕平移为 (+0.0692, −0.0490, +0.1502) mm，右拇指两个主动目标分别改变 −0.0000589、−0.0017593 rad。将同一组目标弦的拇指指令固定在原值时，其他腕和手指目标及采样时序完全一致，拇指节点 4 在第 13 个采样点达到 −155.771 µm，超过原 150 µm 限值。有拇指补偿的路径在终点保留 −45.000 µm。这个对照测量主动目标，不改变实际顺应性或力控制 ABI。尝试在这两个接触同步恢复无名指会在第 12 个采样点失去拇指接触。

## 固定腕进近

- 无名指节点 3 的登记 SDF 在分支的第 15 个采样点出现负间隙。连续几何只认证第一个窗口，认证端点仍开离 +0.207630 mm。下一目标的 cooked 无名指包络间隙上界 −0.196902 mm，超过 −0.150 mm 允许深度，同时精确对向条件也未通过。该分支未并入组合指令。
- 小指在新腕位单独闭合 0.262691 rad，共 53 个连续认证窗口。认证端点在节点 6 仍有 +0.574099 mm 的登记 SDF 间隙。下一目标的 source 小指凸包间隙上界 −0.211254 mm，超过许可深度。仅看登记 SDF 时，第 453 个采样点才出现独占节点 6 的负间隙，位于认证边界之后。

合并后的 60 Hz 序列共 60 个窗口，由原 5 个窗口、2 个重新就位窗口和 53 个小指闭合窗口组成。逐段核对原 24D 速率与精确反序取反的命令。左手分量都为零。整个序列最大归一化关节分量为 0.991286，右腕平移范数最大为 0.959535，右腕旋转范数最大为 0.330880。两个新阶段分别覆盖 16 和 424 个 SDK 采样间隔，各独立重查的最小连续几何裕量分别为 8.709 和 2.364 µm。

右掌 source 凸包持续相交。连续证明仍遵循原选定掌面的登记 SDF 护栏，另对原生 cooked 三角扇逐样本检查同一材料站位和深度。source 凸包相交由独立几何检查明确保留。这里的预压是几何负间隙。无零载接触阶段、实际正反作用力和整手包握资格记录。新的原生步数、PPO 更新均为 0。

## 复现

使用 Ubuntu DH116 Python，并加入原 Isaac USD 的 Python 和动态库路径。

~~~bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B \
  -m tools.audit_dh116_power_contact_reseat \
  --palm-receipt "$EVIDENCE/ag2404-power-palm-witness-native/outer.json" \
  --continuous-audit "$EVIDENCE/ag_power_thumb_route_continuous_6504_v4.json" \
  --partial-audit "$EVIDENCE/ag_power_partial_opposition_6504_v1.json" \
  --bilateral-detour "$EVIDENCE/power-bilateral-detour-archive-v1.json" \
  --certified-prefix "$EVIDENCE/sync-power-round6-corridor.Gv0oyf/result.json" \
  --right-palm-cooked-record "$EVIDENCE/ag2171-power-palm-parity-native/gpu/native/right_base_link.json" \
  --output "$RESULT"
~~~

结果为 sync-power-round7-contact-reseat.vKLRjs/result.json，SHA-256 为 82ad76c356528fb8b83b55674f5cb9da3dcd6bd337818d0cee290c6d29c94fdf。122 项 CPU 回归通过，其中新增 5 项。日志为 sync-power-round7-cpu.lQo2f1/pytest.log，SHA-256 为 9d0e3b668e74f35ba008b67a46bd01e6359684186d89b7e619a089e030fec1d3。冻结原生配方和原生执行清单未修改。
