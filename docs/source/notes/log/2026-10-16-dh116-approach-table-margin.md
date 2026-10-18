# 恢复接近末段桌面余量诊断

父节点：[T500 D4实际抓取](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。阶段：恢复后的approach。Baseline/Candidate：5e3bcc4+dirty，未改产品代码。

## 实测终态

stage_restore_trial1_approach 在step30075停止，PID18934/session97738已结束，不再等待或自动重启。恢复复验成功，但approach FAIL，close/hold/lift NOT_RUN。最终1000次点Jacobian探针通过：位置最大6.14e-7m、Jacobian最大6.44e-7m/rad。

summary.measured_source_guard_failure：左finger22_link对桌面保守下界0.0009995456110796694m，阈值1mm。前一步/步前保护仍通过，非目标净接触为零。不是已证实碰撞或穿透，更不是抓取成功。

## 离线根因核验

读取固定双Panda DH116 USD，以SourceCollisionScene源凸包、SideKinematics FK比较同一finger22_link；读取失败summary实测active_q和原approach证书。单核、数值单线程，未启动Isaac。

- 失败时名义轨迹时钟3.4075s：桌面几何间隙1.284095664mm；扣运动余量后1.202283906mm。
- 实测active_q重建：几何间隙1.081403871mm；扣余量后0.999592112mm，与运行时实测body结果差约0.047µm。
- 名义终点：几何间隙1.201414196mm；扣余量后1.119602438mm。
- 原速度上限×dt的全关节保守运动余量0.081811758mm。
- 失败左臂最大关节跟踪误差0.000558166rad；手耦合误差4.07e-6rad。实际臂姿态差可解释约0.203mm间隙损失；不是Jacobian绑定失效证据。

结论：源名义路径仅有小间隙余量，未覆盖实际跟踪偏差；不能依赖源证书代替物理验证。下一步为D4-approach-tracking-margin：保持同一wire与指腹目标，搜索更大桌面余量姿态/路径并增加跟踪包络筛查，所有1mm保护、速度/接触门不变。当前未生成修复候选，未启动新试验。原阶段文件受配置SHA约束，变更配置后不能篡改pin强行复用。

Key files：Go2Pvcnn/go2_pvcnn/control/dh116/{collision_scene,approach}.py；Go2Pvcnn/scripts/verify_fixed_dual_panda_dh116.py；外部assets/dh116_v2/stage_restore_trial1_approach/{summary,point_probe}.json。
