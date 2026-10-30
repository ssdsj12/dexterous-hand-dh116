# 右手对捏参考适配与实际验证

用户指出旧记录中左手圆杆滑入掌内，要求双手参照右手方式。当前完成了新预抓姿和演示，尚未实现该姿态下的稳定物理提升。

## 几何参考与修正

参考为`runs/unrestricted_grip_lift_20261030_c`的case35末帧实际link位姿。原右手保持时密集采样手物穿透约0.275mm，左手约0.900mm；原食指预定义patch距杆4.211mm，未覆盖实际接触，改用完整`finger14_link`/`finger23_link`末节凸包最近表面点。

圆杆的轴向滚转不唯一，不能用完整物体四元数决定手掌摆放。采用由杆长轴和世界重力上方向构成的无滚转参考坐标系，再做左端Y反射及det(R)=+1刚体拟合。左右手分别用自身FK优化；其余手指和手掌朝向也可调整以避桌。

最初只平移物体中心的A拟合产生0个可用组合，作为失败诊断保留。修正参考后的B，左右各32 starts/400 steps，耗时99.083/73.689s，输出1组候选。独立CPU复核：手入杆、杆入手和自碰均0；左/右桌面间隙0.4720/0.4590mm；双手AABB的Y方向分离310.467mm。接触区域在拇指/食指末节，几何不等同于实际承力。

## 实际试验

- 固定微闭合A：3环境，闭合增量0.0015/0.003/0.005rad，之后原物理手腕按计划上移。实际FK与关节读回一致，双指未形成持续承力，圆杆未抬起；选择结束诊断并完整保存约3446个物理步/环境。
- 接触存在性B：1环境、3616物理步、178.701s，无reset；最高实际底面离桌约0.0000685mm，保持0s，失败。没有给圆杆外力、附着、修改重力或材质，也没有PPO更新。

显式诊断的`contact_seek`只调整主动索引1/2/7/8。前8个原始子步负载均值低于0.1N时，以0.02rad/s补闭合；已接触保持，采样深度超过0.8mm时退让。0.1N是接触存在性判据，不是力目标或上限。原0.2Nm、角度和速度限幅保持，lift仍独立按时间执行。采样深度来自60Hz源顶点，**不能声称它限制了完整连续碰撞深度**。本控制尚未抓稳，不能作为已修复的默认控制器。

记录回放明确标为`RECORDED PHYSICS / NOT LIVE`，高度和力来自原始数据，刚体变换只用于可视化。它没有执行新的物理步，不能当作另一次提升验证。后续episode的控制器回放还需要原reset与采样相位；代码已标注未验证独立回放。

## 验证与保留

- 纯几何/寻触helper累计103项红→绿；最终42文件回归`1043 passed in 19.66s`。
- 独立静态规格/质量复核通过；实测原始状态、接触、故障和配置单独保存。运行源码冻结为`right_pinch_physics_a_sources.*`和`right_pinch_physics_b_sources.*`，后者22文件，包含实际运行的包装入口、helper和候选数据。
- 预抓姿配置为`configs/dh116_training/right_pinch_pregrasp_v1.json`，明确`qualified=false`。本轮实际失败，合格教师0，没有把旧姿态下达到高度的案例算到新姿态。
- 物理记录位于服务器`/root/hand/runs/right_pinch_physics_20261030_a`及`_b`；几何记录位于`/root/hand/runs/right_pinch_reference_20261030_a`及`_b`。本地备份和回执保留在`C:/Users/xk/Documents/project/hand-deployment-10-30/surface-teacher-validation/`，运行数据不进Git。

## 复现

在`/root/hand`中使用全新输出目录：

```bash
DH116_NUM_THREADS=4 bash scripts/python.sh scripts/run_dh116_pinch_contact_seek.py --candidates configs/dh116_training/right_pinch_pregrasp_v1.json --output runs/new_right_pinch --seconds 10 --wall-seconds 600 --maximum-poses 1 --reference seed --closure-margins 0 --thumb-spreads 0
DISPLAY=localhost:11.0 DH116_NUM_THREADS=2 bash scripts/python.sh scripts/replay_dh116_grip_recording.py --recording runs/new_right_pinch --output runs/new_right_replay --case 0 --start-time .2 --paused --device cuda:0 --kit_args=--allow-root
```

回放需要有效X11会话，`DISPLAY`采用该SSH会话实际值。展示新预抓姿只是当前完成项；稳定承力和无明显穿透的真实提升仍待解决。
