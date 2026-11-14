# 右 Pinch 源凸包与杆段距离复核

基线fe7b42ccbb0410d87ec6e420d56ee1812ac8d1a5。读取固定失败V4候选，未再优化Reset，未运行Isaac或PPO。

新增tabletop_polyline_closest_support.py：proposal/refine/validate。883个原始凸包顶点不改，转为完整1762面halfspace；优化变量仅凸包内点XYZ和杆段alpha共4个，最多64迭代/段。本次最多4段、wall60s，仍保留全部原40相邻杆段。

优化只提出单位法向；最终净空必须对全部原始顶点和段端点重新计算支撑下界。5mm杆半径、2µm数值裕量、原桌面2.1mm门不变。原tabletop_polyline_support_certificate和原生使用路径不改。

另保留凸包内可行点/段上可行点作为距离上界证据；数值上界额外加2µm。近似QP成功不能直接当作净空证明。任何优化失败保留原下界，未处理段不删，不通过全局最小值掩盖失败。

## 结果

scripts/audit_dh116_right_pinch_closest_support.py只审V4末态source finger13，结果：

| 相邻杆段 | 原下界mm | 独立复算下界mm | 可行点距离上界减半径mm | 迭代 |
|---|---:|---:|---:|---:|
|2|4.819430|4.841407|4.845407|3|
|3|-0.045660|-0.044425|-0.040425|7|
|4|-0.038932|-0.037122|-0.033122|8|
|5|4.692240|4.693835|4.697835|5|

段3/4的可行上界仍小于5mm半径；这说明该源凸包与杆段胶囊存在几何重叠，而非仅证明方向不够紧。这里的数值不是实际native碰撞穿透深度，不是PhysX cooked近端碰撞实测，更不能计为接触或抓取成功。

审计5.429783s，exit0只表示完成；original_other_shape_safety_passed=false，native_recipe_created=false，teacher_admission=false。没有改变末态指腹SDF失败事实，没有全路径/新原生。

输出/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/closest-support-20261113-0_9gw9ho/run01/result.json，SHAe0f4c1a4bb6341826a867e45cd84c55f7aaa01640eedc453a31811b924702b37。
父候选SHAbd311cc064544f0b6c4ee295ac49d7b5e72f883dacd22c9f36ea6daa8787b865。

## 保留的失败

先前一次CPU凸组合QP为883顶点权重+alpha的884变量稠密SLSQP，成本过大。仅中断该确认为本任务的CPU进程，保留KeyboardInterrupt；不是原生进程，不称为自然退出成功。随后改用等价4变量halfspace形式，没有删顶点或改物理。
失败元数据/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-reset-direction-v4-20261113-379rl4yl/closest-support-interrupted.json，SHA849e2afcf2484243970559511b13b69018783197ddb07856d6de995bcbe61ab4。

## 测试

先9个缺模块RED，再通过清晰/重叠、预算、完整几何篡改拒绝、平移不变、优化器失败不晋级、虚假上界拒绝等15新测试。相关回归54 passed、0 failed、6.27s。
日志/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/closest-support-full-20261113.log，SHA429b464d52cd221a65ba02de6cac43f38040a232592288c6931a2b486e0cce3a。

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_polyline_closest_support.py tests/test_dh116_polyline_support_certificate.py tests/test_dh116_right_pinch_reset_direction.py tests/test_dh116_right_pinch_measured_direction.py
```

下一工作包应调整独立右Reset的掌-杆相对方向，而非放宽碰撞门或继续盲目闭合。原支架/材料/控制/关节保护/成功标准、528文件与2Checkpoint由发布器复核。R0稳定性和实际有效承托仍独立待资格，Teacher0/3；本包WireSeg训练0。
