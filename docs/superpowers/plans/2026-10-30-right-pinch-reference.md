# 双手采用右手对捏几何的实施计划

> **For agentic workers:** Use subagent-driven-development for bounded pure geometry helpers and independent read-only diagnosis. Only the parent may start Isaac.

**Goal:** 按用户提供的右手截图，将左手从圆杆深进掌内改为拇指、食指附近对捏，并验证实际提升。

**Architecture:** 沿用已授权的原生几何搜索与物理验证设计。由已记录右手的真实link位姿构造圆杆坐标系内的参考，反射到另一端后用左手自身FK和合法旋转拟合；不直接复制关节角或反射四元数。只用原12主动、10联动从动与原执行器限幅，不设力目标，不改质量、重力、摩擦或桌面。

**Tech Stack:** 现有Torch、USD、SciPy、pytest与Isaac环境；不安装依赖。

## 1. 复核实际几何

- [x] 停止记录回放，分别检查C/case35初始、闭合及保持帧的两手实际凸包与圆杆穿透和桌面间隙。
- [x] 检查右手保持几何；已确认需要杆轴/重力的无滚转坐标系，且原食指patch偏离实际接触。

## 2. 合法几何匹配

- [x] 新增`tabletop_pinch_reference.py`及`tests/test_pinch_reference.py`，先红测，再实现Y反射、合法刚体拟合、坐标系转换和无滚转杆轴坐标系。
- [x] 多启动拟合两手自身FK的完整末节参考，保留完整手物/自碰/桌面代价；CPU独立复核1组零穿透、最小桌面间隙0.459mm，双手AABB分离310mm。
- [x] 生成包含来源、左右手各自角度、手腕位姿及完整几何检查的候选文件。几何不作为承力证据。

## 3. 实际验证与展示

- [x] 完成固定微闭合A，原始480Hz测量和源身份保存；未提升，拒收为教师。
- [x] 完成逐指寻触B，提升独立按时间执行，无力目标，普通PPO不变；仍未提升，拒收为教师。
- [x] 检查验收标志，未将几何可行或记录回放冒充成功教师。
- [x] 打开Isaac展示新姿态，暂停在本轮记录0.2s；明确标注NOT LIVE。
- [x] 更新README、CHANGELOG和验证记录，保存本地源码及原始数据后提交10-29；不启动长期PPO。
- [ ] 后续仍需实现新姿态下稳定承力、无明显穿透的真实提升；本次不宣称已完成。
