# 双手 Pinch 材料方向原生端点：真实数据与原动作路径对照

本包对已经完成的独立18步原生运行做只读分析，不重新启动Isaac、不推进物理、不训练。新增 `audit_dh116_native_director_endpoints.py`，复用原director逐行validator与原closure archive validator，不另设放宽的原生门槛。

## 实际结果

- 原生代码/配置指纹：`abfae81f2cc9bfad7cb1698ad88afd5f9aeb8a89f87861fbd436c6defc16cf74`，测试HEAD `c5b8e7679f68f9fb9b550889ff9f2791ac8e3877`。
- GPU3，1环境，单实例；2初始化+8零动作+8原手指动作=18 SDK步，12杆子步/SDK，0 PPO更新。墙钟77.042022秒。
- 外层与inferior自然退出0，signal=null；正常退出GDB栈、动态库与maps已保存。不是崩溃栈；没有SIGTERM/SIGKILL/os._exit。
- reset SDK2 + SDK3…18，共17组真实同钟数据；每组41节点、40个真实材料段XYZW四元数、末项padding独立。共680个真实段端点四元数，未声称连续接触子步轨迹。
- director reader所有弱引用在owned drop、STOP和app.close后均false；共享Tensor View与其他native owner也通过既有独立退出validator。
- 0手指—杆pair样本；7872个杆—桌面子步pair样本。7872不是唯一接触数量，不是手指有效承托。
- 原手—桌面静态Tensor测量仍NA；原全部碰撞几何保护保留，不能用NA宣称没有接触。

## 对照原动作路径

对照已发布原closure实际回执 `dd64cce38312cb3fe044e210c8560f44fae33b218e5b8e2ba97435a3969c9bc0`，它与本次属于两个真实独立episode。按相同SDK tick、rod tick、policy/substep、相同原始24D action及profile绑定后，16行节点位置、左右17D实测q/qd、SDK目标的最大绝对差均为0。代码只报告这两个有限运行的逐值结果，不声称长期可重复性、任意策略下无影响或因果证明。

第一窗口动作全0，第二窗口只在原索引1/2/7/8给0.1/0.1/0.05/0.1；腕部全部0。旧动作与物理未被修改。

## 几何统计及解释范围

新增`geometry_metrics`按节点/段计算P50/P90/P95/min/max；先拒绝非有限、非单位真实四元数、退化线段与无效rest，再进行角度运算。四元数q/-q等价，padding不参与段统计。真实rest来自float32构造节点差分，并非不存在的native rest getter。

SDK18相对实际reset：

| 项 | 数值 | 解释 |
|---|---:|---|
| 最大节点位移 | 8.948454372e-8 m | 位移范数，不能称物体净Lift |
| 最大四元数范数偏差 | 1.014079680e-7 | float32读回误差 |
| 材料轴/当前线段最大夹角 | 3.818191821e-6° | 接近数值精度，不作物理精度证明 |
| 材料框架相对reset最大转角 | 4.182619578e-6° | SO(3)转角，不是接触滑移 |
| 段长度相对变化范围 | -2.980232416e-7…2.980232194e-7 | 无量纲，非独立材料标定 |

每行保留完整run/env/episode/reset/physics/rod/timestamp与material reference。当前没有手指contact station，接触子步材料方向、continuous_s、tangential_slip与physical_error_bound继续NA；不能用杆整体静止、框架转角或节点index给0滑移。Teacher准入与抓取资格均false，三抓法Teacher就绪仍0/3。

## 测试、证据与恢复

先新增12项测试并运行，预期缺实现的ImportError保存在 `director-summary-red-Iv9rrA`；随后实现，12项通过。发布前合并旧材料读回和closure实际归档回归：**63 passed，0失败、0跳过，12.91秒**。测试含整体平移、轴向扭转、q/-q、padding、应变单位、NaN、退化几何、错误SHA，以及本次真实归档只读重放。CPU合成数据只用于函数测试，没有被写入原生数据。

命令：

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_dh116_native_director_endpoints.py tests/test_dh116_material_director_readback.py tests/test_dh116_dual_pinch_closure_readback.py
bash scripts/python.sh scripts/audit_dh116_native_director_endpoints.py --output <独立目录>/native-summary.json
```

原生回执SHA `11731b7d0a95edfd9573c53ccd08af238f9db47831eb37bf7b3ea0c27b2d9dd6`。
原生物理JSONL SHA `67c1000f1f26f7f393825c641044bbfbe57ba323e0f38808efdf32c56f2427a1`。
独立退出审计SHA `214c27cab43d21f18eda400fa59680326d5e40d471ea0694f573973d8b9a0431`。
本次分析结果SHA `810dccb287f7af537533d55001d0b3aaba1769980818ab24771977599189fbbf`。

私有分析目录：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/director-summary-release-Hfanru`。完整maps、日志、原始数据均不上传。528保护源码与2旧Checkpoint核验通过。原生18步预算已耗尽，不允许重跑。下一步分别推进安全接触姿态和显式仿真参数/稳定性/R0定义域绑定；不把本包升级成PPO资格。

| 文件 | SHA256 |
|---|---|
| scripts/audit_dh116_native_director_endpoints.py | 05897b4cb48ec8203aa4f83f5d73543134aca19ae1bc54c224db51b13ec22e70 |
| tests/test_dh116_native_director_endpoints.py | c3299e65013e052a53931c4d0e067f20ad6511fdfb64de11f1f6f30bacdc1740 |
