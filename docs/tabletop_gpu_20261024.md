# 512 环境 GPU 扩容记录（2026-10-24，本地未上传）

用户要求利用剩余显存增加到512环境。保留双DH116、高摩擦中空刚体吸管、
20维动作/136维观测、反作用力反馈、close直接复位、实际抓稳后lift及全部硬保护。
使用单Isaac进程、CPU0、数值单线程；不删除旧资产和结果。

## 已确认问题与修复

1. 原4环境后端在CPU运行，约46条训练transition/s。直接改环境数量不能使用GPU显存。
2. v42/v43 GPU试验虽然数值有限且未触发保护，实际接触力全零，指尖穿入物体。
   Kit详细日志显示32段细长薄壁凸包CPU回退。因此单凭无异常不能准入GPU训练。
3. GPU运行时将每个碰撞网格顶点归一化进行烘焙，再用平移和非均匀缩放恢复
   原世界顶点。保留管腔、壁厚、质量、惯量和材质，不修改原USD资产。
   独立USD舍入测试最大世界顶点误差1.675e-11m。
   NVIDIA说明GPU凸包兼容性与缩放前的形状长宽比有关：
   https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html
4. v44两环境实测接触恢复，每手峰值约0.20N；6个完整准入回合安全。
   这是接触/物理可用性证据，不是成功抓取。
5. v45成功加载512环境，显存约3.7GB，尚有大量余量；但是逐环境Python任务
   仅约71transition/s，主动停止于接触预检，没有开始PPO、没有产生训练成绩。
6. 增加NumPy批量任务计算及批量物理I/O，逐物理子步保持原规则。差分测试对照
   原TabletopEnv的命令、观测、奖励、阶段、超时、硬故障、NaN、有限值溢出、
   接触中断、滑动、非法首动作和混合槽提前终止。保留原episode/terminal处理。

## 实现

- `scripts/train_dh116_tabletop_gpu_ppo.py`：GPU专用入口、合同校验、复位/接触预检、PPO。
- `tabletop_gpu_isaac.py`：克隆场景、两组手/一组物体/四组接触传感器批量视图，
  全部视图的环境路径顺序断言、GPU环境ID碰撞隔离、批量读写。
- `tabletop_batch_geometry.py`：完整碰撞凸包净空和薄壁烘焙坐标归一化。
- `tabletop_batched_task.py`、`tabletop_gpu_vec_env.py`：批量等价控制/奖励/任务和SB3适配。
- 每个环境保留summary；前8个环境记录完整轨迹，防止512份完整轨迹常驻内存。
- 同维度权重迁移比较除环境布局/后端外的全部合同，再加载原检查点核对；
  保留策略，重新创建优化器和学习计数，重新做物理准入。

## 运行阶段

v41旧检查点保存于17764步、118优化epoch。v46三十二环境实测已结束：
64只手接触预检、不同槽物体状态/单槽复位映射检查以及全部96个完整准入回合通过。
状态probe_complete，正常退出；抓取仍为0/96，不把安全失败回合算抓取成功。

最终512进程v47已启动：`runs/tabletop_gpu_20261024_v47_env512/monitor.json`。
当前为contact_preflight，1024只手均测得接触；显存约3669MiB，预检约107 transition/s。
仍有主机端/仿真吞吐瓶颈。容量验证不等同于加速或成功学习；512完整准入和PPO
优化更新尚待运行。进程自动按接触预检→三批完整准入→PPO→最终评估顺序执行，
不需要再次手动启动。v47源代码哈希保存在source_manifest.json。
配置从v41同维度权重初始化，计划1048576新transition、每8192步保存，512次最终评估。
每轮rollout为64×512=32768条样本，不能将首次8192步存档当作已完成一次参数更新。
原检查点和全部试验结果保留。

## 软件验证

- 批量几何与既有环境、反馈、向量训练相关191项回归通过。
- 批量任务17项差分测试通过；新增危险边界先复现失败再修复。
- 最终六文件合并回归208项通过；唯一警告来自刻意触发的旧实现float32溢出对照。
  审查复核通过。512完整物理准入和新模型优化/重载尚未完成，不沿用历史成绩。

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh -m pytest -q \
 tests/test_dh116_batch_geometry.py tests/test_dh116_batched_task.py \
 tests/test_dh116_tabletop_env.py tests/test_dh116_force_feedback.py \
 tests/test_dh116_tabletop_vec_env.py tests/test_dh116_tabletop_vec_training.py
```
