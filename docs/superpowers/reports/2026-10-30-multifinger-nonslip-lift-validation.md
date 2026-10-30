# 10-30 分支：多指分担与防滑提升的当前进度

这是用户要求保存并上传的当前开发版本，**尚未实现稳定多指抓取后的防滑提升保持**。代码、几何候选和实际短测分别记录，不将纯测试或几何可行当作成功。

## 本次实现

- 所有五指均可参与，不再锁定拇指/食指组合；双手仍为12主动+10从动，原0.2Nm、主动0.3rad/s上限、质量、重力、桌面和材料保留。
- 通过原生接触点/法向、COM点Jacobian移点及mimic折叠，以原位置驱动补力或退让；内部夹持强度自适应，参与指目标均衡分配，不固定用户目标N值。
- 原始144步窗口检查持续承力、原始CV、各指与两手分担、对向法线；至少任意两指/手持续承力。不能挑出两个稳定指而忽略另一根间歇重载指，也不能将零力零方差当作稳定。
- 以杆上三点相对各手的坐标检查漂移；稳定后锁定参考，不每帧重置来掩盖滑移。目标为实际底面离桌30mm、相对偏移≤0.5mm并连续保持5s。
- 普通PPO和旧诊断默认行为保留；当前没有启动PPO。新入口仍是显式诊断。

## 离线与单元验证

几何搜索使用每手16 starts、200 steps、4 CPU线程，搜索及独立生产模型复核共约79.4s，输出4组初始候选。每手至少三指近接触并有近对向区域；最大采样手入杆0.08253mm、杆入手0.07387mm，自碰及双手互碰为0，最小桌面净空0.31886mm。这些是源USD采样/凸包平面结果，尚不证明原生碰撞面的完全等价或实际承力。

规范候选保存为 `configs/dh116_training/multifinger_pregrasp_v1.json`，明确 `geometry_only=true`、`qualified=false`；搜索脚本为 `scripts/generate_dh116_multifinger_pregrasp.py`。运行元数据和日志保留在本地及服务器，不提交Git。

本次最新执行：

```bash
DH116_NUM_THREADS=1 bash scripts/python.sh scripts/generate_dh116_multifinger_pregrasp.py --self-test
DH116_NUM_THREADS=2 bash scripts/python.sh -m pytest -q tests/test_multifinger_control.py tests/test_multifinger_monitor.py tests/test_pinch_reference.py
DH116_NUM_THREADS=1 bash scripts/python.sh -m py_compile scripts/run_dh116_multifinger_lift.py scripts/generate_dh116_multifinger_pregrasp.py src/go2_pvcnn/control/dh116/tabletop_multifinger_runtime.py
```

搜索自测通过，定向回归 **214 passed in 2.75s**，编译检查通过。监测模块含CPU/CUDA、partial reset、未填满窗口、间歇重载和固定参考漂移等用例，均经过先红后绿。该定向回归不是全仓测试，不证明实际抓取成功。

## 实际Isaac短测 A

单个Isaac进程、4环境、每环境2880步（6s），墙钟约200.45s。原始结果保留在 `/root/hand/runs/multifinger_lift_20261030_a`；运行入口和控制源码冻结在 `runs/deployment/surface_teacher_20261030/multifinger_a_sources`。

4组均未通过稳定/均衡条件，phase保持0，未进入提升；reset均0，成功0，教师0。部分环境中食指、中指或无名指已有实际承力，但仍有缺接触、力波动和载荷不均。不能因设有“稳定后提升”逻辑就宣称已经实现实际提升保持。

```bash
DH116_NUM_THREADS=4 bash scripts/python.sh scripts/run_dh116_multifinger_lift.py --candidates configs/dh116_training/multifinger_pregrasp_v1.json --output runs/new_multifinger_lift --seconds 6 --wall-seconds 600 --maximum-poses 4 --reference seed --closure-margins 0 --thumb-spreads 0
```

加长持续时间尚未验证，当前不得将短测替代30mm/5s全流程验收。

## 已知未完成项

1. 多指持续接触与均衡尚未收敛，仍需实际控制验证；源几何与原生接触面之间的差异尚未完全解释。
2. 滑移时撤销超前的外部手腕目标仍需完善并实测；只减小上层目标可能残留 `TensorWorld.palms` 的队列目标。本次phase0未触发此路径。
3. 新入口的在线5s结果还需独立重放同一窗口的原始各指力、相对位移、固定参考与源几何。上传前已禁止沿用旧height-only事件导出成功案例，并将旧高度摘要单独标识；没有把旧 `first_qualified` 改文案后当作防滑成功。该记录防护只做定向回归/编译检查，未另开一次Isaac。
4. 仍需完整真实提升保持和全过程几何审核。本分支未接入PPO教师发布，未声称硬件标定通过。

当前上传只保存这些已实现和未完成状态；运行日志、原始张量、压缩备份、缓存和密钥不进入Git。
