# DH116 有载闭指准备与力可行性

父节点：[T500 D4](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate：5e3bcc4 + dirty worktree，未提交。Inline/core0 单线程；不是抓握成功报告。

## 输入与发现

- 源路径组合[0,2]、四源表面参考、50cm/8mm自由Cosserat wire不变。`grasp_approach_trial1` session45112仍运行，15:53为721/960步；不重启。
- 连通前弧v5静态四通道通过，图已查看。外区12mm/60°且仅含参考点的连通面，内接触查询6mm/30°。其余全部面保留1mm保护，下界1.663/2.261/2.441/1.500mm；非PhysX cooked或抓握资格。
- 四点法向z=-.4984/-.0331/-.1925/-.1236，均有向下分量。线缆重力约.98621N，原.6N/点只是单测默认，不能用作抬升参数。

## 静力与驱动预算

新增 `dh116/grip_loads.py`：64射线内接摩擦锥、全6D平衡及独立残差回算。`assets/dh116_v2/grasp_support_loads_v1.json`：μ=.5时每点上限.6N或1N均不可行；上限2N时存在载荷 .98759/1.26861/.86344/.71329N，力/矩残差1.52e-11N/1.10e-14Nm。μ=.35、上限3N不可行，不能声称摩擦鲁棒。刚体静力预筛不证明柔性线缆夹持。

接触力折算左食指约.14288Nm；K=5静态误差预测1.64°超原1°门。后续短时pilot显式使用合成K=10/D=.2、arm60/hand.2Nm，非厂家参数。`drive_headroom`将PhysX drive预算设为总限值减独立gravity绝对值，避免两者叠加超过预算；passive保持0。旧运行证据不改写。

## 实现与验证

- `--closure-pilot`是独立初始切触诊断，不是完整approach→grasp。已接实际PhysX COM点Jacobian、四点显式力目标、全50体/完整非前区保护、实际反力与完整wire记录，尚未GPU运行。
- 接触必须有正法向载荷≥5mN、≥99%该步接触时长、非分离速度≤1mm/s及接触几何；距离不证明抓住。
- 外部 `DeformX/PyElastica-Mesh/co_sim/moving_contact.py`新增精确同时间不可变运动缓存，不改积分/接触公式；cached/uncached有载状态和冲量逐数组一致。静止1000细步profile13.003→3.826s，仅局部profile，非Play FPS。
- 后端新增PositionVerlet各中点/端点运动范围记录，供连续间隙检查。
- 合并命令：单核/单线程、go2 Python、USD libs环境下 `pytest -q tests/test_dh116_*.py tests/test_fixed_dual_panda_o6_moving_contact.py tests/test_fixed_dual_panda_o6_moving_contact_models.py`，**301 passed in39.96s**；日志 `/tmp/dh116-closure-preparation-regression.log`。

## 后续

先读approach最终summary，再运行独立有载闭指pilot。真实闭指、1s保持/固定材料滑移、30mm抬升、完整连续轨迹与3次重复仍OPEN。goal active，禁止以静态预筛填成功字段。
