# Tripod精化指腹去除未使用的粗间隙计算
基线69ba0a488102715c338bd72d1ec5b72f19ebe9b9。修复既有Teacher控制链的已测重复计算，不新增算法模块或物理开关。

## 根因和最小修改
shared check_supported_shape在注册连续精化分支先做separation(world,*tube)，随后无条件以完整节点rod_envelope_clearance.lower_m覆盖结果。
现有API已支持tube=None；仅PolicyOwnedTripodGeometry._evaluate的target分支传None，其余部位仍传原archived_tube。
共享脚本、连续证书、支架与完整节点检查不改；.00015m上限、finite、passed及非target .00025m间隙原样保留。
不全局修改共享函数，避免改变其他调用对malformed tube的原有错误行为。没有删掉实际来源验证、重采样或碰撞guard。

## 失败测试与CPU回归
复用原test_tripod_uses_three_distinct_pad_sources_and_two_hand_means场景，spy调用真实guard/separation/certificate；没有新增fixture或测试专用生产接口。
旧代码该场景有24次随后被覆盖的粗算，新断言先失败（24.54s）。
修改后三文件35通过0失败（133.01s），覆盖三指SDF缺失、wrench/body/support来源拒绝，以及连续包络的真实穿透、全节点历史、支架与上限等原用例。
独立复核指出总证书数相等不等于逐次证明，现已在每次refined guard内检查真实证书增加，再复验primary 1通过26.65s（与35项重叠，不累加）。
更新Registry后另外60通过14.32s，与前组三文件不重叠，共95通过、0失败。

## 真实历史数据等价与计时
对原v9的64个完整物理geometry输入，旧69ba0a48算法reference与当前算法完整packet一致，失败原因、raw SHA、间隙、精化证书和所有资格flags均保留。
比较双方使用当前source provenance；只在期望副本更新sources.tripod.implementation_sha256为当前代码SHA，原JSONL和SHA不变。
这不是把旧数据当作新原生采集，也没有给source/force/grasp晋级。
在原control_source_scope下三轮交替完整geometry.validate，8样本中位0.808820971s ->0.749788013s，节省0.059032958s，1.07873倍。
不包含control scope建立耗时，不把全部separation热点都当可消除项；这项CPU收益尚未证明整轮原生能在180s内通过。
benchmark SHA 6ed96d25fb8dde63bbc39949656ea9abca7cb8ae1f65a42851f4df1503481f35。

## 审查、保护和下一项
独立只读规范/质量核对通过，逐证书断言建议已落实；Antigravity无额度，没有冒称新回复。
本包0次Isaac/0PPO，原v9仍为已消耗的失败attempt，失败Checkpoint不恢复、不入专家。
528保护文件、3旧Checkpoint、原Reset与固定双RGB不变。R2/ForceProxy关闭。
下一项基于已保存线程测量，准备独立有限CPU资源配方；原预算/动作/物理/成功标准不变，不重跑v9。
