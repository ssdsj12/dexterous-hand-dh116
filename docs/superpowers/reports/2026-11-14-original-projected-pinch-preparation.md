# 原安全投影约束下的 Pinch 独立实验准备

本工作包不更改控制器、物理、材料、USD、动作幅度或关节保护。默认及历史实验保持原样。新实验只替换“请求偏移必须完全等于未投影路径”的实验目标校验，要求从实际预动作 source collision geometry 重算原投影，并逐步核对请求、执行、SDK提交和跨步连续性。

## 证据和修改

上一实验 receipt SHA `25a75775ec43e03f6c722ae395df8584ac149e38c87e18ac77aba52f6d8d745d` 在SDK176主动终止，非SIGSEGV。SDK175的真实位置和原source envelope独立重算表明，左finger13_link触发原safe_z。向上投影差0.000809032891944mm，原投影重算误差0。原关节/碰撞保护未失败，旧实验的“零投影”要求不再适用于这段动态跟踪；此前条件路径CPU通过不能转为动态路径资格。

- `tabletop_projected_pinch_geometry.py`：`validate_projected_target`核验原24D请求、实际root seed、原动作积分、原projection逐项重算与SDK目标；`validate_continuity`核验相邻实际输入采用上一子步的applied offset/angle。所有source/cooked table、SDF、穿透、关节和左右手隔离保护保留。
- `prepare_dh116_projected_pinch_native.py`：仅在显式scope安装owned evidence，嵌套scope不重复包装；异常后恢复所有绑定。`inspect_projected_closure`逐条审计实际原始输入及跨步状态。
- `verify_dh116_projected_pinch.py`、`run_dh116_projected_pinch_work_packages.py`：独立worker身份、GDB、依赖回执和外层自然退出审计。新scope marker与原owner marker同时要求。
- `deformx_projected_pinch_contact_v1.json`：复用原初始姿态/动作/真实柔性杆，不复制或改写资产。2 setup +232 measured =234 SDK步，1环境、0PPO、GPU3、180秒、每冻结指纹一次。
- 四个新测试文件分别验证目标合同、scope/预算、完整/提前终止owner、真实CPU子进程入口及post-exit顺序。

新recipe SHA：`129e0b0e3455e276733554d15c1ccbd431772dc04572e168943ed3d327109254`。
入口code/config SHA：`720b2499cbdc52c53c25bcdde920168072650db120325ae27177cdb1be52fba0`。
167个源文件与外部依赖纳入冻结；调度器总指纹另按最终配方读回绑定。

## 验证

CPU先失败：目标合同10项缺失；native路由8项缺失。首次worker回归12通过/3失败，原因是复用CLI仍通过旧模块调用函数、测试导入层级变化，以及fixture只保存一个marker。均修复后重新验证，失败日志保留。

```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python CUDA_VISIBLE_DEVICES='' DH116_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_projected_pinch_geometry.py tests/test_projected_pinch_native.py tests/test_projected_pinch_owner.py
# 20 passed, 344.57s
# log SHA c2d621d86b03e9b78dc93193d70ef74c0bf3c05dddba37928324dd030317b2ed
```

入口修正后追加47项（worker/native/geometry/旧geometry）全部通过，24.18秒。两次合计49个不同测试；集成时将运行同一完整49项集合，以readiness实际记录为准。完整owner使用合成SDK，232条物理记录与提前终止9条记录均保存并审计，**不计原生执行/成功**。

另在RTX4090 GPU3运行单组原投影CUDA/CPU重算算术对照通过；未启动Isaac、0物理步、0PPO。这只验证算术路径，不构成原生物理证明。

## 尚未获得的资格与下一步

本工作包原生运行0。旧left失败记录不改写，新配方执行前经Windows发布和SHA核对，再从用户持续批准范围绑定新指纹、检查单实例资源锁。原生必须保留全部234步（若失败则完整失败前缀）、原投影原始输入、实际cooked geometry、contact pair、wrench、GDB映射、资源释放及外层退出码。不能把合法安全投影描述成未投影路径通过，不能把raw接触当原指腹有效承托。5s Hold、30mm Lift、0.5mm Slip与Teacher训练资格均未取得。
