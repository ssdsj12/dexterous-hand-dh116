# 柔性杆参数来源读回：原生 getter、配置与未知量分离

基线 `9b2567745f16d062d90f6449727d4f7cf5bd8a4b`。新增独立 `tabletop_flexible_parameter_readback.capture_parameters`，直接使用现有FreeRodRuntime、DualHandRodRuntime与已锁定RopeBatch接口。没有修改旧构造器、场景、动作、奖励、材料、关节、模型或Checkpoint；没有复制第三套物理配置。

本包为**CPU API doubles验证通过、原生入口尚未接入**，不应把上一包的材料方向原生通过移植为本参数getter验证。PPO0更新，新增原生0步。

## 已实现调用路径

1. 核对实际runtime/base/loaded backend类型、二进制与原有构造器源码SHA；运行时配置必须完整等于冻结候选文件。
2. 在同一实际SDK/solver边界读取，限定显式cuda:index和原默认stream；前后同步，校验clock、solver error、配置、constructor nodes/rest未变化。
3. 调用真实API `RopeBatch.inv_masses()` 与 damping/drag/drag_quad/drag_quad_t/num_itr getter；节点质量必须为正且符合原kg/m构造。getter结果与原构造时缓存对照，缓存不代替当前读取。
4. 返回独立host拷贝与每字段unit/frame/provenance/valid/uncertainty。整包identity绑定run/env/episode/reset/physics/rod/timestamp；不存在的getter不虚构。
5. 正常与异常路径都显式清空runtime/base/rope/sim/backend引用，不引入另一个native owner、不增加gc.collect、不推进物理或reset。

`inv_masses`原C++实现会download求解器数据；本函数在现有独占读阶段显式同步后使用它，返回值复制完毕后再同步并核对时钟。CPU验证不能替代此新增getter组合的原生生命周期验证。

## 字段来源

| 字段 | 来源/限制 |
|---|---|
| mass_kg、各节点inverse mass、无固定节点 | 实际inv_masses getter及原lumped-mass模型核验；不使用配置总质量代替 |
| node_count、fixed_root_orientation | 实际RopeBatch属性 |
| damping、drag、drag_quad、drag_quad_t、num_itr | 实际getter；drag/damping量纲仍标solver_coefficient_units_unqualified |
| material rest length | 原float32 constructor节点差分；不是native rest getter |
| radius、rod gravity、linear density、EA、EI、substeps | 来源锁定配置，明确not_native_readback；不当成测量值 |
| global friction | 原set_contact全局配置0.5/0.4 |
| 十个左右pad mesh friction | 原add_mesh参数max(杆配置,原hand material)=1.0/0.8；仅配置来源 |
| table mesh friction | 原table add_mesh配置0.5/0.4 |
| Young/shear modulus、kg/m³体密度 | NA；不从独立EA/EI或未校准体积材料假设硬凑 |

配置值与native getter值均不是硬件材料测量。所有hardware_measured=false，uncertainty=None且uncertainty_calibrated=false；没有把unknown=1当校准误差。全局摩擦与逐mesh参数分别编码，避免将0.5错误灌入手指有效材料字段。这里不声称实际接触律已通过标定。

## 失败证据与修复

先写测试、运行缺模块的失败，保留 `parameter-readback-red-XCaIuq`。初版22项通过后，增加来源/中途变化回归，得到**2失败、23通过**（`parameter-readback-source-red-ydUSCr`）：

- 全局摩擦的泛称字段容易误当手指pair材料；改为global_friction_*并明确逐mesh配置来源。
- getter期间CUDA stream发生变化未被拒绝；增加读取后默认stream检查，不仅检查开始时。

修复后本模块25项+原runtime/material director回归共**72 passed、0失败、0跳过，6.16秒**。负例覆盖配置/二进制/类型/reset/clock/boolean clock/SDK/stream/质量/固定节点/shape/NaN/getter不匹配/solver error/根固定/rest，以及getter期间配置或clock变化与异常frame引用释放。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q -p no:cacheprovider tests/test_flexible_parameter_readback.py tests/test_deformable_grasp_runtime.py tests/test_dh116_material_director_readback.py
```

私有证据：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/parameter-readback-release-2kDLw3`。原生数据与模型没有上传。528源码+2旧Checkpoint核验通过。

## 后继与准入

下一包将本读回作为独立旁路接入实际reset与受控endpoint，保存getter快照、同钟身份、错误前缀和资源释放证据，再冻结新的有限原生配方。不能复用已耗尽的18步recipe执行预算。本模块当前的reward_radius_sample=None、native_qualification=false、teacher_admission=false均保留；R0还需要明确适用性、approach基准、stable和固定材料接触来源。Pinch/Tripod/Power PPO就绪仍0/3。

| 文件 | SHA256 |
|---|---|
| src/go2_pvcnn/control/dh116/tabletop_flexible_parameter_readback.py | 9c96056dd57c825c11b90cbbfd2f0f813e61aa1f0fd91b06cf31604e74ea4093 |
| tests/test_flexible_parameter_readback.py | 97ed505d5e01352a6fdb6c1afadd36dd938845da477b073403a8a707852f6540 |
