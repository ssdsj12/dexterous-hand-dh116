# 原生参数旁路读回：独立 18 步准备

基线 `89c8c0cb7b31ed08bd7809ae53895766d9236693`。本包仅增加参数来源旁路和独立执行入口；不修改原物理、材料、动作、初态、关节保护、资源释放或历史模型。此处仅报告 CPU 验证，原生和 PPO 均为 0；后续原生结果单独记录。

## 实际实现

- `tabletop_flexible_parameter_readback.py` 抽出 owned packet 构造并新增 `validate_packet`：从冻结构造节点、实际 endpoint 身份、原始 getter、原配置和 source SHA 独立复算完整字段/单位/provenance。未知材料常数必须为 None，不许补零；配置半径不得改称 native 实测。拒绝额外字段、错误身份、伪造校准、pin/mass/getter 错误以及错误奖励准入。
- `prepare_dh116_dual_pinch_parameter_interval.py` 在原 director reader 的同一 owned scope 添加参数读回；仅在实际 bound runtime.clock 与同一个 RopeBatch 匹配时读取。每个已完成物理边界先读参数、再读原 director。只有 owned host 数据写入账本，不持有额外 native view。
- 沿用原来的 2 setup + 8 zero + 8 finger-only 共 18 SDK 步。一个 reset + 16 measurement 共 17 参数包。物理动作索引1/2/7/8固定为.1/.1/.05/.1，所有掌平移/旋转保持0。不得把18步读回称为完整episode、抓取或Hold成功。
- reset 读回失败禁止首个动作；中途失败保留当前已完成物理行及 NA/error，原 joint/clearance 守卫优先，之后不再推进。异常帧显式释放 runtime、bound clock、rope、backend、sim 和 reader 引用。
- `verify...` 只接受最新 director 实验真实 receipt SHA `11731b7d0a95edfd9573c53ccd08af238f9db47831eb37bf7b3ea0c27b2d9dd6`，并核对实际 physical/result/clock/director_reset 字节以及独立 postexit SHA。不编造父输出路径。
- `validate...` 先运行全部原 closure/director/outer exit/GDB/弱引用/时钟/几何审计，再逐包重新验证参数及 producer summary；任一失败撤销本包资格。
- 独立 wrapper 负责 worker 重入标记、单实例 GPU3 资源锁、原生一次预算和真实父 receipt 解析。不能用普通脚本绕过。

## 来源与未取得的资格

inv_masses、节点数、自由根、drag/damping/num_itr 为实际 native getter；质量由实际 inverse mass 计算。radius、EA、EI、gravity 和全局/逐mesh摩擦只有锁定配置来源，**不是 native getter 或硬件标定**。全局摩擦 .5/.4 与手指接触对 1/.8 分开，桌面 .5/.4。体密度、Young/shear modulus、材料误差标定保持未知。

此包不会把 configuration 转成实测，不产生 R0 半径样本或 Teacher 资格；真实双侧接触、固定材料锚滑移、稳定保持及三抓法 PPO 仍待解决。

## 失败和回归证据

先写失败测试：
- parameter-native-red-SywcJ3：23 项因缺少 disk validator 失败。
- parameter-owner-red-7Tf8MH：两个新入口模块缺失的 collection errors。
- parameter-owner-test-TCWLPE：3 failed / 11 passed；旧 CPU Rope 替身 inverse mass 返回 [41]，与真实 installed getter [1,41] 不符。修正替身而未放宽生产校验。
- parameter-worker-test-cF3DGR：2 failed / 15 passed；新测试误指向更早父 receipt/audit，修正目标及 SHA 注入对象，不更改真实父资格规则。
- 新 recipe 曾错误将保护 assets 加入 publishable source_files，生成器已修复，资产继续沿原 source/protection SHA 门，不修改资产。

最终日志、数量、代码 SHA 和冻结预算见下列机器生成审计附录。CPU SDK doubles 不计为 Isaac 实验。

## 下一步

经 Windows 导入同一提交发布、核对远端后，依本轮“批准，后续自动批准”执行本新冻结配方一次。1 env / GPU3 / 18 total SDK steps / 180秒 / 0 PPO；不复用已消耗 director 或 closure 配方。结束必须检查 outer=0、inferior=0、signal=null、正常 app.close 和 GDB maps；失败只保存证据，不重跑。随后继续真实接触、R0 和 Pinch/Tripod/Power 环境准入。

## 审计附录

```json
{
  "cpu_passed": 164,
  "cpu_failed": 0,
  "cpu_seconds": 276.44,
  "log": "/home/xk/.local/share/dh116-autonomous-3c80166/evidence/parameter-native-release-Te1IWv/pytest.log",
  "log_sha256": "d9f19efc7fcfaa57127a7e95f46bdd0bc719acc0641f35c351314b90453c12a6",
  "command_spec_sha256": "2ee09c8653d90a76e836a46ff90f337c141f248425ede1a22d2dd63fc81be819",
  "code_config_sha256": "7d7887f3739f97a593ea52fed72c5d052aabcf31c1f66fcfd246862986525f4c",
  "worker_code_config_sha256": "0387fe143472ddb96a62dbe0eca4bb96e94c38a5f19cd79beddcf4281b6bc89d",
  "budget": {
    "environments": 1,
    "gpu_index": 3,
    "min_disk_free_bytes": 1073741824,
    "min_gpu_free_mib": 4096,
    "physics_steps": 18,
    "timeout_seconds": 180,
    "updates": 0
  },
  "source_sha256": {
    "configs/dh116_execution/deformx_dual_pinch_parameter_readback_v1.json": "2ee09c8653d90a76e836a46ff90f337c141f248425ede1a22d2dd63fc81be819",
    "scripts/prepare_dh116_dual_pinch_parameter_interval.py": "5bd0d1ba541f9d8ec871461e96dfa1a46ee7e57e0b89141da3a28e72a826499f",
    "scripts/run_dh116_dual_pinch_parameter_work_packages.py": "2e3ad9ac7afe3c8a70ec4aeeb180d3a49fdeb3a2d76f9e21cdf65715e2dbdcdf",
    "scripts/validate_dh116_dual_pinch_parameter_interval.py": "8493bfb2a660f506b40ac768458b43cf5e402a3742abd18c32c58ca29a35e51b",
    "scripts/verify_dh116_dual_pinch_parameter_interval.py": "4fbfd8c223117b54ce62dde2eae339ed3ef549bb5b9f782464a29edc711e8882",
    "src/go2_pvcnn/control/dh116/tabletop_flexible_parameter_readback.py": "53c46f2c99e8c2d0ed33fd2dc305161f990cbf7e66e16f35974473be748b6912",
    "tests/test_dh116_dual_pinch_parameter_interval.py": "2f9cc3341aeff799635b4d666be3d0bf6510d0543870fc8516d751aee311c3da",
    "tests/test_dh116_dual_pinch_parameter_postexit.py": "1fbb661f2bb43827280205f263d73d5ead0ab7268ebd5d43bb091b09df75884a",
    "tests/test_dh116_dual_pinch_parameter_worker.py": "cd47c5442147b36329f091cdb59f557960ca84c59abc041f3ec9b54dfda7e9bf",
    "tests/test_dh116_parameter_packet_validation.py": "2f018f30a463585f8b44f33e5a8edb548e73fdc66398707b4eab57d15f519f00"
  },
  "protected_files_and_checkpoints": "verified unchanged"
}
```
