# Stage 7 — descriptor 原生失败与 USD ABI 修复

2026-11-10；父提交 `9f517b01843d5442b8af0c260c7fc116904afa53`。本次只发布一个明确ABI修补及其测试，不改变原控制、物理、材料、执行器或模型合同。

## 原生事实与根因

冻结source12的 `descriptor1_v01` 实际执行1个控制步、8个480Hz物理帧；8环境共64条descriptor均为structure/collection NA，provider错误计数为0。原Teacher自然退出，全部回合截断、成功0。本次不是FK/J原生验证通过。

封存后先核对退出、原输入/源码、SHA、CRC与Core/Model时钟，再读取失败证据；闭合完整性通过。实际属性异常为 `Property.GetPropertyStack(Attribute) did not match C++ signature ... UsdTimeCode time`。首个env0的左右手各24条joint进入inactive，internal composed joints均为0；其中各自原本禁用的root_joint也被保留，不能将全部inactive都归因于ABI。该env两手各23个link的实际instance-proxy检查均为false；没有依据修改遍历或放宽jointEnabled判据。

只在 `ComposedUSDStageFacade.capture` 中lazy取得一次 `Usd.TimeCode.Default()`，传给Attribute和Relationship的 `GetPropertyStack(default_time)`。保留原 `Get()`、graph/单位/关节/来源判据、raw/DTO、NA与失败partial。构造函数不读USD；默认关闭。原失败归档及source12/15不改。

## 实际CPU命令与结果

针对修补的本机独立检查：

```powershell
$env:PYTHONPATH='src'
& 'C:/Users/xk/AppData/Local/uv/cache/archive-v0/ZwbokMniIH5FktEP/Scripts/python.exe' -B -m pytest tests/test_bimanual_mpc_model_descriptor.py tests/test_bimanual_mpc_model_descriptor_integration.py tests/test_bimanual_mpc_mounted_kinematics.py -q -p no:cacheprovider
```

实际84 passed、16.19秒、exit0；另由独立审查进程分别运行descriptor52项与integration/FK32项，均通过。新增必需TimeCode fake在修前重现0/22内部关节，修后恢复22；Default取得失败不补数字time，stack异常仍NA，两类Property共享同一个Default对象。

独立source16完整CPU胶囊的实际入口：

```sh
/mnt/mydisk/xk/envs/dh116/bin/python -B /mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/stage7_descriptor_abi_cpu16/stage7_descriptor_abi_cpu16.py --execute --helper-sha e36ea260a0e79ed53fe9979e13cb5a152b93dc342dc4908d0c397b32b539917e --overlay-sha 83096e6847e4df41fd5abcc46ceb0a0bb5257a1a7649c39f27d33afe60314214 --archive-sha 7c925de1d4b7ca6049c76fb092c5ff0050ca9a032ccf68a2ac1b67b7dd3b5514
```

实际55个文件：1364 passed、3 skipped、238 subtests，pytest292.93秒、入口296.794252秒、exit0。CUDA不可用的3项跳过不算原生验证。此独立1522文件胶囊也包含尚未发布/接入的H8、接触和载荷模块；此总数不是本次新增测试数或这些模块的原生资格。实际pytest完整argv、环境和JUnit保存在下列本地原件，原始日志不上传Git。CUDA为空，Torch/OMP/MKL均2线程；源码、原控制器和父source15前后字节不变。

## 身份与可复核原件

以下路径均相对本机仓库 `C:/Users/xk/Documents/project/dh116-publish-11-5`；原生文件位于服务器 `/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/` 下对应独立目录。

| 原件 | SHA256 |
|---|---|
| `src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_model_descriptor.py` | `f6a30a9b538da2268db94da7c87ea200fbe2eecda6de47618323f49f5e8c3273` |
| `tests/test_bimanual_mpc_model_descriptor.py` | `d98dae9af33c99fb32d1f5388b4333101dc9de50595239420db735c6b4b6ba23` |
| `artifacts/stage7_descriptor1_outcome12.json` | `509a4613ea22bfcbc7c5167a3f732809668c268bcec3901ec05622bb85364efe` |
| `artifacts/stage7_descriptor1_analysis12.json` | `890e3f0dd76837a061aae8ec120bd282599d732a5a438dc995575f981a496bf0` |
| `artifacts/stage7_descriptor1_joint_classification12.json` | `09d2d465885f0ca13dd1a620c89a8baead4b40d8317a51a92d32c38f5ea02c99` |
| `artifacts/stage7_descriptor_abi_source_manifest16.json` | `e4347e6b5f0b4751b172da9c2260fb18f557617dee843623ca0f821fd728bb2b` |
| `artifacts/stage7_descriptor_abi_cpu_validation16.json` | `b8283bf86ee51d0da1f074a6bea15189354a14c3618b5b5e6d92732fc6a472f0` |

完整CPU原件另有 `artifacts/stage7_descriptor_abi_cpu_command16.json`、`stage7_descriptor_abi_cpu16.log/xml`。原生model raw SHA为 `a564cb0131893fcf7a63f6feb663a9e624a24006e0ab8ec005ca7d283b618e52`，保持只读。失败后的current-geometry分析也保留为失败，不补造FK误差。

## 尚未取得的证据与下一步

修复版 `stage7_descriptor_abi_cpu16/descriptor1_v02` 仅准备1-control计划，command SHA `24bceff889a1eaa27207eda1c7c5884e0d06ca2e9ffdee760af558cda0a6c727`，Native执行次数0。必须等唯一固定腕双侧恢复实例自然结束，再显式执行、封存后审和独立FK/J比较；不能自动排队重试。

正在运行的固定腕B/both使用独立source15及原压力控制/逐指预算，并未接入新H8/contact wrapper。原Teacher仍默认，二指已有4/4回归证据与三指0/4历史均保留；本次没有新Teacher或Student成功。3844D Actor、371D Critic、24D Tanh、30mm净抬升/5秒/0.5mm不变，B3/PPO和新专家池仍受原生资格门禁，PR保持Draft。
