# 左 Pinch 234 步原生账本独立复审（2026-11-13）

基线64bbe668f4bce20829e00c5a3e3e1b6c2c01e695。原生预算已经消耗；本包新物理步0、新PPO更新0，没有重新启动Isaac。

## 根因、代码和边界

新增 scripts/audit_dh116_pinch_segment_postexit_v2.py：
- 原旧验证器在函数内导入margin operation，导致新segment结果version被拒绝。新validate_artifacts保留全部旧检查，只绑定实际segment operation和原sdk_clock_sample；测试用AST核对全部验证逻辑。
- 独立Worktree重放另发现source路径差异：原native参数包指向integration配置，字节SHA与Worktree相同。load_locked分别核验两处配置与冻结CONFIG_SHA，再只在审计期间绑定实际执行ROOT。没有删去参数来源比对。
- 核验原失败receipt、全部原artifact、GDB日志、outer退出、recipe、scheduler代码和worker指纹。修改任何源配置或原账本都拒绝。
- reaudit只读原账本，在独立JSON保存结果；原failed receipt不修改、不标记passed。新的审计证据可证明该限定执行范围，不授予Teacher/抓取资格。
- 未改变原运行脚本、控制器、物理、关节、SDF、动作、配置或历史模型。

## 原生实际结果（历史运行，非本包新增）

输出 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-pinch-segment-contact-20261113-41heHL/native-runs/flex.f7-pinch-segment-contact-native/attempt-oqk81uft/native 。

234总步=2初始化+232测量；29完整60Hz控制窗口、480Hz物理、每步12杆子步；1环境、0PPO，134.936361秒/180秒。outer=0、GDB inferior=0、signal=null、app.close返回。没有重试或强制退出。真实执行journal、SDK时钟、source/cooked几何、原关节和wrist保护、native参数、director/Reset来源均逐条重放。

| 实际接触 | 左拇指 | 左食指 | 右手 |
|---|---:|---:|---:|
| 正法向载荷pair条数 | 526 | 495 | 0 |
| 首次SDK步/solver子步 | 191/2 | 193/9 | 未建立 |
| 最大法向载荷N | 0.269332 | 0.287659 | 未建立 |

两指同时接触495个连续solver子步，85.9375ms；这些不是有效双手抓取。桌面仍有110047条杆接触pair，不能证明脱离外部支撑。双侧最小source桌面净空分别2.117664/2.116345mm，原门槛2.1mm。19行使用完整3D逐杆段证明，最小下界+0.491006mm；未改5mm杆半径和2µm数值裕量。

初始四指source/cooked最低桌面净空mm：
finger14 2.992700/3.086673，finger23 2.833199/3.172788，finger43 55.877917/56.030032，finger53 73.364324/73.510211，穿透均0。
Reset及动作与上一margin尝试相同，只更换独立几何证明方法。前次SDK216退出1的原失败保留。

自然退出0不等于所有引用已提前释放：SimulationContext构造frame仍被观察到，手/视图/solver等弱引用已消失和callback释放单独记录。手-静态桌面Tensor接触仍NA。不能把此结果泛化为相机C5生命周期全部解决。

## 不可变身份

- 原失败receipt SHA 97f83e1e001ea28cfa08ebcf4017d717f7adcc11aad38402280e780f48311746
- 原physical.jsonl SHA 170b9bc8090eeb3f4291ce4102335eabbcb1d49892e333d1aa5b4ac4d9585fdf
- outer_exit SHA e209c69eea0f2c19b2ae75ede964cdcaa658c52c06b6cdaab6fdccaa913055dc
- native.log SHA bddb0ee7341006c68d9af60ce2da181bef3ace605b8a26019f966144ee431c19
- recipe SHA ab5e73125f1bae212b8a4865a362dd1dad208a3535f58deab2e84dcf04802c80
- scheduler代码/配置 SHA 186855563b67e02e6998797fb57d416da79d35bfa70311b96815ff9f844f76df
- worker SHA 61dd815e39e6ea293b1f7dd51333db1843386b565aa119b4cb4f57162d071ed1
- 新disk-only结果 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/approved-pinch-segment-contact-20261113-41heHL/postexit-v2-audit.json
- 新结果 SHA 57cbc1b6b97eb594368e4702746241c3be3a3b0c562d99293ab2b470b30706cf

## CPU验证与失败保留

先RED 6失败（缺少模块），随后26通过/1失败定位跨Worktree路径差异；原日志不覆盖。
最终28 passed、0 failed、46.98秒：
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_pinch_segment_postexit_repair.py tests/test_dh116_pinch_contact_failure_audit.py tests/test_dh116_polyline_support_certificate.py
```
日志 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/segment-postexit-full-v2.log，SHA b6c76b8626ff7c413d3c451bf43e66c073a373655049337acd1fac8c74a03356。
重放命令 scripts/python.sh scripts/audit_dh116_pinch_segment_postexit_v2.py --output <新的文件>。拒绝覆盖结果。

## 剩余与后继

原flex.f7-pinch-segment-contact-native仍failed，限定的实际执行通过由独立复审证据表达，后继必须显式核对双SHA，不能伪造父passed回执。
Teacher0/3：右手/Tripod/Power各自接触路径待完成，66/371同钟状态、R0稳定来源与连续材料滑移不齐。0.4875秒无法满足5秒Hold；R2关闭。
WireSeg本包新增0训练步，历史128seg/128depth不变；视觉问题不阻断无相机准备。
接下来独立检查右Pinch实际几何与接触缺口及R0数据来源，不重跑这两个已消耗配方，不扩大原生预算。528保护文件、两Checkpoint由发布器再次检查；PR2保持Draft。