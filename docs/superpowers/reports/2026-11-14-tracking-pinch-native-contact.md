# 2026-11-14 Tracking-aware Pinch 原生接触结果

## 实际执行

冻结源码HEAD `81b93cb8acc8bbbd05f29235252b22007d0ec83f`。新配方只运行一次，GPU3、1环境、0PPO更新、234总SDK步（2初始化+232测量），原生进程120.192828秒，外层和GDB inferior自然退出0，signal=null。完整退出后账本审计通过；不是仅检查marker，也没有强制退出。原生结果提交 `66f6aaef5c43ae2f25c2b97b6ace0a4c90c69235` 经Windows发布。

原生180秒预算只覆盖原生进程，CPU重放/发布耗时另计。本批预算已消耗，禁止同配方重跑。授权依据本聊天“批准，后续自动批准”，执行前绑定本次冻结指纹；不复用旧批次授权文件。

| 身份 | SHA256 |
| --- | --- |
| 配方 | a747d3bb5102b1e77145ff12790c6b5948e715e84e935c5ab595866889a6c1e9 |
| 调度code/config | 4b900443ee3cf69ce0c387fa7489b554eaa948ce7143f4384e23e44c6263fc3e |
| 入口preflight | c3c5d66267667c31478b1964218b835a3ba66ee305e27707d398b3e767d8fc61 |
| 实际scheduler receipt | 83a8758210fd6aa35da239c32819b91b9b6967307d30f7e380c67f965381cf11 |
| physical.jsonl | d313e51b7857e78630f2117237bf5e48ecd9a05ad607ce8c3435077d592b178a |
| 完整退出后audit | f37981e40594a10402b7f99fcb7cb6b8219aa91d40fdbed42ebd72d194c8397d |
| 最终接触重放分析 | 6fea160db84893089082f619c2f0925316418cbf89bc00af08ea6306ccfe8b41 |

私有原生目录：
`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-tracking-pinch-20261114-5dfid_2s/native-runs/flex.f8-tracking-pinch-native/attempt-zgggtx6_/native`。
报告/CPU证据目录：`.../evidence/tracking-pinch-native-20261114`。原始大账本未上传。

## 真实接触结果与失败边界

env_id=0，episode_uid=`dual-pinch-4a031febc8c2410686d88233007bc3ba:one`，reset_id=1，rod_uid=rod0。232个连续测量边界均通过原保护，没有joint/mimic/几何终止。

| 指腹 | 原始solver pair | 落入原指腹区域 | 区域内pair法向力幅值范围（N） |
| --- | ---: | ---: | ---: |
| 右拇指 finger14 | 1100 | 276 | 0.268729～0.466108 |
| 右食指 finger23 | 1116 | 311 | 0.227707～0.397771 |
| 左手 | 0 | 0 | 未建立接触 |

587条pair通过原区域判定；1629条仍因outside_inner_ellipse拒绝，没有放大ellipse。相比旧候选原生0条区域内pair，本候选出现真实改善；候选初态及有界路径均有变化，这不是单因素因果结论。

右侧两指同时处于原区域的实际solver窗口276个：SDK212～234，连续0.047916667秒。依据真实1/(480×12)秒时钟逐窗口计数，没有填补缺失窗口。左侧0、双侧0。因此：
- 这不是双手对向承托资格。
- 不是5秒Hold，也未执行Lift成功验收。
- 法向力是实际DeformX仿真pair量，不是真实DH116硬件力测量；连续样本的力不可累加当作手掌载荷。
- 材料锚Slip、接触深度和完整R0资格保持未通过；Teacher/Student成功均未授予。

## 桌面与非目标几何

以下是本次全232边界最小值，最小值各有独立physics_step，不能拼接成同一轨迹。原桌面保护2.1mm不变。

| 指腹cooked hull | 左侧最小桌面净空mm@SDK | 右侧最小桌面净空mm@SDK |
| --- | ---: | ---: |
| finger14 | 3.086673@3 | 3.128120@233 |
| finger23 | 3.172788@3 | 3.269964@224 |
| finger33 | 38.728448@3 | 32.196376@234 |
| finger43 | 56.030032@3 | 46.890452@227 |
| finger53 | 73.510211@3 | 59.943479@229 |

这五个指腹的桌面穿透均0；所有source/cooked几何整体最小桌面净空：左2.117664mm@3，右2.164655mm@233。右侧非目标间隙下界仅0.005388mm@145，仍很窄，不宣称对其他初态稳健。左侧下界1.367406mm@218。

右末端实际root相对命令XYZ误差[0.000750,0.000804,0.002724]mm。右finger12实际减命令误差-0.000785359rad；其他主动关节误差和逐步身份保存在分析JSON，未用命令位置代替实际位置。

## 代码与测试

新增 `scripts/audit_dh116_tracking_pinch_contact.py`：锁定实际receipt/audit/ledger SHA；复用旧接触区域与同边界FK分析；保存每个480Hz边界、连续solver窗口、实际normal-force幅值、带发生身份的全部cooked pad净空。所有资格字段仍为false，重放新增物理步/PPO更新0。

新增 `tests/test_tracking_pinch_contact_replay.py`。先保存模块缺失和几何统计缺失的失败测试，再完成实现。相关CPU回归20通过、0失败/跳过（22.95秒），日志SHA `c2a02d37c2d38dbfd9c8737ca76067fe4ed54dcf0344ff904575b7f3f7d2ea09`。源码入口88回归通过，原生结果发布前28回归通过；分别计数，不相加当独立用例。

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest tests/test_tracking_pinch_contact_replay.py tests/test_effective_right_tracking_replay.py tests/test_deformable_pad_region.py -q
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh scripts/audit_dh116_tracking_pinch_contact.py --output /absolute/private/new-analysis.json
```

封存后首次发布事务曾因新Registry遗漏resource_class/requires_approval/retry_policy而在提交前被校验拒绝。保留原事务和失败原因后补齐同一调度合同，再验证/恢复；没有执行未受门控原生任务，也没有放宽Registry。

## 下一工作包与门槛

1. 复用本次真实杆位置和左右实际关节数据，解决左Pinch仍未到达线缆的问题；保留本次右手改进，重新做双侧独立接触/完整路径预检，不能只复制右手变换。
2. 对587条区域内pair继续按原标准核验对向载荷、拒接载荷与材料锚来源；区域命中不能代替完整有效承托。
3. 保持单Isaac、独立配方、一次预算；新候选通过CPU预检并发布后才执行新的对应有界批次。
4. Tripod/Power继续各自接触/Reset/状态动作/R0资格，不能借Pinch诊断通过获得Teacher资格。
5. 本包WireSeg新增训练步数0；没有启动Teacher PPO或Student训练。长PPO及完整5秒回合不在本次234步预算内。
