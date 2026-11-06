# 固定相机视觉 Student 的定期高度汇报

本文记录 2026-11-06 已设置的 Codex 定期监控规则。任务在本机 Codex 中运行；本文不包含 SSH 密钥、认证信息或应用内部配置。Git pull 本身不创建或启动定时任务。监控仅检查已有训练，不启动 Isaac，不修改、暂停或重启训练。

## 训练与数据范围

当前 run/tmux 为 `dh116_gpu3_fixed1024_palm_lift_resume129_20261105T122312Z`，目录位于 `/mnt/mydisk/xk/dh116/runs/`。GPU3 上 1024 个物理环境使用固定 224x96 RGB-D；二指 `pinch_thumb_index`、三指 `tripod_thumb_index_middle` 各512。3844D 四帧感知 actor 与 371D 特权 critic 分离，使用 `support_deficit`、rollout32、micro512/infer256、Kit32、`usd_parse`。结束条件为累计完成第2048次 PPO 更新，不能按 episode 数代替。

当前段从第129次完整 PPO checkpoint 严格恢复，初始4718592 transitions、2669 episodes、successes0。第5轮的原4096视觉策略曾显式缩容为1024；历史迁移元数据的 source_iteration5 描述该旧转换，本段初始129以 `training_start.json` 为准。模型、Adam、LR、RNG和累计计数延续；物理状态及传感历史按普通 resume 从 reset 开始。

每次读取 `monitor.json`、`collection_status.json`、`training_start.json`、完整 `iterations.jsonl`，检查 `vision_student_ppo.pt` 的路径、大小和更新时间。结合指定 tmux、匹配 run 的训练进程及 `nvidia-smi -i 3` 确认状态。`monitor.json` 表示已完成的更新；`collection_status.json` 的 iteration/step/phase/time 表示正在采样或优化，未完成更新不能计入完成数。

## 每次汇报

用户明确要求定期显示二指和三指分别抬升多高。每次定时检查连接成功后发送一次简短进度及高度表，正常推进也汇报。开头写已完成 PPO/2048、最慢环境回合数和累计原生 successes；表格两行分别为二指/三指，并列出：

| 列 | 取值 |
|---|---|
| 本轮物体净抬升峰值 | `grasp_styles.{style}.net_rod_lift_max_m`，来自最新完整 PPO 行，乘1000换算毫米 |
| 物体历史最高及轮次 | 下列完整视觉链内同字段的最大有限值及所属 PPO 更新数 |
| 左掌实际上移 | `left_palm_lift_max_m` 的本轮峰值 / 历史最高值及轮次 |
| 右掌实际上移 | `right_palm_lift_max_m` 的本轮峰值 / 历史最高值及轮次 |

同时记录两组 `episodes`、`successes`、`episodes_since_launch`、`successes_since_launch`；后两项仅为本进程启动后的新增量。每组左右掌 `*_valid_envs` 应各512；`*_max_env_id` 是全局环境编号。可补充 `*_last_mean_m`，它是每环境最后有效样本的组均值，不必对应同一时刻。缺失值写“未记录”，负数保留并标作下移。

左右掌位移是实测 body-link 世界Z减去该环境本回合物理 reset 后的实测Z，峰值覆盖本轮存活物理子步。不能以目标/命令、名义初始位姿、物体整体高度或二指/三指分组代替左右手。左右峰值可以来自不同环境和时刻，不表示同一双手轨迹。启动期 `env_*/latest_episode.json` 不能作为当前训练读数。

## 历史边界与成功判据

历史仅纳入以下三段完整 JSONL 行，忽略未写完的尾行、非有限值和超过上限的记录：

1. `dh116_gpu3_vision4096_all_kit32_mixed2048_20261105T094038Z`，截至第5次更新。
2. `dh116_gpu3_fixed1024_224x96_parse_mixed2048_20261105T104905Z`，截至第129次更新。
3. 当前 `dh116_gpu3_fixed1024_palm_lift_resume129_20261105T122312Z`，截至最新已完成更新。

手掌字段仅在当前段第130轮起存在。更早的手掌历史保持未记录，不能补零、从命令推算或混入其他容量测试。旧371D特权 actor、教师示范、离线与确定性测试不属于本视觉策略的原生成功。

物体净lift排除支架预抬60mm；`exploration/max_actual_lift_m` 基准不同，不能代替。达到30mm只满足高度门槛，成功以本训练原生 PPO successes 为准，沿用净升30mm、连续保持5s及固定锚滑移不超过0.5mm的完整判据。首次出现成功必须在完整迭代日志中核实最早更新及对应抓法。

单次 SSH 失败保持安静，下一次再确认；连续两次失败报告上次已知数据和轮次。明确失败/退出、需要操作、活跃进度约30分钟无更新时突出说明。同一已报且未变化的故障不重复提醒。确认达到2048预算或明确终止后报告最终状态并删除该已完成监控，监控不自动重启训练。

## 本次只读验证

2026-11-06 03:21:14 UTC 的快照为第1328轮，44007424 transitions、34985 episodes、最慢环境27、successes0。`collection_status.json` 正在第1329轮采样，训练 PID150108、tmux和GPU3均匹配；checkpoint与完整日志更新时间约43秒，采样状态更新时间约0.35秒。完整链读取5+124+1199=1328行，没有不完整行。两种抓法左右掌均512有效环境。

| 抓法 | 当轮物体净升 mm | 物体最高 mm@轮次 | 当轮左掌 mm | 左掌最高 mm@轮次 | 当轮右掌 mm | 右掌最高 mm@轮次 |
|---|---:|---|---:|---|---:|---|
| 二指 | 4.524309 | 26.163260@269 | 0.122219 | 17.189264@635 | 0.057757 | 13.167381@964 |
| 三指 | 2.919395 | 31.813391@482 | 0.616342 | 17.587364@707 | 9.403050 | 22.348583@293 |

本次只做源码 SHA256 对齐、现有完整日志与进程核对、字段/单位复算和发布文档检查。上一轮实现的1429项CPU回归及原生恢复证据见[掌心诊断验证报告](superpowers/reports/2026-11-05-palm-lift-telemetry-validation.md)，不将其当作本次新测试。当前训练代码已在前一提交 `7d195b9d8300a3bff4e13b63e448f13bc0f1ca89` 发布，本次补充最新进度与定期汇报规则。下一步继续训练至2048完整更新并按原生成功判据评估学习效果。
本次发布执行以下 Git 检查，结果为退出码0；新增说明的相对链接均解析到已有仓库文件：

```powershell
git -c core.autocrlf=false -c core.whitespace=cr-at-eol diff --cached --check
git -c core.autocrlf=false diff --cached --name-only
```

通过现有 SSH 通道实际执行的只读字段检查如下，结果为 `fields: PASS`。它只读取当前训练文件，不导入 Isaac 或启动第二个训练实例：

```python
import json, math, pathlib
run = pathlib.Path('/mnt/mydisk/xk/dh116/runs/dh116_gpu3_fixed1024_palm_lift_resume129_20261105T122312Z')
m = json.loads((run / 'monitor.json').read_text())
assert (m['num_envs'], m['actor_observation_size'], m['critic_observation_size']) == (1024, 3844, 371)
assert sum(g['episodes'] for g in m['grasp_styles'].values()) == m['episodes']
assert sum(g['successes'] for g in m['grasp_styles'].values()) == m['successes']
for label in ('pinch_thumb_index', 'tripod_thumb_index_middle'):
    g = m['grasp_styles'][label]
    assert g['num_envs'] == 512
    assert g['left_palm_lift_valid_envs'] == g['right_palm_lift_valid_envs'] == 512
    for k in ('net_rod_lift_max_m', 'left_palm_lift_max_m', 'right_palm_lift_max_m'):
        assert math.isfinite(g[k])
assert (run / 'vision_student_ppo.pt').stat().st_size > 0
```

运行目录核对覆盖699个文本源文件及32个ASCII PLY资产，共731个文件。699个源文件的服务器/本地SHA256全部一致；32个服务器PLY的SHA256和字节数全部等于Git HEAD blob。Windows工作树PLY仅存在CRLF换行表示差异，无需新增资产提交。上述字段检查执行时训练已推进至第1333轮，原生successes仍0，结果PASS。
