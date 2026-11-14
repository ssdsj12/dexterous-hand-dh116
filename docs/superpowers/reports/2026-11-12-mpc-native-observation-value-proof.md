# MPC 原生旁路的 371D 观测逐值核验

在真实 `wp3_mpc_privileged_sync32_v02` 的 64 行 371D／24D 同钟归档上复算。只读原始数据，CPU 执行。实际动作、episode、env、策略步、相机侧 episode、SDK 时钟、首个 480 Hz 反馈和终止标识均绑定。所有候选字段按原 371D 归一化和 float32 输出逐值比较。缺证、语义不一致和数值不一致的位保持 false。

## 可由旁路证明的字段

`tabletop_mpc_observation_validity.audit_observation_mask` 使用 `SequenceContextCapture` 保存的本步首子步 `raw_measured_packets[0]`、`physics_context`、`controller_start` 及该回合已执行的上一步动作。派生指腹几何复用 `tabletop_dexhands_feedback.geometry_features` 的几何切片。

| 原 371D 索引 | 数值来源 |
|---|---|
| 0–35、55 | 原生主动关节位置／速度、上一关节命令、原任务 offset |
| 60–73、78–87、100–120 | 桌面净空、物体相对中心／根四元数／根速度、实测主从关节残差、掌偏移、驱动刚度命令、圆柱底部和旧指腹 FK 相对位置 |
| 153–184、191–238 | 实测腕位姿及 reset root seed、腕速度、腕安装关节角、已提交腕角／掌偏移、当前关节命令、同回合上一条实际 24D issued 动作 |
| 239–250、321–350 | 旧指腹与圆柱派生几何、全部指腹参考点 FK 相对位置 |

所有 64 行均有 **204 / 371** 个字段逐值精确匹配，包含此前已核验的 24 个主动关节字段。结果仅证明 Isaac 任务同钟观测值和来源一致性。驱动刚度等命令字段仍标为命令，SDK 姿态仍按现有模拟器根坐标语义。

## 仍缺的字段

| 原 371D 索引 | 数量 | 缺口 |
|---|---:|---|
| 36–47、52–54、56–59、74–77、88–99、121–131、144–146、185–190 | 55 | Reset 参考、计时器、闭合及抬升控制器状态、抓取基准、力控制内部历史和请求旋转需要独立命名读回或原配置证据。历史数值相等的 `initial.command` 和实际提交腕角仅作诊断参考。 |
| 48–51、132–143、147–152、251–320 | 92 | 接触对、对象侧方向、载荷和原生测力资格仍待验证。可复算的几何候选、接触分类和力派生值保持当前准入状态。 |
| 351–370 | 20 | 原成功窗口、压力标志和固定锚须按同 episode 的上一完整 480 Hz 区间独立复算。原控制步摘要可核对其中 9 个字段的诊断值，不能授予力学资格。 |

现有采集的 371D 原始值保留。新证据独立保存 `privileged_valid` 矩阵和源文件 SHA，`qualified_for_bc` 与 `force_contact_qualification` 保持 false。未运行 Teacher 训练或新增原生实验。

## 证据和复现

实际输入 SHA256：

- 371D 同钟归档 `5f9a419049121a61caa451a03878c6d5be673f94b2fa98b28bf6bca56d0fdfed`
- 原生 sequence context `28d08b29af9379058fda45f2d0704fa623c542a06a19c9aca9ece453341b161f`

新证明产物 SHA256 `0744cae536bd2f557674cc45c2e9ff8f3700b57b6bde24abbe87c541dbf067bd`，摘要 SHA256 `19ca4661af57a46528a0653d876708e974376b1a7e488181154ab2a26554e717`。原始证据与结果仅保存于 Ubuntu。第一次计算审计完整保留各块的比较数量与误差，输出 SHA256 `109029ed5a55205dc42580a86ac9f61fec165e59fa1c8b374d3c58fbd10b5d97`，只供诊断参考。

```bash
export PYTHONPATH="$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
python=/mnt/mydisk/xk/envs/dh116/bin/python
"$python" scripts/audit_dh116_mpc_observation_validity.py \
  --archive "$archive" --archive-sha256 "$archive_sha" \
  --sequence-sidecar "$sequence" --sidecar-sha256 "$sequence_sha" \
  --output "$new_proof"
"$python" -m pytest -q -p no:cacheprovider tests/test_mpc_observation_validity.py \
  tests/test_mpc_privileged_capture.py tests/test_dh116_privileged_action_audit.py
```

Ubuntu CPU 回归 37 项通过，0 失败。新增 7 项用例覆盖字段误差、相同数值但缺少源、episode 与时钟错位、动作被替换、原生读回缺失、前步动作失配、源 SHA 不匹配和禁止覆盖证明输出。原资产、控制、动作合同、528 个保护文件和 2 个历史 Checkpoint 保持原值。

## 腕部上一完整区间

追加核对原 367–370。两项 wrist Z request 来自同一 episode 上一条实际 issued 24D 动作的索引 14、17。两项 wrist Z increment 来自该区间第一个 pre-control 实测 wrist pose 与第 8 个原生 post-physics wrist pose 的差。每个 480 Hz 子步都须存在，SDK 时间、后端读回时钟和 live 标志必须对应，结束腕位姿还须等于下一策略步的实测起点。缺步、时钟异常、inactive、terminal 或 reset 边界阻止区间入证。初态反馈缓冲区按原合同初始化为零。初态没有上一物理区间，四个位保持 false。

在同一份 v02 原生数据，62 个非 reset 行的四个值全部逐值匹配。整批仍有 204 个字段每行都通过，新增 4 个字段仅在完整物理区间可证，其余 163 个字段没有独立资格。新证明 SHA256 为 `46aad98236260dc62ca21c99b95aa88e8b46109b4f8dbab9de97d20953e40edb`，摘要 SHA256 为 `10b88ea5292a4101aa9d66bbc19f6eeb81bd6be913505b9fd57e948a9e299235`。同一条 CPU 回归命令通过 43 项。未变更当前原生 T5 采集或力学资格。
