# Stage 6：连续视觉监督与闭环控制实施记录

基准`26f1bbcc5a0e596da55c5d89896befcb5622cda7`，分支`codex/dh116-lift-repair-20261106`。本阶段按用户已指定方案直接修改代码和训练流程。3844D Actor、371D Critic、24D具名动作、原M0/Teacher/物理与30mm/5s/0.5mm标准保持；条件命令展开不冒充物理世界模型。

## 提交1：连续数据构建器已经接入

新增`tabletop_vision_sequence_dataset.py`，并在原监督数据模块末尾增加`validate_sequence_context`和`join_sequence_context`。原v1数据及`validate_dataset`函数保持原样。新的sidecar按`run_uuid/env_id/episode_uid/tick`联结实际动作、真实previous_action、传感/控制时间和8个物理子步；缺测窗口排除，重复身份/格式/SHA错误拒绝。

16步连续窗口不跨episode/reset/terminal/来源/缺口。长episode的完整滚动窗口覆盖末尾，只有原本短episode允许有效mask padding；从长episode缺测切出的短片段不能伪装短完整示范。支持来源均衡及40%全轨迹/60%事件混合，缺类别重新归一化，每个合法窗口保持正概率。CONTACT/LIFT_ONSET/HOLD取原生事件元数据，RECOVERY需要独立资格，均不进入Actor输入。单来源明确fit_only。

实际CPU命令（服务器新可写副本`/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01/source`）：

```bash
CUDA_VISIBLE_DEVICES='' DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 \
bash scripts/python.sh -m pytest tests/test_vision_sequence_dataset.py tests/test_vision_supervision_dataset.py \
-q -p pytest_mock -p no:cacheprovider --basetemp=/tmp/stage6-data-commit1-20261109
```

结果：76 passed，5.10 s，exit0。合成CPU数据仅验证schema/窗口/采样，不计原生成功；本阶段尚无BC模型或来源纯Student成功可报告。

## 独立原生采集与固定输入

新实验根为`/mnt/mydisk/xk/dh116/runs/stage6_sequence_20261109_v01`。输入manifest从已冻结Stage5 launch的真实参数解析模型路径，并验证源c58模型SHA，不猜权重文件名。原第1024轮、c58、4698样本池、全部Stage5冻结目录继续只读。

训练前已冻结新200例manifest，seed116602、SHA `298f379c599b34c5e58c3fdecb3957912503d42b81ce62a3f3c6b6f03bae9bed`。原200例保留回归用途，新集仍是同reset族内小XY/yaw扰动，尚为authored_unverified，原生次数0，不声称来源族/材料/柔性线缆泛化。

原Teacher/MPC正在新目录采集真实pre-control上下文。每次作业均独立冻结源码、记录SHA并持有原单实例锁；任何未完成、未通过完整目标一致性或失败的数据不放行B2。PointNet激活、可微shadow、B0/B1/B2、分支与双侧恢复的后续实现和实际结果将在对应提交补入，不以本数据提交提前授予资格。
