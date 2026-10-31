# Teacher → student 迁移诊断与11-1修复

本记录区分教师、离线拟合、确定性student原生执行和随机PPO训练；各类结果不互相代替。用户授权自主修复并发布`11-1`，此前未上传的10-31多指、教师、student和MPC源码一并保留。未改变物体重力、碰撞材料、执行器上限或真实成功阈值。

## 保存的训练

服务器：`a21702624634466304168937:/root/hand`；环境`/opt/conda/envs/hand`。原AR训练在iteration475保存后有意中断，累计53533个新episode，每环境最低8，native successes=0。原始4096计数不清零，目标仍为每个环境累计2048个新episode，跨二指/三指块延续，按最低回合128、256…2048切换，不设2048次update上限。

不可变断点：`runs/deployment/new_host_20261031_a21702624634466304168937/before_student_fix_4096.pt`；SHA256 `bdd1b06301891ef3daebe0e420fc948df9d5e18f11ab816334cc5c7147463b75`。没有启动第二个Isaac。

## 同一新服务器的原生对照

根目录：`runs/student_transfer_fix_20261031_as`。

| 对照 | 原生结果 | 含义 |
|---|---|---|
| `reference_pinch`原二指teacher回放 | 2/2真实success，1175控制步，无recovery | 示范和新服物理场景仍可成功；不是student成功 |
| `bc_baseline_tripod`原BC均值 | 2环境、1800步，0成功；最大净升11.109/15.569mm | 低训练集MSE没有转化为闭环抓取 |
| `holding_v1_pinch`加强保持拟合BC | 2环境、792步，0成功；最大净升0.081/3.169mm | 单独提高保持样本权重不够，普通碰撞和初期偏离仍在 |
| `ppo_mean_tripod`原475轮actor去噪 | 16环境、776步，各结束至少一回合；0成功；最大净升34.225–43.195mm，最大连续保持0.083–0.250s | 已学到提升，但没有保持；actor前后权重hash一致 |

475轮actor抽取文件SHA256 `03b57087a3dbe7bfaa027c863f891a846e2fb09b417286597533082391470b57`。原BC初态观测与三指teacher首条逐位相同，约60步后观测RMSE=0.167；不支持把失败归咎于初始姿态加载错误。

## 诊断结论与修正

1. **离线MSE不能作为抓取验收。** 原混合BC的MSE约0.000648，仍在纯student原生对照失败。原teacher保持零动作时，BC右腕平均Z指令还残留约−0.18至−0.27mm/s；5s积分误差足以超过0.5mm。这是指令积分分析，不是假装测得真实滑移。补只读原生评估、每环境height/接触/depth/ordinary/速度/drift/hold诊断、模型及合同SHA与无优化证明。
2. **PPO示范辅助更新受旧PPO Adam动量影响。** 新显式`stable_hold_v2`改用独立actor Adam，PPO自身state保留；辅助权重作用于实际学习率，保持样本和腕通道额外加权，并保存/恢复辅助优化器。旧模式默认行为保留。
3. **提升奖励未充分对齐成功接触。** 新模式仅有效对向指腹、ordinary≤5%、depth≤0.8mm时给密集lift奖励；实际连续native保持增加`120*(hold/5)^2`分/秒。独立奖励观察器从净升3mm开始，按固定锚点真实漂移给`-160*clamp((drift/0.5mm)^2,0,4)`分/秒，避免只在30mm处才惩罚滑移。它只读物理，不改动作、不固定物体，其events从不计入success。
4. **原随机噪声会继续扰动已接触的手。** 原std约0.079–0.090；新显式`holding_v2`初始0.005、上下限0.001/0.3。仍为真实Gaussian采样、原始采样动作及一致log probability，不裁剪样本伪造PPO概率。该改动尚不能单独证明必然成功。
5. **证据准入可选。** 默认PPO继续自由探索；仅显式启用严格验收模式时才检查真实student成功证据和累计诊断预算。保存证据必须核验actor/合同/实际训练reset族，不能按段绕过或仅信布尔值。教师成功及MSE不能通过student准入。
6. **因果接触历史是可选实验。** 原351维缺少teacher的控制周期均值/EMA/depth历史，但已有时间和上一动作，离线检查也没有发现完全相同obs对应冲突动作。401维扩展有严格合同、显式零列扩展和CPU测试；本次4096恢复仍使用351维，不能把该实验称为已证实根因或已原生成功。

真实成功仍要求：杆底面相对初始端支架净升≥30mm、连续5s、三点相对双手固定锚漂移≤0.5mm、有效对向指腹、ordinary≤5%、原生深度≤0.8mm和速度<0.03m/s。力均分是软目标，不设固定力或抓取阶段门槛。

## 复现入口

显式准备派生断点（拒绝覆盖源文件，另外保存来源SHA与变化清单）：

```bash
bash scripts/python.sh scripts/prepare_dh116_surface_repair_resume.py \
  --input runs/deployment/new_host_20261031_a21702624634466304168937/before_student_fix_4096.pt \
  --output runs/student_transfer_fix_20261031_as/repaired_resume.pt \
  --demonstrations runs/surface_multistyle_teacher_20261031_ao/surface_demonstrations.pt
```

从该派生断点启动4096环境，每环境目标2048episode；各块复用完整checkpoint，不清已有计数：

```bash
bash scripts/python.sh scripts/train_dh116_surface_multistyle.py \
  --demonstrations runs/surface_multistyle_teacher_20261031_ao/surface_demonstrations.pt \
  --resume runs/student_transfer_fix_20261031_as/repaired_resume.pt \
  --output runs/surface_multistyle_4096env_2048episodes_20261101_at \
  --num-envs 4096 --episodes-per-env 2048 --episodes-per-style 128 \
  --imitation-mode stable_hold_v2 --noise-mode holding_v2
```

仅看当前训练：`bash scripts/switch_current_tensorboard.sh <上述run目录>`；TensorBoard使用`runs/current`，不扫描全部历史训练，不修改或删除旧日志。

## 验证边界

本次成功对照是teacher的2/2，确定性student仍0，不能宣称已学会稳定抓取。后续按恢复训练的native successes继续监控。

全库测试受缺失Open3D、历史DeformX绝对路径资产和旧stage fixtures影响：首次collection有11个依赖错误；排除该11文件后4129 passed、91 failed、8 skipped、14 errors、64 subtests。没有装回用户已移除的DeformX，也不声称全库通过；未运行干净基线，不能断言每个失败均与本次改动无关。最终相关回归见下。

## 最终回归与派生断点核验

- 实际在hand环境通过`DH116_NUM_THREADS=4 bash scripts/python.sh -m pytest -q ...`执行surface全组及effective-pad、窗口、原生测量、free-wrist、multifinger相关回归：**1373 passed、85 subtests passed**，唯一warning为测试0输出维度网络时的PyTorch初始化提示。完整argv与输出保留于`runs/student_transfer_fix_20261031_as/release_cpu_tests_final_command.json`、`release_cpu_tests_final.log`。
- 独立审查发现非合格速度窗口的置零坐标会制造虚假漂移惩罚，先以−640分/秒复现，再将奖励读取限定为完整、物理合格窗口的真实drift；真实超限滑移仍扣分，原success事件不变。修复后复审无新增P1/P2阻塞。
- `prepare_dh116_surface_repair_resume.py`实际执行退出0。派生SHA256：`7b7951784e90faf8c51d81e394fd0e387886257c70db4defe72d8046c3e7e3fe`；actor/critic逐张量相同、非noise PPO状态/参数组不变、counter/RNG不变，仅noise变化坐标moments清零、独立辅助Adam初始化，全部tensor有限。来源模型及示范文件SHA前后复核一致。
- 诊断原件与旧475断点已下载本地`hand-deployment-a21702624634466304168937/transfer_diagnosis_evidence.zip`，压缩包SHA256 `e3c8ad42834f85bcb6cafc5ef4dbb2805ffa44bca25e32f85e0e0bc57ac32934`，内含逐文件manifest。模型和原生日志不纳入Git。

## 原生续训已验证

`surface_multistyle_4096env_2048episodes_20261101_at`已从475轮恢复并完成476、477轮真实PPO更新；当前首块为二指，4096环境，原53533个episode/min8保留，success0、numerical reset0。每块目标最低128回合，最终目标2048；回合预算跨块延续。每次更新同时包含真实on-policy PPO和独立示范辅助step，辅助optimizer已有6组状态，checkpoint全部tensor有限。验证原件：`runs/student_transfer_fix_20261031_as/resume_native_verification.json`。

后台tmux：`dh116_4096env_2048episodes_20261101`。`student_fix_maintenance.json`已改phase=training，status_file=`multistyle_status.json`，首次success监控已跟随新run，健康且success0不重复通知。

专用TensorBoard只列`block_00_pinch_thumb_index/tensorboard`，新增`reward/surface_hold_progress`、`reward/surface_slip_window`及`reward/total`均可见；远端6006及本地16006 HTTP200。历史日志保留。最终是否学会5s稳定抬升仍需后续原生训练和独立验证，本次没有伪造成功记录。
