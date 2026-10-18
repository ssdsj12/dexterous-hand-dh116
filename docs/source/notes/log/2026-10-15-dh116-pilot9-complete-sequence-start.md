# Pilot9闭指诊断完成，启动连续抓取验收

父节点：[D3/D4](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。Baseline/Candidate `5e3bcc4` + dirty，Inline/core0，全部原保护。

## Pilot9完整物理证据

session89681/PID78902正常结束，12000步/1.2s，summary.passed=True仅为独立触线初始化闭指诊断。主仓assets/dh116_v2/closure_pilot_trial9_iteration1保留summary、trajectory、point_probe、aggregation及aggregate_producer.py。

- 四路首次有效step1141，之后没有接触丢失，最长连续1.0859s。
- 全部90%预载首次step6582，最长预载连续.5411s，不足正式保持1s（且未经历连续接近）。
- 末载荷.838669/.801575/.877087/.908784N，全部valid。
- 最大首次接触后材料滑移.676679mm。
- 非目标保守下界最小1.0000027mm，未放宽门；这是保守证书下界，不是声称实测真实间隙恰为此值。
- 396点J探针通过：max位置1.46999e-5m、J2.18463e-5m/rad、速度1.15033e-9m/s。
- grasp/hold/lift资格仍全部False；不能将summary局部passed解释成完整抓取成功。

## 实际协作停止验证

随后单独cooperative_stop_trial1，session11188结束。纯重力保持中创建STOP_REQUESTED，3282步前正确转入异常/finally，summary.passed=False、error明确operator stop requested，108已完成探针保存且通过，run_manifest存在并标initialized_not_qualified。无SIGINT，真实Isaac正常停止保存路径已验证。未使用wire/抓取来声称此演练成功。

## 当前连续流程

新session68391运行grasp_sequence_trial1_iteration1，最大160000×100µs=16s模拟时间。预位姿postpilot8_prepose_v1 SHA74895d9ff65b25cc3947c4180dcfc9af40437e6f8473f7670bfcf687b081fdfb，approach_source_v1 SHA37ad0f20ceef8b30c4902128da1339aefb83edb9ef55416f5ef562682b73eb4f，postpilot7_point_references_v1；位置迭代1，其余与pilot9物理/夹力/保护相同。显式--grasp-sequence（不是closure-pilot），实际settle/approach/close/hold/lift门；手接触之前不得假绑定，任何保护失败停止。

日志/tmp/dh116-grasp-sequence-trial1-iteration1.log。启动尚非验收；下一步检查实际初始化/状态转移与报告。严格一个Isaac实例/core0。安全停止文件机制已验证，可在输出目录创建STOP_REQUESTED；不用SIGINT丢失证据。

## 首个实际阶段转移

session68391/PID80300确认活、taskset亲和力0。step5233（约.5234s模拟时间）实际stage=approach，passed=True，permit_close/permit_lift=False。说明.5s实测落定门已通过。wire站点高度约.307997535m、支撑误差约2.47µm；双手尚在接近起始，端点位置误差约30mm。当前仅完成settle，未完成approach/close/hold/lift。runner_snapshot.py和source_snapshot/{dh116,co_sim,elastica}已存本次输出目录。

后续同实例连续观察到step9998（约1.0s模拟时间）：stage仍approach、passed=True，四点到终点误差28.659/28.680/28.773/28.741mm，较初始30mm已实际推进。左/右臂跟踪误差.001642/.001353rad，最大关节速度.03249/.01638rad/s；未触发保护。仍未进入闭指、保持或抬升。PID80300此时运行约13:56、CPU99.9%，原session68391继续。

后续验证等待至step14937≈1.494s，stage=approach/passed=True，四点距终点22.979/22.978/23.064/23.057mm；左右arm误差.002756/.001393rad、速度.05500/.02780rad/s，原保护未触发。PID80300活，CPU单核，未重新启动。当前只是接近中，不是接近门/抓取门通过。

继续同一session验证等待至step19705≈1.971s，approach/passed=True，四点距终点16.10–16.13mm；左右arm误差.003190/.001596rad、速度.06370/.03189rad/s。PID80300仍活、CPU99.9%；未闭指、未触发保护、没有重启。
