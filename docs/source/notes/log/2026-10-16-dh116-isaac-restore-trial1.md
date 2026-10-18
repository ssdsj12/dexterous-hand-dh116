# 实际Isaac阶段恢复试验1启动

父节点：[D4阶段恢复](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)，[计划Task5](../../docs/superpowers/plans/2026-10-16-dh116-stage-reset.md)。用户明确允许正常停止旧实例并切换。

旧trial2_load115：STOP_REQUESTED协作停止，PID15485消失，summary/point_probe保存；failure_step21790、724点J全部通过。passedFalse及operator stop明确，不当成功。没有删除旧轨迹。

新执行session97738，调度脚本/home/xk/coding/M1/assets/dh116_v2/run_stage_restore_trial1.py。初始生产者PID17982，output stage_restore_trial1_producer，日志同名.log，状态stage_restore_trial1_status.json。全部数值单线程/taskset core0，核验线程亲和力均0。参数沿用trial2的载荷/保护/材料/驱动配置，只新增--save-stage-states；不改机器人产品代码。

调度顺序：生产者实际落定合格→独占stage_approach.json→固定实际文件SHA→STOP_REQUESTED→等待子进程退出且最终点J通过→同参数新实例加restore路径/SHA，输出stage_restore_trial1_approach。任一步失败报告并停止，无自动重试。后续恢复实例继续尝试接近/闭指/保持/抬升并保存合格阶段，不会授予连续whole-run资格。

当前仅启动，真实恢复复验尚未发生。后续读status与新summary/trajectory确认，勿重复启动或轮询旧PID15485。

## 实际恢复复验结果（覆盖上段启动状态）

生产者在step4999实际settle→approach，导出stage_approach.json；SHA为69d7573f08ddb29e4f5f92cf897abfa910abfdc9de0e8d15b5e3529dcb96497d，source.passed_stages仅settle。随后协作停止，最终164点J全部通过，旧PID17982消失。runner_snapshot及source_snapshot/dh116、co_sim已保留。

同一调度session97738启动恢复实例PID18934，output stage_restore_trial1_approach。首帧恢复后wire支撑误差2.47µm、无约束；进入restore_validate而非直接授权接近。实际trajectory第1000步（第1001条/0.1001s）readyTrue、abortedFalse、passedTrue，转入approach；复验区间最大材料锚点变化8.858646185109102e-7m（0.886µm）。imported_evidence=['settle']，continuous_grasp_qualified=False。最新核验step1177/approach/passedTrue，PID18934活跃、core0。

结论：首个真实“落定阶段保存→新Isaac加载→新物理复验→继续接近”链路通过。没有证明close/hold/lift恢复或连续抓取成功；实例继续原160000步上限，切勿重启。源summary中approach FAIL是人为停止未执行完该阶段，不影响其已合格settle入口，不能将其当完整成功。下一步观察实际接近→闭指，合格时自动导出后续阶段文件。

后续已确认同PID18934活跃，session97738 wait正常返回仍运行。step2467/approach/passedTrue，四点误差29.988/29.993/29.964/29.960mm，处于起始渐进轨迹；左右arm速度.008096/.004078rad/s，误差.000418/.000300rad，非目标净接触0，手耦合/跟踪门通过。恢复实例也保存了runner_snapshot与source_snapshot。不重启、不热改；此轮属于已验证等待，未夹持抬升。

最新同进程再验证：PID18934 elapsed3:57/CPU99.3%，step2798左右速度.010596/.005330rad/s、手门通过；wait同session50s仍运行，step3460/approach/passedTrue，四点误差29.899/29.912/29.839/29.828mm继续缩小。无重启，继续原阶段验证。

同session再次50s验证等待：PID18934在5:09仍活CPU99.5%；最新step4357/approach/passedTrue，四点误差29.643/29.661/29.607/29.587mm，左右臂速度.021932/.011007rad/s。尚未进入闭指，不修改代码或参数。

后续50s验证等待：PID18934在6:22仍活CPU99.7%；step5261/approach/passedTrue，四点误差29.185/29.207/29.219/29.192mm，左右臂速度.027942/.014054rad/s。仍运行同一实例，未闭指、未放宽保护。

再次验证等待：PID18934在7:35仍活CPU99.8%，同session等50s后step6165/approach/passedTrue；四点误差28.515/28.535/28.655/28.623mm，左右速度.033477/.016881rad/s。正常接近中，未闭指；继续原实例。

两次50s验证等待：PID18934在8:48仍活CPU99.7%；step7062通过，左右速度.038518/.019454rad/s。随后step7764/approach/passedTrue，四点误差26.890/26.902/27.204/27.173mm继续下降；未进入闭指，不重启不改参数。

同PID在10:59仍活CPU99.8%，同session50s后step8680/approach/passedTrue，误差25.813/25.819/26.142/26.116mm；左右速度.046457/.023468rad/s。继续第二次50s等待，仍运行；未闭指或抬升，安全门未改变。

同PID在13:09仍活CPU99.8%，两次50s等待均仍运行。step10286通过、左右速度.052846/.026709rad/s；末step11003/approach/passedTrue，四点误差22.887/22.887/22.970/22.964mm继续收敛。仍未闭指，未重启或改参数。

同PID在15:24仍活CPU99.8%，两次50s等待仍运行。step11917通过、左右速度.057871/.029168rad/s；末step12646/approach/passedTrue，四点误差20.653/20.652/20.568/20.571mm。继续同实例，未提前授权闭指。

同PID在17:33仍活CPU99.8%，两次50s等待仍运行。step13501通过、左右速度.061236/.030877rad/s；末step14212/approach/passedTrue，四点误差18.374/18.371/18.285/18.289mm，继续接近；未闭指/抬升，不重启。

同PID在19:41仍活CPU99.9%，两次50s等待仍运行。step15083通过、左右速度.062987/.031928rad/s；末step15798/approach/passedTrue，四点误差15.961/15.956/15.992/15.992mm。仍按原轨迹和安全门推进，无提前闭指。

同PID在21:50仍活CPU99.9%，两次50s等待仍运行。step16633通过、左右速度.063777/.031889rad/s；末step17323/approach/passedTrue，四点误差13.640/13.633/13.748/13.747mm。继续原实例，未闭指/保持/抬升。

同PID在23:58仍活CPU99.9%，两次50s等待仍运行。step18109通过、左右速度.063548/.031889rad/s；末step18770/approach/passedTrue，四点误差11.495/11.488/11.603/11.602mm。原安全门不变，仍未闭指。

同PID在26:06仍活CPU99.8%，两次50s等待仍运行。step19552通过、左右速度.061233/.030954rad/s；末step20190/approach/passedTrue，四点误差9.440/9.432/9.525/9.527mm。接近继续收敛，未改变阶段门或重启。

同PID在28:15仍活CPU99.8%，两次50s等待仍运行。step20988通过、左右速度.058196/.029301rad/s；末step21623/approach/passedTrue，四点误差7.486/7.479/7.489/7.495mm。尚未闭指，继续同一物理实例。

同PID在30:23仍活CPU99.9%，两次50s等待仍运行。step22388通过、左右速度.054000/.027271rad/s；末step23032/approach/passedTrue，四点误差5.703/5.696/5.607/5.618mm。安全门未改变，仍处接近阶段。

同PID在32:33仍活CPU99.9%，两次50s等待仍运行。step23798通过、左右速度.048761/.024584rad/s；末step24433/approach/passedTrue，四点误差4.050/4.041/3.922/3.937mm。继续原接近末段，未提前授权闭指。

Git baseline/candidate：5e3bcc4+dirty，未提交。前置489回归已记录于[接线日志](2026-10-16-dh116-stage-state-runtime.md)。
