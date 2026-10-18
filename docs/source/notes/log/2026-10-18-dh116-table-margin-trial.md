# 左手桌面余量增加及实际测试

父节点：[D4-approach-tracking-margin](../todo/T500-m1-dual-panda-o6-bimanual-mpc.md)。用户批准增加余量再测试。基线5e3bcc4+dirty，未改产品控制代码/安全门。

离线试验：绕线缆轴旋转正方向使中指靠桌；负方向使拇指靠桌，排除。改为保持朝向，将左手接近起终点整体上移1mm，右手、桌面、wire、指腹参考、物理与夹力均不变。新姿态IK收敛，起终点完整源几何通过，整手桌面最小间隙从1.2014mm增至2.2014mm。偏移后的实际承载能力仍待验证，不能当作成功抓取。

外部可复现脚本assets/dh116_v2/screen_table_margin_20261018.py、certify_raise1mm_20261018.py；各轮结果独立保留，未覆盖旧证据。新approach通过94次joint-box区间检查及实际loader验证；prepose在wire初始高度.318和落定高度.308均通过。仅名义源证书，不包含全部跟踪误差包络，不代表物理成功。

配置目录assets/dh116_v2/table_margin_raise1mm_20261018：prepose SHA c477908661abbaa7ac9f0d1a5ed866aa55a3afa8b3259770dfcfabe6e90ef1be；approach SHA 27f36f412b57e6ddc9fe56c9820533af113d12734a6a91b3821f193744b1a7aa。

实际测试已启动PID6780/session44334，output assets/dh116_v2/table_margin_raise1mm_trial1，log同名.log。沿用上一试验所有运行参数，core0单线程，GPU0，--grasp-sequence --coordinated-closure --save-stage-states，160000步×100µs。已确认进程及settle实际轨迹。新配置不符合旧状态文件pin，不伪造兼容性；新运行自行保存合格阶段供后续恢复。

下一步保持同进程，检查接近末段桌面保护，再验证真实闭指/保持/抬升。尚无这些成功证据。
