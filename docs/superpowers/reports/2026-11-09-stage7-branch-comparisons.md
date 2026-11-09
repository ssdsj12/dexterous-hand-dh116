# Stage 7：64 项短切换聚合附件

原生 64/64、六时点后处理 64/64，以下 256 行窗口数据和 256 行事件数据来自最终交叉投影，不是运行日志或新增原生实验。投影 SHA256：13270507cc48769e8d2fd43061e612b1e162a179baa6af15e947985ef1f5cdbd；原始生成表 SHA256：584b3a1c7ba6fdd97e60fcf6a3b0fc593f6b86fd1a5c4dca3868daeeaa97eda4。本附件只补充口径说明并更正两个诊断漂移表头，未修改数值行或事件单元格。

物体净高度已排除支架预抬 60 mm，但 HOLD 窗口还继承 Teacher 建立的约 30 mm；Delta 另扣除同 env/episode 最后一个真实前缀样本。最长 60 步只有 1 s，bounded_censored 是正常截断，不是失败或成功 episode；继承的原生 Hold 也不是 Student 独立保持时长。四来源属于同一标称布局的场景副本；CONTACT/LIFT_ONSET 恰好同 tick，不能双算独立物理条件。

Diagnostic prior-anchor drift 来自最低 3 mm 的 surface_slip_tracking；原生 30 mm/5 s 成功窗口使用另一份锚与计时。诊断漂移跨 0.5 mm 不能直接指认原生成功窗口同锚违规，锚重建后的回落也不代表真实恢复。该值含三个杆标记在左右手系下的相对转动效应，不是逐指腹摩擦滑移速度。

第一张表只计 Student-owned 段，接触为聚合双侧对向接触，不冒充逐指载荷。第二张表记录整个冷重放轨迹的首次事件并保留真实 owner；Teacher 前缀事件不是 Student 首次接触丢失。accepted>0 的早期闪断、原 0.05 N bearing、两次 Teacher 健康对照缺失和原标定跟踪告警分别定义；原标定告警不等于原生关节终止。NA 可能是本有界窗口未观察到事件或该测量未记录，不能补成 0。

峰值 packet 直接保存 env/source/execution episode UID、policy/World 步、timestamp 和 raw 行指针；六时点事件 packet 另有派生 SDK 步及 NPZ 指针，不能说 raw-only 峰值已经记录后两项。上述分类记录保存在 R6/stage7_branch_table_projection_final_v01/projection.json，R6 定义与[主报告](2026-11-09-stage7-teacher-student-divergence.md)一致。不同环境或不同时刻峰值不能拼成同一条轨迹；所有误差和时序均不作唯一因果证明。

| Job | env | Student steps | Peak net mm | Delta peak mm | Contact s | Contact loss | Diagnostic prior-anchor drift max mm | Ending |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| branch_E0_PRE_CONTACT_01 | 0 | 1 | -0.006611399 | 6.360271e-06 | 0 | 0 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_01 | 1 | 1 | -0.008345381 | -0.0003901753 | 0 | 0 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_01 | 2 | 1 | -0.007099062 | -0.0004746604 | 0 | 0 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_01 | 3 | 1 | -0.007106372 | -0.0004806542 | 0 | 0 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_04 | 0 | 4 | -0.006611399 | 6.360271e-06 | 0.02708333 | 2 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_04 | 1 | 4 | -0.008345381 | -0.0003901753 | 0.02083333 | 3 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_04 | 2 | 4 | -0.007099062 | -0.0004746604 | 0.02291667 | 3 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_04 | 3 | 4 | -0.007106372 | -0.0004806542 | 0.02291667 | 3 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_16 | 0 | 16 | -0.006611399 | 6.360271e-06 | 0.2270833 | 2 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_16 | 1 | 16 | -0.008345381 | -0.0003901753 | 0.21875 | 4 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_16 | 2 | 16 | -0.007099062 | -0.0004746604 | 0.2229167 | 3 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_16 | 3 | 16 | -0.007106372 | -0.0004806542 | 0.2208333 | 4 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_60 | 0 | 60 | 6.598621e-05 | 0.006683745 | 0.55625 | 23 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_60 | 1 | 60 | 0.01198078 | 0.01993599 | 0.5333333 | 17 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_60 | 2 | 60 | 8.058507e-05 | 0.006704986 | 0.56875 | 25 | NA | bounded_censored |
| branch_E0_PRE_CONTACT_60 | 3 | 60 | 0.000559185 | 0.007184903 | 0.5291667 | 45 | NA | bounded_censored |
| branch_E0_CONTACT_01 | 0 | 1 | -0.008395078 | 0.0003629293 | 0.0125 | 1 | NA | bounded_censored |
| branch_E0_CONTACT_01 | 1 | 1 | -0.009533042 | -6.523126e-05 | 0.01666667 | 0 | NA | bounded_censored |
| branch_E0_CONTACT_01 | 2 | 1 | -0.009178098 | 0.0002616172 | 0.0125 | 1 | NA | bounded_censored |
| branch_E0_CONTACT_01 | 3 | 1 | -0.009182911 | 0.0002511451 | 0.0125 | 1 | NA | bounded_censored |
| branch_E0_CONTACT_04 | 0 | 4 | -0.008395078 | 0.0003629293 | 0.06041667 | 2 | NA | bounded_censored |
| branch_E0_CONTACT_04 | 1 | 4 | -0.009533042 | -6.523126e-05 | 0.06666667 | 0 | NA | bounded_censored |
| branch_E0_CONTACT_04 | 2 | 4 | -0.009178098 | 0.0002616172 | 0.0625 | 1 | NA | bounded_censored |
| branch_E0_CONTACT_04 | 3 | 4 | -0.009182911 | 0.0002511451 | 0.0625 | 1 | NA | bounded_censored |
| branch_E0_CONTACT_16 | 0 | 16 | -0.008395078 | 0.0003629293 | 0.2604167 | 2 | NA | bounded_censored |
| branch_E0_CONTACT_16 | 1 | 16 | -0.009533042 | -6.523126e-05 | 0.2666667 | 0 | NA | bounded_censored |
| branch_E0_CONTACT_16 | 2 | 16 | -0.008917271 | 0.0005224436 | 0.2625 | 1 | NA | bounded_censored |
| branch_E0_CONTACT_16 | 3 | 16 | -0.008947893 | 0.0004861631 | 0.2625 | 1 | NA | bounded_censored |
| branch_E0_CONTACT_60 | 0 | 60 | 0.0008880472 | 0.009646055 | 0.4854167 | 47 | NA | bounded_censored |
| branch_E0_CONTACT_60 | 1 | 60 | 0.0001540426 | 0.009621853 | 0.5645833 | 19 | NA | bounded_censored |
| branch_E0_CONTACT_60 | 2 | 60 | 0.00156008 | 0.01099979 | 0.4895833 | 50 | NA | bounded_censored |
| branch_E0_CONTACT_60 | 3 | 60 | 0.002166031 | 0.01160009 | 0.50625 | 51 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_01 | 0 | 1 | -0.008395078 | 0.0003629293 | 0.0125 | 1 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_01 | 1 | 1 | -0.009533042 | -6.523126e-05 | 0.01666667 | 0 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_01 | 2 | 1 | -0.009178098 | 0.0002616172 | 0.0125 | 1 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_01 | 3 | 1 | -0.009182911 | 0.0002511451 | 0.0125 | 1 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_04 | 0 | 4 | -0.008395078 | 0.0003629293 | 0.06041667 | 2 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_04 | 1 | 4 | -0.009533042 | -6.523126e-05 | 0.06666667 | 0 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_04 | 2 | 4 | -0.009178098 | 0.0002616172 | 0.0625 | 1 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_04 | 3 | 4 | -0.009182911 | 0.0002511451 | 0.0625 | 1 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_16 | 0 | 16 | -0.008395078 | 0.0003629293 | 0.2604167 | 2 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_16 | 1 | 16 | -0.009533042 | -6.523126e-05 | 0.2666667 | 0 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_16 | 2 | 16 | -0.008917271 | 0.0005224436 | 0.2625 | 1 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_16 | 3 | 16 | -0.008947893 | 0.0004861631 | 0.2625 | 1 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_60 | 0 | 60 | 0.0008880472 | 0.009646055 | 0.4854167 | 47 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_60 | 1 | 60 | 0.0001540426 | 0.009621853 | 0.5645833 | 19 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_60 | 2 | 60 | 0.00156008 | 0.01099979 | 0.4895833 | 50 | NA | bounded_censored |
| branch_E0_LIFT_ONSET_60 | 3 | 60 | 0.002166031 | 0.01160009 | 0.50625 | 51 | NA | bounded_censored |
| branch_E0_HOLD_01 | 0 | 1 | 30.10905 | 0.04936745 | 0.01666667 | 0 | 0.005186736 | bounded_censored |
| branch_E0_HOLD_01 | 1 | 1 | 30.10545 | 0.04975731 | 0.01666667 | 0 | 0.01162835 | bounded_censored |
| branch_E0_HOLD_01 | 2 | 1 | 30.09862 | 0.04917124 | 0.01666667 | 0 | 0.01164573 | bounded_censored |
| branch_E0_HOLD_01 | 3 | 1 | 30.1371 | 0.04898566 | 0.01666667 | 0 | 0.04770222 | bounded_censored |
| branch_E0_HOLD_04 | 0 | 4 | 30.25454 | 0.1948482 | 0.06666667 | 0 | 0.01026721 | bounded_censored |
| branch_E0_HOLD_04 | 1 | 4 | 30.25356 | 0.197872 | 0.06666667 | 0 | 0.02063985 | bounded_censored |
| branch_E0_HOLD_04 | 2 | 4 | 30.2429 | 0.1934525 | 0.06666667 | 0 | 0.02127045 | bounded_censored |
| branch_E0_HOLD_04 | 3 | 4 | 30.28086 | 0.1927405 | 0.06666667 | 0 | 0.05569299 | bounded_censored |
| branch_E0_HOLD_16 | 0 | 16 | 30.81095 | 0.7512629 | 0.2666667 | 0 | 0.08149568 | bounded_censored |
| branch_E0_HOLD_16 | 1 | 16 | 30.8259 | 0.7702102 | 0.2666667 | 0 | 0.1445246 | bounded_censored |
| branch_E0_HOLD_16 | 2 | 16 | 30.7855 | 0.7360534 | 0.2666667 | 0 | 0.1023752 | bounded_censored |
| branch_E0_HOLD_16 | 3 | 16 | 30.83854 | 0.7504227 | 0.2666667 | 0 | 0.1442487 | bounded_censored |
| branch_E0_HOLD_60 | 0 | 60 | 32.59538 | 2.535695 | 1 | 0 | 0.298016 | bounded_censored |
| branch_E0_HOLD_60 | 1 | 60 | 32.47315 | 2.417465 | 1 | 0 | 0.5162977 | bounded_censored |
| branch_E0_HOLD_60 | 2 | 60 | 32.43381 | 2.384365 | 1 | 0 | 0.3154534 | bounded_censored |
| branch_E0_HOLD_60 | 3 | 60 | 32.70906 | 2.620943 | 1 | 0 | 0.5041699 | bounded_censored |
| branch_B0_PRE_CONTACT_01 | 0 | 1 | -0.006606114 | 1.164476e-05 | 0 | 0 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_01 | 1 | 1 | -0.008274677 | -0.000319471 | 0 | 0 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_01 | 2 | 1 | -0.007093689 | -0.0004692876 | 0 | 0 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_01 | 3 | 1 | -0.007101003 | -0.000475285 | 0 | 0 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_04 | 0 | 4 | -0.006606114 | 1.164476e-05 | 0.02291667 | 2 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_04 | 1 | 4 | -0.008224504 | -0.0002692979 | 0.02916667 | 2 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_04 | 2 | 4 | -0.007093689 | -0.0004692876 | 0.025 | 2 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_04 | 3 | 4 | -0.007101003 | -0.000475285 | 0.025 | 2 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_16 | 0 | 16 | -0.006606114 | 1.164476e-05 | 0.2229167 | 2 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_16 | 1 | 16 | -0.008224504 | -0.0002692979 | 0.2291667 | 2 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_16 | 2 | 16 | -0.007093689 | -0.0004692876 | 0.225 | 2 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_16 | 3 | 16 | -0.007101003 | -0.000475285 | 0.225 | 2 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_60 | 0 | 60 | 9.619228e-05 | 0.006713951 | 0.4583333 | 6 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_60 | 1 | 60 | 6.762349e-05 | 0.008022829 | 0.43125 | 43 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_60 | 2 | 60 | 3.336805e-05 | 0.006657769 | 0.45625 | 41 | NA | bounded_censored |
| branch_B0_PRE_CONTACT_60 | 3 | 60 | 5.654876e-05 | 0.006682267 | 0.4520833 | 45 | NA | bounded_censored |
| branch_B0_CONTACT_01 | 0 | 1 | -0.008388144 | 0.0003698637 | 0.0125 | 1 | NA | bounded_censored |
| branch_B0_CONTACT_01 | 1 | 1 | -0.009530664 | -6.285356e-05 | 0.01666667 | 0 | NA | bounded_censored |
| branch_B0_CONTACT_01 | 2 | 1 | -0.009188147 | 0.0002515677 | 0.0125 | 1 | NA | bounded_censored |
| branch_B0_CONTACT_01 | 3 | 1 | -0.009187222 | 0.0002468343 | 0.0125 | 1 | NA | bounded_censored |
| branch_B0_CONTACT_04 | 0 | 4 | -0.008388144 | 0.0003698637 | 0.05833333 | 2 | NA | bounded_censored |
| branch_B0_CONTACT_04 | 1 | 4 | -0.009530664 | -6.285356e-05 | 0.06666667 | 0 | NA | bounded_censored |
| branch_B0_CONTACT_04 | 2 | 4 | -0.009188147 | 0.0002515677 | 0.0625 | 1 | NA | bounded_censored |
| branch_B0_CONTACT_04 | 3 | 4 | -0.009187222 | 0.0002468343 | 0.0625 | 1 | NA | bounded_censored |
| branch_B0_CONTACT_16 | 0 | 16 | -0.00774744 | 0.001010568 | 0.2333333 | 4 | NA | bounded_censored |
| branch_B0_CONTACT_16 | 1 | 16 | -0.008081433 | 0.001386378 | 0.2125 | 2 | NA | bounded_censored |
| branch_B0_CONTACT_16 | 2 | 16 | -0.007375861 | 0.002063854 | 0.2145833 | 3 | NA | bounded_censored |
| branch_B0_CONTACT_16 | 3 | 16 | -0.007382089 | 0.002051967 | 0.2145833 | 3 | NA | bounded_censored |
| branch_B0_CONTACT_60 | 0 | 60 | 4.156257e-05 | 0.00879957 | 0.34375 | 52 | NA | bounded_censored |
| branch_B0_CONTACT_60 | 1 | 60 | 0.007643995 | 0.01711181 | 0.5125 | 29 | NA | bounded_censored |
| branch_B0_CONTACT_60 | 2 | 60 | 9.633581e-05 | 0.009536051 | 0.3791667 | 54 | NA | bounded_censored |
| branch_B0_CONTACT_60 | 3 | 60 | 5.342006e-05 | 0.009487476 | 0.38125 | 54 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_01 | 0 | 1 | -0.008388144 | 0.0003698637 | 0.0125 | 1 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_01 | 1 | 1 | -0.009530664 | -6.285356e-05 | 0.01666667 | 0 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_01 | 2 | 1 | -0.009188147 | 0.0002515677 | 0.0125 | 1 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_01 | 3 | 1 | -0.009187222 | 0.0002468343 | 0.0125 | 1 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_04 | 0 | 4 | -0.008388144 | 0.0003698637 | 0.05833333 | 2 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_04 | 1 | 4 | -0.009530664 | -6.285356e-05 | 0.06666667 | 0 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_04 | 2 | 4 | -0.009188147 | 0.0002515677 | 0.0625 | 1 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_04 | 3 | 4 | -0.009187222 | 0.0002468343 | 0.0625 | 1 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_16 | 0 | 16 | -0.00774744 | 0.001010568 | 0.2333333 | 4 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_16 | 1 | 16 | -0.008081433 | 0.001386378 | 0.2125 | 2 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_16 | 2 | 16 | -0.007375861 | 0.002063854 | 0.2145833 | 3 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_16 | 3 | 16 | -0.007382089 | 0.002051967 | 0.2145833 | 3 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_60 | 0 | 60 | 4.156257e-05 | 0.00879957 | 0.34375 | 52 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_60 | 1 | 60 | 0.007643995 | 0.01711181 | 0.5125 | 29 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_60 | 2 | 60 | 9.633581e-05 | 0.009536051 | 0.3791667 | 54 | NA | bounded_censored |
| branch_B0_LIFT_ONSET_60 | 3 | 60 | 5.342006e-05 | 0.009487476 | 0.38125 | 54 | NA | bounded_censored |
| branch_B0_HOLD_01 | 0 | 1 | 30.10925 | 0.04956306 | 0.01666667 | 0 | 0.005248375 | bounded_censored |
| branch_B0_HOLD_01 | 1 | 1 | 30.1054 | 0.04970687 | 0.01666667 | 0 | 0.0112333 | bounded_censored |
| branch_B0_HOLD_01 | 2 | 1 | 30.09885 | 0.04940075 | 0.01666667 | 0 | 0.01151987 | bounded_censored |
| branch_B0_HOLD_01 | 3 | 1 | 30.13749 | 0.04937566 | 0.01666667 | 0 | 0.04693145 | bounded_censored |
| branch_B0_HOLD_04 | 0 | 4 | 30.25543 | 0.195744 | 0.06666667 | 0 | 0.01201954 | bounded_censored |
| branch_B0_HOLD_04 | 1 | 4 | 30.25385 | 0.1981584 | 0.06666667 | 0 | 0.01444494 | bounded_censored |
| branch_B0_HOLD_04 | 2 | 4 | 30.24407 | 0.1946196 | 0.06666667 | 0 | 0.02263302 | bounded_censored |
| branch_B0_HOLD_04 | 3 | 4 | 30.28147 | 0.1933516 | 0.06666667 | 0 | 0.04693145 | bounded_censored |
| branch_B0_HOLD_16 | 0 | 16 | 30.81301 | 0.7533197 | 0.2666667 | 0 | 0.1265026 | bounded_censored |
| branch_B0_HOLD_16 | 1 | 16 | 30.83435 | 0.7786659 | 0.2666667 | 0 | 0.1183788 | bounded_censored |
| branch_B0_HOLD_16 | 2 | 16 | 30.79201 | 0.7425605 | 0.2666667 | 0 | 0.1440553 | bounded_censored |
| branch_B0_HOLD_16 | 3 | 16 | 30.82479 | 0.7366728 | 0.2666667 | 0 | 0.08596587 | bounded_censored |
| branch_B0_HOLD_60 | 0 | 60 | 32.85481 | 2.795128 | 1 | 0 | 0.3688583 | bounded_censored |
| branch_B0_HOLD_60 | 1 | 60 | 32.94372 | 2.888032 | 1 | 0 | 0.5257884 | bounded_censored |
| branch_B0_HOLD_60 | 2 | 60 | 32.6857 | 2.636251 | 1 | 0 | 0.4395982 | bounded_censored |
| branch_B0_HOLD_60 | 3 | 60 | 32.63192 | 2.543798 | 1 | 0 | 0.5072673 | bounded_censored |
| branch_B1_PRE_CONTACT_01 | 0 | 1 | -0.006605142 | 1.261723e-05 | 0 | 0 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_01 | 1 | 1 | -0.008258396 | -0.0003031904 | 0 | 0 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_01 | 2 | 1 | -0.007092874 | -0.0004684728 | 0 | 0 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_01 | 3 | 1 | -0.007100091 | -0.0004743728 | 0 | 0 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_04 | 0 | 4 | -0.006605142 | 1.261723e-05 | 0.02708333 | 2 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_04 | 1 | 4 | -0.008229681 | -0.0002744748 | 0.01875 | 1 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_04 | 2 | 4 | -0.007092874 | -0.0004684728 | 0.02916667 | 2 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_04 | 3 | 4 | -0.007100091 | -0.0004743728 | 0.02708333 | 2 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_16 | 0 | 16 | -0.006605142 | 1.261723e-05 | 0.1895833 | 4 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_16 | 1 | 16 | -0.008229681 | -0.0002744748 | 0.1875 | 3 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_16 | 2 | 16 | -0.007092874 | -0.0004684728 | 0.1958333 | 4 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_16 | 3 | 16 | -0.007100091 | -0.0004743728 | 0.2041667 | 3 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_60 | 0 | 60 | -0.006605142 | 1.261723e-05 | 0.2604167 | 35 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_60 | 1 | 60 | -0.008229681 | -0.0002744748 | 0.3375 | 45 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_60 | 2 | 60 | -0.007092874 | -0.0004684728 | 0.2625 | 33 | NA | bounded_censored |
| branch_B1_PRE_CONTACT_60 | 3 | 60 | -0.007100091 | -0.0004743728 | 0.5354167 | 40 | NA | bounded_censored |
| branch_B1_CONTACT_01 | 0 | 1 | -0.008389498 | 0.0003685095 | 0.002083333 | 2 | NA | bounded_censored |
| branch_B1_CONTACT_01 | 1 | 1 | -0.009531859 | -6.404904e-05 | 0.01666667 | 0 | NA | bounded_censored |
| branch_B1_CONTACT_01 | 2 | 1 | -0.009224989 | 0.0002147261 | 0.0125 | 1 | NA | bounded_censored |
| branch_B1_CONTACT_01 | 3 | 1 | -0.009221494 | 0.0002125617 | 0.0125 | 1 | NA | bounded_censored |
| branch_B1_CONTACT_04 | 0 | 4 | -0.008389498 | 0.0003685095 | 0.04791667 | 3 | NA | bounded_censored |
| branch_B1_CONTACT_04 | 1 | 4 | -0.009531859 | -6.404904e-05 | 0.06666667 | 0 | NA | bounded_censored |
| branch_B1_CONTACT_04 | 2 | 4 | -0.009224989 | 0.0002147261 | 0.0625 | 1 | NA | bounded_censored |
| branch_B1_CONTACT_04 | 3 | 4 | -0.009221494 | 0.0002125617 | 0.0625 | 1 | NA | bounded_censored |
| branch_B1_CONTACT_16 | 0 | 16 | -0.008389498 | 0.0003685095 | 0.15 | 8 | NA | bounded_censored |
| branch_B1_CONTACT_16 | 1 | 16 | -0.009531859 | -6.404904e-05 | 0.1666667 | 5 | NA | bounded_censored |
| branch_B1_CONTACT_16 | 2 | 16 | -0.009224989 | 0.0002147261 | 0.1604167 | 6 | NA | bounded_censored |
| branch_B1_CONTACT_16 | 3 | 16 | -0.009221494 | 0.0002125617 | 0.1625 | 6 | NA | bounded_censored |
| branch_B1_CONTACT_60 | 0 | 60 | -0.008389498 | 0.0003685095 | 0.2041667 | 24 | NA | bounded_censored |
| branch_B1_CONTACT_60 | 1 | 60 | -0.009531859 | -6.404904e-05 | 0.4354167 | 37 | NA | bounded_censored |
| branch_B1_CONTACT_60 | 2 | 60 | -0.009224989 | 0.0002147261 | 0.2583333 | 19 | NA | bounded_censored |
| branch_B1_CONTACT_60 | 3 | 60 | -0.009221494 | 0.0002125617 | 0.375 | 44 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_01 | 0 | 1 | -0.008389498 | 0.0003685095 | 0.002083333 | 2 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_01 | 1 | 1 | -0.009531859 | -6.404904e-05 | 0.01666667 | 0 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_01 | 2 | 1 | -0.009224989 | 0.0002147261 | 0.0125 | 1 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_01 | 3 | 1 | -0.009221494 | 0.0002125617 | 0.0125 | 1 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_04 | 0 | 4 | -0.008389498 | 0.0003685095 | 0.04791667 | 3 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_04 | 1 | 4 | -0.009531859 | -6.404904e-05 | 0.06666667 | 0 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_04 | 2 | 4 | -0.009224989 | 0.0002147261 | 0.0625 | 1 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_04 | 3 | 4 | -0.009221494 | 0.0002125617 | 0.0625 | 1 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_16 | 0 | 16 | -0.008389498 | 0.0003685095 | 0.15 | 8 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_16 | 1 | 16 | -0.009531859 | -6.404904e-05 | 0.1666667 | 5 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_16 | 2 | 16 | -0.009224989 | 0.0002147261 | 0.1604167 | 6 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_16 | 3 | 16 | -0.009221494 | 0.0002125617 | 0.1625 | 6 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_60 | 0 | 60 | -0.008389498 | 0.0003685095 | 0.2041667 | 24 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_60 | 1 | 60 | -0.009531859 | -6.404904e-05 | 0.4354167 | 37 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_60 | 2 | 60 | -0.009224989 | 0.0002147261 | 0.2583333 | 19 | NA | bounded_censored |
| branch_B1_LIFT_ONSET_60 | 3 | 60 | -0.009221494 | 0.0002125617 | 0.375 | 44 | NA | bounded_censored |
| branch_B1_HOLD_01 | 0 | 1 | 30.10957 | 0.04988514 | 0.01666667 | 0 | 0.005802481 | bounded_censored |
| branch_B1_HOLD_01 | 1 | 1 | 30.10594 | 0.05024866 | 0.01666667 | 0 | 0.01231216 | bounded_censored |
| branch_B1_HOLD_01 | 2 | 1 | 30.09912 | 0.04967554 | 0.01666667 | 0 | 0.01193885 | bounded_censored |
| branch_B1_HOLD_01 | 3 | 1 | 30.13752 | 0.04939804 | 0.01666667 | 0 | 0.04700172 | bounded_censored |
| branch_B1_HOLD_04 | 0 | 4 | 30.25548 | 0.1957945 | 0.06666667 | 0 | 0.02870101 | bounded_censored |
| branch_B1_HOLD_04 | 1 | 4 | 30.25626 | 0.2005676 | 0.06666667 | 0 | 0.03480722 | bounded_censored |
| branch_B1_HOLD_04 | 2 | 4 | 30.24407 | 0.1946264 | 0.06666667 | 0 | 0.03616168 | bounded_censored |
| branch_B1_HOLD_04 | 3 | 4 | 30.27822 | 0.1900971 | 0.06666667 | 0 | 0.04830419 | bounded_censored |
| branch_B1_HOLD_16 | 0 | 16 | 30.80111 | 0.7414212 | 0.2666667 | 0 | 0.3567784 | bounded_censored |
| branch_B1_HOLD_16 | 1 | 16 | 30.83138 | 0.7756892 | 0.2666667 | 0 | 0.5509321 | bounded_censored |
| branch_B1_HOLD_16 | 2 | 16 | 30.78201 | 0.7325655 | 0.2666667 | 0 | 0.3440326 | bounded_censored |
| branch_B1_HOLD_16 | 3 | 16 | 30.77879 | 0.6906749 | 0.2666667 | 0 | 0.06510617 | bounded_censored |
| branch_B1_HOLD_60 | 0 | 60 | 32.32786 | 2.268174 | 1 | 0 | 0.5059978 | bounded_censored |
| branch_B1_HOLD_60 | 1 | 60 | 32.74132 | 2.685636 | 1 | 0 | 0.5509321 | bounded_censored |
| branch_B1_HOLD_60 | 2 | 60 | 32.1728 | 2.123356 | 1 | 0 | 0.5320116 | bounded_censored |
| branch_B1_HOLD_60 | 3 | 60 | 31.62844 | 1.540322 | 1 | 0 | 0.5323714 | bounded_censored |
| branch_B2_PRE_CONTACT_01 | 0 | 1 | -0.006605218 | 1.254086e-05 | 0 | 0 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_01 | 1 | 1 | -0.008271545 | -0.0003163388 | 0 | 0 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_01 | 2 | 1 | -0.00709294 | -0.0004685387 | 0 | 0 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_01 | 3 | 1 | -0.007100147 | -0.0004744294 | 0 | 0 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_04 | 0 | 4 | -0.006605218 | 1.254086e-05 | 0.02708333 | 2 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_04 | 1 | 4 | -0.008044877 | -8.967088e-05 | 0.03125 | 2 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_04 | 2 | 4 | -0.00709294 | -0.0004685387 | 0.02291667 | 2 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_04 | 3 | 4 | -0.007100147 | -0.0004744294 | 0.02708333 | 2 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_16 | 0 | 16 | -0.006605218 | 1.254086e-05 | 0.2270833 | 2 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_16 | 1 | 16 | -0.007817903 | 0.000137303 | 0.2208333 | 4 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_16 | 2 | 16 | -0.00709294 | -0.0004685387 | 0.2083333 | 6 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_16 | 3 | 16 | -0.007100147 | -0.0004744294 | 0.2270833 | 2 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_60 | 0 | 60 | -0.006605218 | 1.254086e-05 | 0.3791667 | 51 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_60 | 1 | 60 | -0.007055128 | 0.0009000778 | 0.3708333 | 27 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_60 | 2 | 60 | -0.00709294 | -0.0004685387 | 0.4895833 | 35 | NA | bounded_censored |
| branch_B2_PRE_CONTACT_60 | 3 | 60 | -0.00359477 | 0.003030948 | 0.3020833 | 24 | NA | bounded_censored |
| branch_B2_CONTACT_01 | 0 | 1 | -0.008389307 | 0.0003687004 | 0.002083333 | 2 | NA | bounded_censored |
| branch_B2_CONTACT_01 | 1 | 1 | -0.009531775 | -6.396411e-05 | 0.01666667 | 0 | NA | bounded_censored |
| branch_B2_CONTACT_01 | 2 | 1 | -0.009192483 | 0.000247232 | 0.0125 | 1 | NA | bounded_censored |
| branch_B2_CONTACT_01 | 3 | 1 | -0.009220848 | 0.0002132076 | 0.0125 | 1 | NA | bounded_censored |
| branch_B2_CONTACT_04 | 0 | 4 | -0.008389307 | 0.0003687004 | 0.04791667 | 3 | NA | bounded_censored |
| branch_B2_CONTACT_04 | 1 | 4 | -0.009531775 | -6.396411e-05 | 0.06666667 | 0 | NA | bounded_censored |
| branch_B2_CONTACT_04 | 2 | 4 | -0.009192483 | 0.000247232 | 0.0625 | 1 | NA | bounded_censored |
| branch_B2_CONTACT_04 | 3 | 4 | -0.009220848 | 0.0002132076 | 0.0625 | 1 | NA | bounded_censored |
| branch_B2_CONTACT_16 | 0 | 16 | -0.008389307 | 0.0003687004 | 0.1625 | 7 | NA | bounded_censored |
| branch_B2_CONTACT_16 | 1 | 16 | -0.009531775 | -6.396411e-05 | 0.175 | 5 | NA | bounded_censored |
| branch_B2_CONTACT_16 | 2 | 16 | -0.009192483 | 0.000247232 | 0.16875 | 5 | NA | bounded_censored |
| branch_B2_CONTACT_16 | 3 | 16 | -0.009220848 | 0.0002132076 | 0.1729167 | 6 | NA | bounded_censored |
| branch_B2_CONTACT_60 | 0 | 60 | -0.008389307 | 0.0003687004 | 0.275 | 58 | NA | bounded_censored |
| branch_B2_CONTACT_60 | 1 | 60 | -0.009531775 | -6.396411e-05 | 0.3125 | 61 | NA | bounded_censored |
| branch_B2_CONTACT_60 | 2 | 60 | -0.009192483 | 0.000247232 | 0.3 | 44 | NA | bounded_censored |
| branch_B2_CONTACT_60 | 3 | 60 | -0.009220848 | 0.0002132076 | 0.2770833 | 50 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_01 | 0 | 1 | -0.008389307 | 0.0003687004 | 0.002083333 | 2 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_01 | 1 | 1 | -0.009531775 | -6.396411e-05 | 0.01666667 | 0 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_01 | 2 | 1 | -0.009192483 | 0.000247232 | 0.0125 | 1 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_01 | 3 | 1 | -0.009220848 | 0.0002132076 | 0.0125 | 1 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_04 | 0 | 4 | -0.008389307 | 0.0003687004 | 0.04791667 | 3 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_04 | 1 | 4 | -0.009531775 | -6.396411e-05 | 0.06666667 | 0 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_04 | 2 | 4 | -0.009192483 | 0.000247232 | 0.0625 | 1 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_04 | 3 | 4 | -0.009220848 | 0.0002132076 | 0.0625 | 1 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_16 | 0 | 16 | -0.008389307 | 0.0003687004 | 0.1625 | 7 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_16 | 1 | 16 | -0.009531775 | -6.396411e-05 | 0.175 | 5 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_16 | 2 | 16 | -0.009192483 | 0.000247232 | 0.16875 | 5 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_16 | 3 | 16 | -0.009220848 | 0.0002132076 | 0.1729167 | 6 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_60 | 0 | 60 | -0.008389307 | 0.0003687004 | 0.275 | 58 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_60 | 1 | 60 | -0.009531775 | -6.396411e-05 | 0.3125 | 61 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_60 | 2 | 60 | -0.009192483 | 0.000247232 | 0.3 | 44 | NA | bounded_censored |
| branch_B2_LIFT_ONSET_60 | 3 | 60 | -0.009220848 | 0.0002132076 | 0.2770833 | 50 | NA | bounded_censored |
| branch_B2_HOLD_01 | 0 | 1 | 30.10945 | 0.0497631 | 0.01666667 | 0 | 0.005123082 | bounded_censored |
| branch_B2_HOLD_01 | 1 | 1 | 30.10584 | 0.0501545 | 0.01666667 | 0 | 0.01129303 | bounded_censored |
| branch_B2_HOLD_01 | 2 | 1 | 30.0991 | 0.04965331 | 0.01666667 | 0 | 0.01147268 | bounded_censored |
| branch_B2_HOLD_01 | 3 | 1 | 30.1376 | 0.04947969 | 0.01666667 | 0 | 0.0469849 | bounded_censored |
| branch_B2_HOLD_04 | 0 | 4 | 30.25586 | 0.1961681 | 0.06666667 | 0 | 0.01370028 | bounded_censored |
| branch_B2_HOLD_04 | 1 | 4 | 30.25616 | 0.2004661 | 0.06666667 | 0 | 0.01853382 | bounded_censored |
| branch_B2_HOLD_04 | 2 | 4 | 30.2444 | 0.1949521 | 0.06666667 | 0 | 0.02338247 | bounded_censored |
| branch_B2_HOLD_04 | 3 | 4 | 30.27983 | 0.1917096 | 0.06666667 | 0 | 0.0469849 | bounded_censored |
| branch_B2_HOLD_16 | 0 | 16 | 30.80842 | 0.7487289 | 0.2666667 | 0 | 0.1174845 | bounded_censored |
| branch_B2_HOLD_16 | 1 | 16 | 30.83884 | 0.7831499 | 0.2666667 | 0 | 0.07257372 | bounded_censored |
| branch_B2_HOLD_16 | 2 | 16 | 30.78892 | 0.7394753 | 0.2666667 | 0 | 0.1278173 | bounded_censored |
| branch_B2_HOLD_16 | 3 | 16 | 30.79437 | 0.7062476 | 0.2666667 | 0 | 0.3780922 | bounded_censored |
| branch_B2_HOLD_60 | 0 | 60 | 31.9959 | 1.936216 | 1 | 0 | 0.5540521 | bounded_censored |
| branch_B2_HOLD_60 | 1 | 60 | 32.36918 | 2.313487 | 1 | 0 | 0.529014 | bounded_censored |
| branch_B2_HOLD_60 | 2 | 60 | 31.8342 | 1.784748 | 1 | 0 | 0.55506 | bounded_censored |
| branch_B2_HOLD_60 | 3 | 60 | 31.5064 | 1.418279 | 1 | 0 | 0.5634306 | bounded_censored |

## First observed events

Each event cell retains its actual owner. Numeric differences, original qualified tracking triggers, accepted-pad loss, missing pad versus two actual Teacher repetitions, slip and failure are distinct. No chronology or floating-point difference is labelled a unique cause. Full episode UIDs, peak times and exact raw pointers are retained in the bound JSON projection.

| Job / env | Action | Target | q numeric | Pad loss | Qualified q | Repeat pad | Diagnostic drift >0.5mm | Failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| branch_E0_PRE_CONTACT_01 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_PRE_CONTACT_01 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_PRE_CONTACT_01 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_PRE_CONTACT_01 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_PRE_CONTACT_04 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/7:right/thumb | NA | NA |
| branch_E0_PRE_CONTACT_04 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/1:right/thumb | NA | NA |
| branch_E0_PRE_CONTACT_04 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/7:right/thumb | NA | NA |
| branch_E0_PRE_CONTACT_04 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/6:right/thumb | NA | NA |
| branch_E0_PRE_CONTACT_16 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/7:right/thumb | NA | NA |
| branch_E0_PRE_CONTACT_16 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/1:right/thumb | NA | NA |
| branch_E0_PRE_CONTACT_16 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/7:right/thumb | NA | NA |
| branch_E0_PRE_CONTACT_16 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/6:right/thumb | NA | NA |
| branch_E0_PRE_CONTACT_60 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/7:right/thumb | NA | NA |
| branch_E0_PRE_CONTACT_60 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/1:right/thumb | NA | NA |
| branch_E0_PRE_CONTACT_60 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/7:right/thumb | NA | NA |
| branch_E0_PRE_CONTACT_60 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/6:right/thumb | NA | NA |
| branch_E0_CONTACT_01 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_CONTACT_01 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_CONTACT_01 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_CONTACT_01 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_CONTACT_04 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@123/1:right/thumb | NA | NA |
| branch_E0_CONTACT_04 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_CONTACT_04 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_CONTACT_04 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_CONTACT_16 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@123/1:right/thumb | NA | NA |
| branch_E0_CONTACT_16 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_CONTACT_16 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_CONTACT_16 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_CONTACT_60 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@123/1:right/thumb | NA | NA |
| branch_E0_CONTACT_60 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@144/4:right/thumb | NA | NA |
| branch_E0_CONTACT_60 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@143/8:right/thumb | NA | NA |
| branch_E0_CONTACT_60 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@143/8:right/thumb | NA | NA |
| branch_E0_LIFT_ONSET_01 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_LIFT_ONSET_01 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_LIFT_ONSET_01 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_LIFT_ONSET_01 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_LIFT_ONSET_04 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@123/1:right/thumb | NA | NA |
| branch_E0_LIFT_ONSET_04 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_LIFT_ONSET_04 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_LIFT_ONSET_04 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_LIFT_ONSET_16 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@123/1:right/thumb | NA | NA |
| branch_E0_LIFT_ONSET_16 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_LIFT_ONSET_16 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_LIFT_ONSET_16 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_LIFT_ONSET_60 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@123/1:right/thumb | NA | NA |
| branch_E0_LIFT_ONSET_60 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@144/4:right/thumb | NA | NA |
| branch_E0_LIFT_ONSET_60 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@143/8:right/thumb | NA | NA |
| branch_E0_LIFT_ONSET_60 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@143/8:right/thumb | NA | NA |
| branch_E0_HOLD_01 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_01 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_01 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_01 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_04 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_04 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_04 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_04 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_16 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_16 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_16 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_16 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_60 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_60 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | student@901/7 | NA |
| branch_E0_HOLD_60 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_E0_HOLD_60 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | student@913/7 | NA |
| branch_B0_PRE_CONTACT_01 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_PRE_CONTACT_01 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_PRE_CONTACT_01 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_PRE_CONTACT_01 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_PRE_CONTACT_04 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/3:right/thumb | NA | NA |
| branch_B0_PRE_CONTACT_04 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/1:right/thumb | NA | NA |
| branch_B0_PRE_CONTACT_04 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/2:right/thumb | NA | NA |
| branch_B0_PRE_CONTACT_04 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/6:left/index | NA | NA |
| branch_B0_PRE_CONTACT_16 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/3:right/thumb | NA | NA |
| branch_B0_PRE_CONTACT_16 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/1:right/thumb | NA | NA |
| branch_B0_PRE_CONTACT_16 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/2:right/thumb | NA | NA |
| branch_B0_PRE_CONTACT_16 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/6:left/index | NA | NA |
| branch_B0_PRE_CONTACT_60 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/3:right/thumb | NA | NA |
| branch_B0_PRE_CONTACT_60 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/1:right/thumb | NA | NA |
| branch_B0_PRE_CONTACT_60 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/2:right/thumb | NA | NA |
| branch_B0_PRE_CONTACT_60 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/6:left/index | NA | NA |
| branch_B0_CONTACT_01 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_CONTACT_01 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_CONTACT_01 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_CONTACT_01 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_CONTACT_04 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_CONTACT_04 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_CONTACT_04 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_CONTACT_04 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_CONTACT_16 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_CONTACT_16 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_CONTACT_16 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@138/2:right/thumb | NA | NA |
| branch_B0_CONTACT_16 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@138/2:right/thumb | NA | NA |
| branch_B0_CONTACT_60 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@138/2:right/thumb | NA | NA |
| branch_B0_CONTACT_60 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@142/2:right/thumb | NA | NA |
| branch_B0_CONTACT_60 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@138/2:right/thumb | NA | NA |
| branch_B0_CONTACT_60 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@138/2:right/thumb | NA | NA |
| branch_B0_LIFT_ONSET_01 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_LIFT_ONSET_01 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_LIFT_ONSET_01 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_LIFT_ONSET_01 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_LIFT_ONSET_04 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_LIFT_ONSET_04 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_LIFT_ONSET_04 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_LIFT_ONSET_04 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_LIFT_ONSET_16 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_LIFT_ONSET_16 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_LIFT_ONSET_16 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@138/2:right/thumb | NA | NA |
| branch_B0_LIFT_ONSET_16 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@138/2:right/thumb | NA | NA |
| branch_B0_LIFT_ONSET_60 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@138/2:right/thumb | NA | NA |
| branch_B0_LIFT_ONSET_60 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@142/2:right/thumb | NA | NA |
| branch_B0_LIFT_ONSET_60 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@138/2:right/thumb | NA | NA |
| branch_B0_LIFT_ONSET_60 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@138/2:right/thumb | NA | NA |
| branch_B0_HOLD_01 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_01 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_01 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_01 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_04 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_04 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_04 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_04 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_16 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_16 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_16 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_16 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_60 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_60 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | student@920/5 | NA |
| branch_B0_HOLD_60 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B0_HOLD_60 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | student@916/5 | NA |
| branch_B1_PRE_CONTACT_01 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_PRE_CONTACT_01 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_PRE_CONTACT_01 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_PRE_CONTACT_01 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_PRE_CONTACT_04 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/3:right/thumb | NA | NA |
| branch_B1_PRE_CONTACT_04 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/7:right/thumb | NA | NA |
| branch_B1_PRE_CONTACT_04 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/2:right/thumb | NA | NA |
| branch_B1_PRE_CONTACT_04 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/7:right/thumb | NA | NA |
| branch_B1_PRE_CONTACT_16 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/3:right/thumb | NA | NA |
| branch_B1_PRE_CONTACT_16 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/7:right/thumb | NA | NA |
| branch_B1_PRE_CONTACT_16 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/2:right/thumb | NA | NA |
| branch_B1_PRE_CONTACT_16 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/7:right/thumb | NA | NA |
| branch_B1_PRE_CONTACT_60 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/3:right/thumb | NA | NA |
| branch_B1_PRE_CONTACT_60 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/7:right/thumb | NA | NA |
| branch_B1_PRE_CONTACT_60 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/2:right/thumb | NA | NA |
| branch_B1_PRE_CONTACT_60 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/7:right/thumb | NA | NA |
| branch_B1_CONTACT_01 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_CONTACT_01 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_CONTACT_01 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_CONTACT_01 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_CONTACT_04 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_CONTACT_04 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_CONTACT_04 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_CONTACT_04 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_CONTACT_16 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@131/6:right/index | NA | NA |
| branch_B1_CONTACT_16 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/5:right/index | NA | NA |
| branch_B1_CONTACT_16 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/4:right/index | NA | NA |
| branch_B1_CONTACT_16 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/5:right/index | NA | NA |
| branch_B1_CONTACT_60 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@131/6:right/index | NA | NA |
| branch_B1_CONTACT_60 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/5:right/index | NA | NA |
| branch_B1_CONTACT_60 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/4:right/index | NA | NA |
| branch_B1_CONTACT_60 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/5:right/index | NA | NA |
| branch_B1_LIFT_ONSET_01 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_LIFT_ONSET_01 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_LIFT_ONSET_01 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_LIFT_ONSET_01 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_LIFT_ONSET_04 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_LIFT_ONSET_04 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_LIFT_ONSET_04 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_LIFT_ONSET_04 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_LIFT_ONSET_16 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@131/6:right/index | NA | NA |
| branch_B1_LIFT_ONSET_16 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/5:right/index | NA | NA |
| branch_B1_LIFT_ONSET_16 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/4:right/index | NA | NA |
| branch_B1_LIFT_ONSET_16 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/5:right/index | NA | NA |
| branch_B1_LIFT_ONSET_60 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@131/6:right/index | NA | NA |
| branch_B1_LIFT_ONSET_60 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/5:right/index | NA | NA |
| branch_B1_LIFT_ONSET_60 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/4:right/index | NA | NA |
| branch_B1_LIFT_ONSET_60 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@132/5:right/index | NA | NA |
| branch_B1_HOLD_01 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_HOLD_01 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_HOLD_01 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_HOLD_01 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_HOLD_04 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_HOLD_04 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_HOLD_04 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_HOLD_04 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_HOLD_16 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_HOLD_16 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | student@891/2 | NA |
| branch_B1_HOLD_16 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_HOLD_16 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B1_HOLD_60 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | student@895/7 | NA |
| branch_B1_HOLD_60 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | student@891/2 | NA |
| branch_B1_HOLD_60 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | student@898/4 | NA |
| branch_B1_HOLD_60 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | student@906/2 | NA |
| branch_B2_PRE_CONTACT_01 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_PRE_CONTACT_01 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_PRE_CONTACT_01 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_PRE_CONTACT_01 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_PRE_CONTACT_04 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/3:right/thumb | NA | NA |
| branch_B2_PRE_CONTACT_04 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/1:right/thumb | NA | NA |
| branch_B2_PRE_CONTACT_04 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/2:right/thumb | NA | NA |
| branch_B2_PRE_CONTACT_04 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/6:left/index | NA | NA |
| branch_B2_PRE_CONTACT_16 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/3:right/thumb | NA | NA |
| branch_B2_PRE_CONTACT_16 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/1:right/thumb | NA | NA |
| branch_B2_PRE_CONTACT_16 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/2:right/thumb | NA | NA |
| branch_B2_PRE_CONTACT_16 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/6:left/index | NA | NA |
| branch_B2_PRE_CONTACT_60 / 0 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/3:right/thumb | NA | NA |
| branch_B2_PRE_CONTACT_60 / 1 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/1:right/thumb | NA | NA |
| branch_B2_PRE_CONTACT_60 / 2 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@122/2:right/thumb | NA | NA |
| branch_B2_PRE_CONTACT_60 / 3 | student@120/pre | student@120/1 | student@120/1 | teacher_prefix@95/7:right/index | NA | student@121/6:left/index | NA | NA |
| branch_B2_CONTACT_01 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_CONTACT_01 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_CONTACT_01 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_CONTACT_01 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_CONTACT_04 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_CONTACT_04 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_CONTACT_04 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_CONTACT_04 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_CONTACT_16 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@132/3:right/index | NA | NA |
| branch_B2_CONTACT_16 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_CONTACT_16 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_CONTACT_16 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_CONTACT_60 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@132/3:right/index | NA | NA |
| branch_B2_CONTACT_60 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_CONTACT_60 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_CONTACT_60 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_LIFT_ONSET_01 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_LIFT_ONSET_01 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_LIFT_ONSET_01 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_LIFT_ONSET_01 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_LIFT_ONSET_04 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_LIFT_ONSET_04 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_LIFT_ONSET_04 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_LIFT_ONSET_04 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_LIFT_ONSET_16 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@132/3:right/index | NA | NA |
| branch_B2_LIFT_ONSET_16 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_LIFT_ONSET_16 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_LIFT_ONSET_16 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_LIFT_ONSET_60 / 0 | student@122/pre | student@122/1 | student@122/1 | teacher_prefix@95/7:right/index | NA | student@132/3:right/index | NA | NA |
| branch_B2_LIFT_ONSET_60 / 1 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_LIFT_ONSET_60 / 2 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_LIFT_ONSET_60 / 3 | student@123/pre | student@123/1 | student@123/1 | teacher_prefix@95/7:right/index | NA | student@133/1:right/index | NA | NA |
| branch_B2_HOLD_01 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_01 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_01 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_01 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_04 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_04 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_04 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_04 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_16 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_16 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_16 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_16 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | NA | NA |
| branch_B2_HOLD_60 / 0 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | NA | NA | student@901/1 | NA |
| branch_B2_HOLD_60 / 1 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | student@909/5 | NA |
| branch_B2_HOLD_60 / 2 | student@876/pre | student@876/1 | student@876/1 | teacher_prefix@95/7:right/index | NA | NA | student@900/7 | NA |
| branch_B2_HOLD_60 / 3 | student@875/pre | student@875/1 | student@875/1 | teacher_prefix@95/7:right/index | student@934/1 | NA | student@892/7 | NA |
