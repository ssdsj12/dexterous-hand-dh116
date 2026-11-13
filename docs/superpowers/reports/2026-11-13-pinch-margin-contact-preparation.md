# 左 Pinch 最大净空初态：独立接触配方准备

基线 e7c8e459aa262053521d1d5bf3e4f56d624881a0。此报告仅为代码与 CPU 准备，新增原生步数0，PPO更新0，Teacher仍0/3。

## 实际改动

- prepare_dh116_pinch_margin_contact.derive_pair/load_pair：SHA锁定既有左右Reset和最大净空离线报告；仅替换左手根位姿、六主动关节与对应原Mimic展开。右手、自由杆41节点、桌面、重力、材料和执行器保持原值。新配置逐字对照派生结果，禁止静默改动。
- action_schedule：29个60Hz动作窗口，2初始化+48静置+160闭合+24保持=234个480Hz总物理步。只有左手动作索引1/2非零，分别0.1855539082010854/0.234063279676594。没有腕部指令和PPO。
- scenario_scope：复用现有双手mounted owner、DeformX参数快照、director端点、逐接触账本及正常clear/STOP/app.close。异常离开后恢复原路径。原2+8+8配方及其全部源文件未改。
- dh116_pinch_margin_contact_audit：将既有完整账本、同钟、自然退出和实际参数审计限定到独立234步范围。保留旧检查，AST回归约束预算差异；18步旧回执不能通过新范围。
- 新专用worker绑定真实最近参数任务回执 SHA f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901。源码/配置SHA be2863aa0e00f87d1115f1458466c61df71bc12826fb842964bef2fad806b677。调度器仍单实例、1环境、GPU3、180秒、一次、不重试。
- 2.1mm逐步桌面保护、关节/Mimic/速度/wrist保护不变；增加指腹SDF最小值-0.3mm上限。几何不是实际接触；手-桌面Tensor通道NA保持NA。

## 几何依据

静态路径四个原相交指腹桌面净空下界：finger14 2.986263mm、finger23 2.824066mm、finger43 55.877937mm、finger53 73.364362mm。全手最低为finger13 2.111213mm。左根由(.4974294575,.2430690315,.4171107820)变为(.4953694955,.2430690315,.4183242614)m，绕杆轴角-0.0182749286rad。不是动态安全保证，约11.213µm余量必须实测保护。

## 验证

Ubuntu CPU最终75通过、0失败，178.96秒。命令：

`DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_pinch_margin_contact.py tests/test_dh116_pinch_margin_contact_worker.py tests/test_dh116_pinch_margin_contact_owner.py tests/test_dh116_native_parameter_task_worker.py tests/test_dh116_measured_closure_geometry.py`

日志 /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-margin-full.log。完整Python owner的CPU替身测试确实跑完234计数，并验证第11步捕获失败时停止、末步保留、正常调用close；不是Isaac原生证据。专用子进程测试同时验证审计拒绝不会生成通过回执。

开发失败保留在 pinch-margin-contact-red.log、pinch-margin-worker.log、pinch-margin-owner.log：缺失模块红测；资产发布清单、测试替身接口和fixture导入错误已修正。尝试凸包极点加速在原始网格上未减少28893个顶点，已撤回；最终沿用原复算，未改变碰撞网格。

## 后续与门槛

先经Windows发布同一提交、重新核验远端，再消耗一次新234步/180秒授权。必须分别报告真实接触、保护终止和自然退出。新配方尚未执行，不授予Teacher、抓取、无桌面支撑或材料滑移资格。R0实际hold/instability及完整371D来源仍未齐备。528保护文件和2历史Checkpoint保持。
