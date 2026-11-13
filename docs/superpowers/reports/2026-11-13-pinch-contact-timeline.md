# 最新 Pinch 真实接触账本：逐子步方向与连续时间

基线6f58f8f220eee120461e6e56999f88a2ce62b975。本包只读已完成的234步原生账本，没有新原生、Isaac实例、GPU任务或PPO更新。原failed scheduler receipt不变；自然退出的独立复审范围不扩张。

## 实际修改

在已有tabletop_deformable_pair_source_basis.py中增加ContactTimeline，复用analyze_rod_pair_basis、原始pair/独立节点合力/mesh wrench校验与compare_contacts，没有重写第二套接触判据。输入必须绑定同一env/episode/reset/rod、连续SDK/solver/policy时钟；每段完整验证后才原子保留。缺失packet为unknown并中断连续区间；新episode必须新建实例。输出深拷贝，不允许调用方篡改已保留账本。

每个solver子步保存各原pad载荷、杆侧世界力、实际节点ID、相向符号候选、完整比较、原边界时间，以及明确标注的SDK子步重建时间。重建时间不是新增的solver原生getter。统计Pinch/Tripod/Power各自所需指腹，不能把Pinch两指记录替代其他抓法。

脚本replay_dh116_pinch_contact_timeline.py核验冻结receipt、postexit audit、完整源JSONL、实际半径参数与contact.cuh来源，重新对齐每个24D动作和physical readback；最多232原测量行、120秒CPU。保留2784行独立timeline，不改原数据和R0。相向符号候选始终不授予effective_opposed_contact、stable、material_anchor_slip或Teacher资格。

## 实测历史数据重算

原始ledger SHA170b9bc8090eeb3f4291ce4102335eabbcb1d49892e333d1aa5b4ac4d9585fdf。
输出/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-contact-timeline-20261113-4ja1oiUi/run01。
result.json SHAc1ffd68c418fad580860f20e47cff777ce9e8b321b49432d16dbd933281be3e8。
timeline.jsonl SHA84be2ff56987b3485a7bf67d3d2ba621c53a70182596c27a9ce871707e287a67。
CPU8.267453秒，exit0，native_imports=[]；232个原480Hz测量步，2784个原solver子步，495个原左右指腹内比较。

| 实际观察 | 子步数 | 连续时间 | 起始SDK / solver子步 / solver tick |
|---|---:|---:|---|
| 左拇指载荷 |526|91.319444ms|191 / 2 /2259|
| 左食指载荷 |495|85.937500ms|193 / 9 /2290|
| 左两指同时有载荷且相向符号成立 |495|85.937500ms|193 / 9 /2290|
| 右两指同时载荷 |0|0|无|
| 双手同时所需指腹载荷 |0|0|无|
| 杆—桌面载荷 |2784|483.333333ms|3 / 0 /1|

所有左相向样本同一episode dual-pinch-a3c7e980097149789f15b74830bccac0:one、env0、reset1；首条policy23/substep6、SDK193、solver子步9，持续到原SDK234尾部。统计区间仍在尾部开放，不声称接触随后丢失或再持续。

495次左侧比较：normal dot范围[-0.778247,-0.773337]，force cosine范围[-0.996913,-0.761050]；两侧向内normal投影的最小值范围[0.479039,0.486223]，向内力最小值范围[0.016602,0.234087]N。因此此短前缀确有同子步方向相向的两指接触，不能再笼统说缺少方向数据。

但全部495对接触发生在相隔12.500003mm的不同材料节点，same-node比较为0；不能称为同一材料横截面、固定锚或零滑移。右手没有对应比较。桌面全程承托。该证据不是5秒保持、30mmLift、0.5mmSlip成功，也没有证明有效力闭合/真实稳定性。

## 验证和限制

先新增测试产生缺ContactTimeline的collection失败；另有调用方修改结果污染内部记录的RED，修复为独立深拷贝后通过。
100 passed，0failed，3.43s：
```sh
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 bash scripts/python.sh -m pytest -q tests/test_dh116_contact_timeline.py tests/test_deformable_pair_source_basis.py tests/test_dh116_flexible_pair_contact_diagnostics.py tests/test_dh116_deformx_pair_capture.py tests/test_dual_hand_rod_runtime.py tests/test_dh116_segment_r0_replay.py
```
日志/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/contact-timeline-full-20261113.log，SHAa346707b66c1fe3e33098519292e837ad3138af738322da29e4f33c07a0bb6a5。
执行命令：python -m scripts.replay_dh116_pinch_contact_timeline --output <独立不存在目录>。
没有对原生退出139/-11作新运行推断。当前234步的退出0仍来自历史独立复审。

下一步：右侧几何约束的非光滑来源已定位为有余量的SDF方向项；另行准备保留完整非线性验算的候选求解。R0正向对向接触、稳定性、材料滑移与生产系数仍需单独资格化。本包不解除Teacher0/3的准入阻断。528保护文件及两个历史checkpoint由集成发布器复核；Windows发布，PR2保持Draft。
