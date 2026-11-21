# 三指真实 Reset manifest 接入

基线 b489922383d1c25cdb6de5cb1319047676133ca3，Ubuntu开发/Windows同commit发布。

新增 tabletop_tripod_reset_manifest.TripodResetManifest、seal_manifest、load_tripod_reset_manifest，
以及实际入口 scripts/prepare_dh116_tripod_teacher_reset.py。已运行入口生成服务器私有manifest，
从上轮自然退出0的17槽真实Reset逐字段恢复初始状态，不以ready/qualified布尔值授权。
当前旧Pinch-only live/static gate尚未修改，三指PPO仍待独立Adapter和live gate。

## 来源与隔离
- 原始outer/GDB/冻结plan/全部artifact SHA、2步初始化预算、自然关闭与资源释放必须有效。
- 原backend profile、显式Reset实现、17槽mesh plan中的原材料/资产/SDF/额外掌面/近节来源仍核对当前真实字节。
- 历史原生回执证明其本轮Reset；新训练consumer不是当时的物理producer，不能借历史回执声称新训练代码已运行。
- 每次加载从raw数值审计、直接mesh读回、371槽common source packets、左右原34形状+10包络重建；
  Caller即使重算manifest SHA，也不能改杆坐标、task config、抓法、左右来源和保护阈值。
- 限定Tripod17，41个固定材料节点。66项common有效，305项未知仍为NaN/false；不开放R2/ForceProxy。
- Manifest仅在Git外保存，原始约199MB raw/SDF展开、模型和Checkpoint未上传、未覆盖。

## 实测
- TDD：4个用例先因缺功能失败，再通过。manifest4通过/0失败，51.57s。
- 原Pinch preflight和原生回执审计回归66通过/0失败，39.94s。
- 合计70项不重叠选定CPU案例；并非全仓或新原生仿真。
- 已执行prepare入口，实际manifest路径/SHA、源指纹和日志SHA见readiness。
- 本包新增原生0、PPO0、抓取成功0；上轮Reset证据只按历史来源引用。
- Antigravity本轮界面仍显示额度耗尽，无新审查回复；独立核验继续。

下一步：独立三指六指腹Reset/策略距离与七路线SourceAdapter，接原有Context/R0/PPO，再做live gate。
原二指reader、三种相机/模型合同、原24D/371D、关节保护和30mm/5s/0.5mm标准不变。

Registry：60通过、0失败，13.97s。528保护文件与3份历史best.pt校验保持原SHA。
