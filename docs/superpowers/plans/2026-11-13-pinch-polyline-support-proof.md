# Pinch 逐杆段三维间隙证明实施计划

> 按 executing-plans、systematic-debugging 和 test-driven-development 执行；用户持续授权不重复询问。

目标：解释234步配方在SDK216的保守几何失败，修复忽略Y方向与整条杆凸包过度保守的问题，不降低任何原保护阈值。
基线35d492d；原生attempt-uoa3kszq已耗尽，不重跑。

设计：保留原XZ全节点下界。独立模块对每个实际相邻杆节点线段建立单位法向支撑平面证书，使用完整source vertices验证投影；每段取可证明下界，全部段取最小，再减原5mm半径与2um裕量。
法向候选为原64XZ方向、XYZ轴、实际凸包面法向和形心至线段中点方向。任何候选只提供证明方向，不以凸包近似距离冒充原表面距离。未能证明仍阻断。实际手-桌接触NA不变，真实接触由原rod pair判断。

- [x] 添加解析盒体、弯杆、穿越、切触、全部原顶点核验、证书篡改、退化输入的失败测试；运行并保留日志。
- [x] 新增tabletop_polyline_support_certificate.py，certify()/validate()均检查完整顶点与原相邻节点，正值才能证明隔离；旧源码不改。
- [x] 新增离线原生账本审计，完整读取214测量行，锁SHA、同env/episode/step，列出四指腹初始净空、实际接触首末步及非目标失败证书。
- [x] 原始216步失败与新CPU证明分别报告；不把离线重算升级为原生通过，不重写原receipt。
- [ ] CPU回归、保护528文件与2Checkpoint、readiness/Registry更新、Ubuntu提交并Windows发布。
- [ ] 后继新原生配方仅在所有证明和回归通过后独立冻结；相同动作/Reset/物理，仅独立间隙实现；至多234步/180秒/0PPO，不复用耗尽批准。
