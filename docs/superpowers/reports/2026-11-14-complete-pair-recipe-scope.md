# 完整接触配方的源码/受保护资产分类修复

发布基线 `3401b3477033178c3f234a226ab7de926a801805`；源码工作基线 `74138372b2ab329869e2e1fdbe914af5721f5b23`。

原生启动前，scheduler.native_recipe→release_path拒绝被错误加入source_files的USD资产。没有创建approval、没有启动Isaac、没有消费预算。原生失败或自然退出结论不适用。

原因：生成器把preflight.sources.local_sources全部并入可发布源码；该映射也包含只读资产。改为旧已核验配方source_files与独立entry.FILES、新路径回归的并集。旧inputs逐字典相同，USD/Checkpoint/原始数据不复制、不修改、不发布；源preflight仍检查自己的完整几何资产指纹。

新增测试先复现release_path失败，随后验证全部发布路径、旧输入锁、预算、父回执不变。原生worker/审计/原fast/live worker回归 **40 passed in 3.82s**。

日志位于私有evidence/complete-pair-r0-20261114/recipe-scope-red.log及recipe-scope-green.log。旧失败清单保留在前一提交。

实验代码/配置SHA保持 `fc37cc7022bff89791f139f473cba270d5d12cb3ae42943b9b6dfecfb53e6560`；新的配方文件SHA `1bdb1e4adc9d29d2ec3a1b34875ab5a7a84ee79358225e7ff0e033e34f867f60`。scheduler指纹发布后重新生成，不复用未成功生成的旧批准。

下一步同原234步/180秒/1env/GPU3/0PPO的新来源诊断；非预算扩展、非已消费重试。Teacher/fullR0/positive contact仍不晋级。