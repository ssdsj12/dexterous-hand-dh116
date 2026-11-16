# 固定双相机清理两次复验：采集通过边界、自然退出失败

本批用户明确批准最多2次，GPU0单实例串行、320×240、零PPO。
第一轮44cff046、合同9e2ac7e78abd3592c753bf99eba0458bfd277c2c8f4d7bce8c1a2ac3402f6086，
167.823792秒，17对图像，44render，0阶段错误，正常app.close marker存在。
原左右annotator失效错误未复现，清理顺序修改实际通过该边界；
最终仍_extensions.so+0x1e445经carb/Py_FinalizeEx SIG11，外层255。

第二轮d58af2d6a92499efe4828aa0369d57d837c2500e，
合同dfe8e5c4880ef5ddc5967d2cbe09bb8fed20c5f6b1743d349948a4bceb64f608，
177.322637秒，同样17对图像、44render，0采集/解绑错误。
但扩展cache候选在prepare阶段拒绝：existing IExtensions cache required。
这只能说明现有对象未匹配假设（缺少或类型不符），不能据此证明缓存是根因；
代理未安装，没有释放缓存，最终依旧SIG11。
本轮Managed关闭因preparation guard而报错，缺少app_close_returned.json；
不能以标准SDK调用可能已返回代替独立close marker和自然退出资格。

两次合计260SDK步、3072测量solver子步、88render、345.146429秒，零PPO。
34/34左右raw帧号、render_frame_number、simulation_time相等；
34份真实41-node wire快照有限且和RGB物理/rod tick对齐。
杆在本段接近静止，来源正确仍不等于柔性弯曲/双向接触/抓取已通过。
首帧右图噪声及双目深度精度尚未资格化。

## 修正候选的默认行为

既然原生前提未匹配，不能默认让未知cache给close新增异常。
prepare_loaded_extension_cache改成默认只读audit：记录cache_present和cached_type字符串，
不调用getter、不删除缓存、不替换framework，也不授予native资格；
只有显式apply=True实验才进入原缓存释放候选，旧实验代码和证据保留。
新增2项失败测试先RED后GREEN，定向31项通过；完整回归由Registry登记。
SDK来源SHA仍校验，日志I/O失败仍在标准close之后报告，保护原生错误语义。

## 证据、发布和后续

私有原件：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/live-stereo-preview-20261116/fixed-dual-cleanup-batch2-20261116。
第二轮日志SHA35991f034ba49f7d46de1a467fb076100dce464d6e4748b9db296a63ab20dd3d。
保留完整PNG、wire JSON、物理账本、GDB栈/映射/退出码，不上传原件或模型。
528保护文件、2历史Checkpoint及原物理/动作/模型不改，PR #2 Draft，经Windows发布并同步原目录。

本批预算剩0，不再原生重跑。下一步先用只读probe确认实际cache对象类型与真实引用所有者，
不能继续猜测清理其他SDK全局变量；窗口/接口和真正自然退出仍需新来源证据。
相机需首帧稳定与对应精度验证，wire需受控形变/接触测试，Teacher/Student准入保持关闭。
