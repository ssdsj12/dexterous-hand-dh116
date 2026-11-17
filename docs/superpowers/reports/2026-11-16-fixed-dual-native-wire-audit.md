# 固定双相机原生与wire核验

实际运行source b22a788828da7123db31311481b5a2464dc448b2，
合同7422ff12e34df883b26443000ac2f7e9a191726184436b4254e3659492e0d48f。
用户“启动isaac并修改；继续测试完相机，然后核验wire是否正确”范围内执行一次独立短预览。
GPU0共享、单实例、1环境、零PPO，130SDK步/1536测量solver子步，151.986秒；预算耗尽无重试。

## 相机实际证据

固定左右相机均完成17张图像，17/17左右raw frame_id、render_frame_number、simulation_time相等，
整个采集没有相机select或物理步穿越。逐帧原生USD位姿及标定readback保护均未报错。
这验证固定双路同render时钟边界，但不等于双目深度或像素对应精度通过。
目视检查原始000_left/right与013_right：右首帧有强噪声，第13帧清晰。
首帧没有因“困难图像”从分母删除；相机全帧图像质量尚未通过。

## wire实际证据

17份wire.json来自同一真实ropesim实例、同一backend profile，
每份3×41节点、40材料方向，数据有限且和RGB物理步/rod tick对齐。
rod tick从0到1536递增；半径0.004999999888m，中心线长度0.5m。
后端binary SHA 8817010163946945495c987923d11f3ade4fbd52482a83bbdd48f017d3149340，
源树SHA 34741b437191fd66bf151331f20476a86f8708414dcc2782a8c69dd15b0efb64；
运行入口先核验实际导入binary及Profile，再从完成边界节点发布CosseratWire mesh。
raw身份校验链确认是求解器节点进入显示，未把刚体root pose代替节点。

但本段节点最大位移仅8.948454e-8m（0.0000894845mm），二阶节点差最大1.49e-8m。
配置声明无固定节点、重力-9.81；本次杆几乎静止，不能由此判定柔性弯曲/自由动力学/双向接触合格。
需要独立受控位移或接触加载后检查形变和反作用；尚未执行。
杆实例和源码通过核验不等于Teacher抓取成功。

## 失败与修复候选

采集后关闭右相机时extra annotator与rgba.detach报Invalid NodeObj，
外层255、子进程SIG11，GDB libomni.usd.so→UsdManager.destroyContext。
本次不是上一批Py_FinalizeEx扩展绑定故障；自然退出仍不通过。
当前原生顺序先detach/destroy左路，再detach右路，怀疑共享图依赖导致后者失效。
本包把资源释放拆成detach阶段：先解绑两路，再销毁任一路render product。
一条顺序回归先失败后通过，完整CPU313 passed/0 failed，日志SHA
babcc3f587234e0ebb849a5be357ffa408823212ef7c9de1ca7cca52f70c2aab。
此清理修正尚未再次原生运行，不能宣称已解决SDK崩溃。

## 原件与下一步

私有证据目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/live-stereo-preview-20261116/fixed-dual-native-20261116/attempt-1。
保留PNG、17份wire快照、physical账本、GDB栈/映射与退出码；原件不上传Git。
本轮短预览预算1次已用完，没有隐含重试。
下一次冻结修正SHA后验证清理；相机还需首帧渲染稳定性和双目对应/深度精度，
wire需独立动态/接触实验。Teacher/Student与抓取资格保持关闭。
528保护文件、2历史Checkpoint核验不变，经Windows发布，PR #2保持Draft。
