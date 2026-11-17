# 左拇指退出隔离变体的 CPU 预检

首次原生运行保留 240 行，接触样本通过且 `app.close()` 返回，外层退出 139。[首轮报告](2026-11-12-deformx-left-thumb-contact-native.md)

本次唯一实验假设是 probe 持有的 Python／SDK 引用在 finalization 前仍存活。复用已通过的 F3 清理顺序，在现有账本落盘、composer 重置和 CUDA synchronize 之后显式放弃本 probe 的手、view、composer、RopeBatch 和本地闭包引用。STOP 返回且 SDK manager view 已释放后，再回收引用环并记录弱引用状态，然后调用原有 close。启用 F3 同款 Python faulthandler 以保留失败栈。未变更源 USD、原关节参数、SDF、动作、240 行预算、接触判据或反作用力过程。

预检命令沿用首轮报告的入口和同一锁定输入，只换独立输出目录。运行授权与单实例调度到位前保持 CPU 预检状态。新的 Python 循环引用测试确认 GC 发生在 ownership checkpoint 之前。OpenUSD 回归 12 项通过，裸解释器拇指和生命周期测试 6 项通过。**尚未运行这个变体，未取得左侧外层退出资格或右侧接触结果。**

| 证据 | SHA256 |
|---|---|
| no-app 源身份预检 | `8466beb3d5341a769d535bd876e4814e8e0b96ac4e499dadade38c3f3bd34960` |
| 12 项 OpenUSD 回归 | `cf8e91f5ca56ee5809498c3c017582903100838996f7a05912be0cbc97c9f1f9` |

独立证据 `/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flex-a-thumb-lifecycle-v2-preflight-zX2lmJ`。528 个保护文件和 2 个历史 Checkpoint 通过 SHA 核验。
