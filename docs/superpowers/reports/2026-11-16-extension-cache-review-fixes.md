# 扩展接口候选复审修正

只读复审复现两项Important：prepare完成frame持有借用App；缓存删除失败后允许重复unload。
已先新增失败回归（4failed/6passed），修复三个prepare的finally借用参数清理，
并使用独立unload_attempted作为sticky一次性标记，删除失败后亦不重复调用原unload。
保留framework/getter闭包合法生命周期；没有缓存原生IExtensions对象到frame或receipt。
新10项缓存测试及既有launcher/window合计29 passed，完整CPU结果由Registry发布记录登记。
当前候选尚未消耗本批最后一次原生预算；剩1，须发布/同步/冻结后才运行。
未修改SDK安装、物理、模型、成功标准；PR #2 Draft。
