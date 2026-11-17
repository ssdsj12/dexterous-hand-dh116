# 截图建议核验：释放接口、CUDA 初始化与34个 cooked hull

## 本轮范围与结论

基线为11-17分支 ad1fd01862871c414b4a5acba76c61bb7e9c2304。本轮只读取真实安装源码、运行CPU隔离测试并复用最新原生快照做离线审计：新增Isaac实例、CUDA初始化、物理/控制/求解器步、Teacher查询和PPO更新全部为0。

截图是待核验建议，不是已经证明的根因或可直接执行的改造指令。可取的是核验真实持有链、明确CUDA设备/初始化边界、继续审计原始cooked数据；不能据此宣称多线程冲突、坏网格或empty_cache已经能够解决SIGSEGV。

## 建议逐项判定

| 建议 | 本机核验 | 判定 |
|---|---|---|
| 检查SimulationView/backend真实所有权与退出顺序 | 既有原生证据已确认STOP后manager为None，但两手Articulation/Data的同一view/backend仍存活；本轮测试确认invalidate只转发backend.invalidate并保留Python字段 | 可取；保留对象不等于已证明SIGSEGV因果，不直接删除SDK缓存字段 |
| 调用self.backend.release并设None | 实际Python wrapper字段是_backend；公开wrapper没有release/close方法；截图的backend分支在实际wrapper形状上不进入 | 不直接采用。底层pybind是否另有release入口仍未资格化，不能编造或用mock接口冒充 |
| empty_cache之后synchronize即可释放仿真 | 真实empty_cache函数只调用PyTorch allocator入口，未释放Python scene/view所有权。真实synchronize函数首先调用_lazy_init | 不能作为PhysX释放或自然退出证明；不在app.close之后追加这组调用 |
| 入口统一CUDA初始化 | 显式设备和初始化边界值得做独立对照，但截图的torch.device("cuda")无index，在当前set_device调用中实际抛ValueError，且没有进入原生setDevice | 可取方向，代码不能照抄。当前CUDA_VISIBLE_DEVICES=2合同下应核对逻辑cuda:0与物理GPU2；本轮没有改变实际初始化顺序 |
| CUDA帧通常说明DataLoader、多线程/多进程初始化冲突 | 原生只观察到上游_lazy_init帧；本轮直接调用实际_lazy_call的CPU路径，保存的是格式化字符串列表，非活traceback | 当前根因未证明；不能把帧名当成线程冲突证据，也不能认定_queued_calls是活帧持有者 |
| 清理或重新导出34个.stl/.obj | 原生getter返回的是34个USD owner的stored cooked表示，不等同于34个可直接替换的源STL。全部组合拓扑检查通过，严格平面检查仍失败，且两次原生raw hull完全一致 | 保留只读审计；没有依据修改资产、焊点、简化、重cook或放宽contactOffset/restOffset |

当前已安装SDK只读源码同时存在tensors.reset入口；这不证明它可以任意替代生命周期释放。本轮未调用全局reset、native invalidate/release或任何未核实的销毁方法。

## 实际测试

新增 tests/test_teacher_cleanup_suggestions.py，共8项：

1. 从实际已安装api.py抽取纯Python的SimulationView构造/invalidate方法，确认_backend字段与公开方法集合。
2. invalidate之后引用仍在，失效和释放不混淆。
3. 截图backend分支不命中真实wrapper字段。
4. 执行真实torch.cuda.empty_cache，mock底层allocator，确认没有发生scene释放调用。
5. 执行真实torch.cuda.synchronize，mock原生入口，确认先进入_lazy_init。
6. 实际set_device(torch.device("cuda"))在原生入口之前拒绝缺少index。
7. 显式cuda:0向被mock的入口传0，不把逻辑卡号当成物理卡号证明。
8. 实际_lazy_call在CPU路径存字符串stack，而非活frame/traceback。

所有GPU/native入口均为CPU spy或fake backend；它们不能证明真实PhysX backend是否支持release，不能证明GPU同步、原生释放、自然退出或抓取成功。每个用例检查CUDA未初始化且未导入Isaac/IsaacLab。真实SDK方法只在内存中通过AST抽取，不修改或分发SDK源码。

定向8 passed；联合已有owner/view/cleanup/cooked-readback回归60 passed，0 failed，0 skipped。准确命令、环境和日志SHA在readiness中。该测试文件需显式DH116_INSTALLED_SDK_AUDIT=1；未启用时明确skip，调度器不应把全部skip当成通过。

## 最新原生快照的离线复验

沿用现有 audit_dh116_teacher_cooked_readback.py，对最新teacher-owner-probe-native-20261117T110429Z快照重新运行，不重新启动Isaac：

| 指标 | 结果 |
|---|---:|
| 实际owner / hull | 34 / 34 |
| 唯一body身份 | 34，无重复 |
| 组合面片/索引/边/连通/顶点环检查 | 34/34通过 |
| 原严格完整校验 | 0/34通过 |
| 平面残差拒绝 | 189/1890个面 |
| SVD诊断非平面面片 | 188 |
| 全部顶点使用、正体积、存储法向绕序检查 | 34/34通过 |
| 自交及空间凸性资格 | 未证明 |
| 资产/plane/容差修改 | 0 |

SVD只作诊断，没有回写平面或几何。188个非平面面片和PhysX合并面的表示语义值得继续核对；另一个略超容差的面也不得静默放宽。原有189个失败保留。

## 发布与下一步

只增加本轮验证测试、报告、readiness与发布/Registry摘要。原控制、物理、安装SDK、模型、528个受保护文件和原生成功标准均不改变。PR2保持Draft。

下一步应继续定位真实_lazy_init帧的上游持有者。主线程、AppLauncher之后而SimulationContext构造之前显式初始化可作为单变量候选，但尚未实现或做原生验证；不得宣称它已修复SIGSEGV。原生预算已耗尽，本轮没有复用。Teacher准入仍blocked。
