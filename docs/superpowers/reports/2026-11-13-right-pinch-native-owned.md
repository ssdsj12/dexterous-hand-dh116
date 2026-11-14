# Right Pinch 独立原生入口与退出校验

## 实际修改

- `scripts/prepare_dh116_right_pinch_native_interval.py`：新作用域绑定右侧已冻结初态、29个原24D动作与232步测量；使用原参数/资源owner。退出作用域恢复所有旧模块绑定。
- `scripts/verify_dh116_right_pinch_owned_contact.py`：验证真实已通过参数任务父回执及SHA；冻结右侧配置、代码、静态证据和后端；专用worker标识。
- `scripts/run_dh116_right_pinch_owned_work_packages.py`：原单实例锁、GDB、自然退出、独立post-exit审计；拒绝234步/1env/0update/GPU3/180秒之外的配方。
- `scripts/validate_dh116_right_owned_contact.py`：原退出校验体仅替换独立operation导入，AST等价测试保护所有原检查。
- `configs/dh116_execution/deformx_right_pinch_owned_contact_v1.json`：新配方，不覆盖旧配方。
- `tests/test_dh116_right_owned_worker.py`、`tests/test_dh116_right_owned_owner.py`：真实CPU子进程重入、失败不晋级、版本/Reset身份、作用域恢复、完整测量和terminal保存。

## 修复中实际失败

首次CPU联调4失败/11通过。原因是给旧margin模块临时设置新VERSION后，其原始初态重新推导的版本发生变化，导致冻结SHA/内容检查拒绝，尚未启动Isaac。修复采用独立validator；旧margin.VERSION与load_pair均不变。原失败日志：
`/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-owned-worker-green01-20261113.log`。
首次缺入口RED日志为同目录`right-owned-worker-red-20261113.log`。均保留。

## CPU与原生证据边界

专项worker 15项通过，包括实际子进程，但子进程注入CPU接口，不计原生。完整owner三项通过；包含234总步（232测量）与第11步注入终止，实际动作/参数采集及退出标记均检查。专项合计18个独立测试通过（分两次运行，有重叠）；发布时重跑全部18项及既有18项动作/目标回归，最终数量以Registry回执为准。CPU动作控制映射历史证据为232行理想跟踪回放，不能替代实际SDK读回。

本提交是新入口准备：0新增原生物理步、0PPO更新、0Teacher成功。原生运行结果另行提交。GPU原生路径必须在测试、完整配方指纹、单实例锁及持续批准范围全部满足后才运行一次；不扩用旧左侧234步预算。

## 配置和保护

使用原自由杆真实重力、桌面、DH116关节保护、DeformX后端；右掌接近总位移−0.5mm，原动作索引6/7/8/17，左手与旋转零。原30mm/5s/0.5mm标准、528个保护文件和两个历史Checkpoint不变。静态安全路径已通过，实际接触/跟踪仍待该新配方验证。PR保持Draft。

## 配方加载补充检查

真实调度器的 CPU `native_recipe()` 发现初版 source_files 把资产路径纳入发布范围，按现有保护规则正确拒绝，尚未启动原生。已新增直接调用调度器的失败回归；只将源码放入 source_files，原始 USD/manifest 留在只读 inputs 以绝对路径和 SHA 锁定，未复制或修改资产。新的16项worker回归通过。该修正与其原始失败日志分别保留；不靠放宽 publishable 规则通过。

冻结当前配方 SHA：`ea4e8bc0068cd2cf5ec0457dca8f574d15e9b9cc26a5251cc687bf993e64b46c`；调度器代码配置 SHA：`2fcb2478d3c04d8e52d47638a10ec0bbecc4f34247d308a058b662d45d38d873`；内部worker代码 SHA：`45c590421e56f36fabf46786eb946c48bebf8783a274375f71c92f2a68e9d36b`。这些是执行指纹，不是成功结果。
