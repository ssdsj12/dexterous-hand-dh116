# DH116 production workflow

- 默认简体中文。唯一项目根目录/mnt/mydisk/xk/dh116；包src/dh116/，开发分支11-17。不新建日期工作树，不修改11-10。
- 按env/physics/rl/hand分层，保留cuda_measurements扩展；禁止src反向import脚本，不添加旧包兼容壳。
- scripts不超过8个入口，tests不超过20份核心测试；新增实现须有实际调用与回归。
- 不放宽物理/几何/关节守卫，不改变24D/371D合同或30mm Lift、5s Hold、0.5mm Slip标准。
- CPU、CUDA、Isaac原生和实际抓取成功独立报告。核验源码SHA、二进制SHA与自然退出，禁止硬退出伪造成功。
- 模型、数据、缓存和日志只写本地runs/build等忽略目录。历史数据保留项目外归档，禁止上传。
- 全部保留测试与必要原生验证通过后，更新README、CHANGELOG与readiness；确认无无关文件，再按用户要求git add -A及commit。
- 发布由Windows从Ubuntu导入同一commit后push，核对远端SHA、提交树和文件清单。Ubuntu不直接push。PR #2保持Draft，不merge、force-push、reset或覆盖用户未提交修改。
