# Tripod完整来源复验的owned-JSON批量解码
基线b4848c3092cd868538b8f211bcb5ecc90cf3eeaa；沿用现有执行链，不调整物理、动作、模型或来源资格。

## 实际代码
tabletop_fast_owned_codec.decode_owned新增纯数据解码：NumPy只探测纯数值结构，按原payload和原dtype创建拥有独立内存的数组。
不能从探测数组转换目标dtype，否则混合float输入会先丢失2**53以上整数；回归明确覆盖9007199254740993。
包含tag/对象/非规则结构的array保留原decoder；非法nonfinite tag不能被bool转换吞成True。
原NaN/Inf/负零/端序/Mask/None和异常行为保留。不量化、不补零、不插值、不改时间戳。
tabletop_tripod_reset_manifest._derive仅修改decode_owned导入；原encode_owned、每次SHA、所有native生命周期/Reset/几何/物理来源验证和data==actual检查均保留。
原tabletop_deformable_r0_replay.py原SHA不变。无全局decoder monkeypatch，无验证/资格缓存，无新依赖。

## CPU正确性
5个新增主要边界用例先在缺少函数时失败；随后原/新owned codec共15通过。
与原始Tripod manifest、Factory、Source回归共22通过（101.37s）；当前v7配方/实际启动/Worker合同/Registry135通过（39.83s）。
两组文件不重叠，共157通过、0失败。15已含于22，不重复累计。
4份真实原生JSON全部经原SHA核验，包含198718823字节mesh_reset_state.json；
两decoder的全部canonical输出SHA一致（见readiness），不是只比较几个抽样顶点。
这些是CPU读取已有原始数据，不是新的原生采集或成功轨迹。

## 实测性能，明确两种口径
同进程3轮交替计时，完整原始JSON解码中位数legacy1.406034s，bulk0.475786s，约2.955倍。
完整manifest来源与几何复验也做3轮交替：旧加载器代码直接从基线Git只读载入隔离reference module，新加载器运行当前实际代码。
旧中位数9.932104s，新8.560429s，约1.160倍；两者均实际重新验证源，返回相同SHA ed086c283c177e52a5d9358c2f4887ccaea3d5c568e4770b19dc517d5f1f6d5c。
因此不能把decoder约3倍描述为原生流程/训练吞吐3倍，也不能声称现有wall预算已足够。
原始计时、代码SHA、文件SHA和CPU恢复点保留Git外；未上传约199MB原始数据或模型。

## 独立复核
只读Agent分别进行了规范与质量审查，核对实际文件SHA，无阻断/重大问题。
重点确认probe不参与最终数值转换、非法bool tag回退、legacy未变、完整资格检查未删。
审查Agent未运行测试或基准，数值结果来自主代理的实际执行。Antigravity仍无额度，这不是Antigravity新审查意见。

## 晋级与下一包
v6未曾运行，保留其配方/身份；因源码变化标为superseded，不记录为原生失败。
v7绑定563文件闭包；指纹148825da36a0aa8a78fba4246ce5c616ba732b55f48f46d9d155e901af0b101f；仅冻结、未运行。
原8控制/66SDK/768solver内部子步/1更新、180/240s上限不扩大，没有消费新的原生预算。
根据v5真实stack/refs与代码，actual_provider.acquire中的stat-cache释放只在_acquire成功后执行；
v5异常路径无stat_cache_release.json，仍保留constructor frame、hand和view等直到Py_FinalizeEx SIGSEGV。
这是已定位的异常路径遗漏；具体原生退出是否因补齐而解决，必须独立验证，不能仅凭CPU宣称通过。
下一包在原已有before_collect接口接通异常/正常两路释放，保持封存生命周期脚本与Reset来源SHA，不添加强制退出。
R2/ForceProxy、成功BC/DAgger仍关闭；无完整抓取资格，30mm/5s/0.5mm标准不变。
528保护文件、3份旧Checkpoint和固定双RGB配置不变；不操作其他用户进程。
