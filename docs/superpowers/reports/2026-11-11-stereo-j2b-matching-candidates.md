# J2-B：独立匹配候选与粗到细验证

基于本地`a5a15fb`继续已批准任务书J2-B；重新fetch确认origin仍为`692e09d`。四候选实际实现并通过软件/列出解析场景门槛；默认SAD、SGBM、reliable_sad的数值路径保留。没有Isaac/GPU计算、模型训练、push或merge，不能据此宣称原生深度、Teacher/Student或抓取成功。

## 文件与合同

- 新`tabletop_stereo_matching_costs.py`：固定5×5的24位Census/Hamming、ZNCC的1-rho、中心差分梯度L1；完整RAW mask/stencil约束和分离竞争模式，逐视差流式计算。
- 新`tabletop_stereo_pyramid.py`：实际.25/.5/1图像金字塔，以实际尺寸比缩放K和disparity，最终原图refine。每层记录K、range、local与global mask；小图不支持的层明确skip。双向都执行，最终LR保持现有规则。
- `tabletop_stereo_depth.py`：四种显式method进入既有compute、批处理、NaN/valid和诊断合同，默认12字段配置及SAD旧五Tensor回归不变；诊断仅声明实际执行的uniqueness与RGB boundary proxy。
- `tabletop_stereo_preprocessing.py`与`tabletop_stereo_error_cli.py`：绑定新源码SHA，旧raw-RGB模型包不静默迁移。没有修改历史Checkpoint。
- 四份`stereo_matcher_{census,zncc,gradient,coarse_to_fine}_j2b_v1.json`独立配置，仅method不同、其他参数相同，默认关闭，必须绑定实际相机。每算法的独立config SHA及完整implementation SHA在[机器摘要](../../contracts/stereo_j2b_validation_summary.json)。
- 四份新J2-B测试，共134项，覆盖真实代价、两侧sign、完整支持、mask填充独立性、分批、K、错误粗峰恢复及解析精度。README、CHANGELOG、readiness与完整文件清单同步更新。

亚像素直接用各自原指标评价13个-.75:.125:.75分数位移，不暗中换成SAD。整个patch使用中心像素的统一视差，不让邻居winner改变中心成本。只接受支持充分、非平坦且有分离竞争证据的局部最优；二进制平台/相等候选用明确integer fallback并标refinement_valid=false。输出float视差不代表每个像素都被亚像素细化。ZNCC内部float64矩降低亮度变换下的消减误差；左右整数与分数窗口variance<=1e-8均invalid。

没有完整B×D×H×W体。固定数量的特征/成本/诊断平面仍会占内存，`cached_cost_state_planes`未测不填假精确值。金字塔每个细层在局部±2后进行全range audit；不信任错误粗峰的自洽局部置信度，global拒绝覆盖local。此策略以计算量换可靠性，**没有搜索加速或VGA/GPU吞吐结论**。

## 失败和修复记录

根集成RED17项为真正的unsupported method ValueError；元数据回归另有真实RED/GREEN。成本初始RED39、alignment RED6/35pass均保留。修复ZNCC矩精度、完整central stencil支持和原批准的central-L1度量，未放宽阈值。金字塔最初缺module属于setup失败，不计语义TDD；后补single-pass语义控制是实现后记录，不能冒充严格test-first证据。之后混合有效性batch的实际RED再修复：prior.any导致scalar诊断随分批改变，现每个细层统一调用local并对无效prior fail closed。

中间全组Windows861/1skip虽无pytest失败，新增24精度用例时source guard正确报变动，不能作为最终源码通过；中间server862/1skip仅对应旧110项版本。数值prelim/v01也因源码变动保留为intermediate。最终v02生产源码稳定，追加24测试只改测试文件，原v02历史SHA及追加后身份并存；最终完整组使用追加后的冻结源码重新执行。

## 实际验证

新增初次组合110passed/49.12s；补充24精度用例24passed/33deselected/29.88s。最终全部相关组Windows **885passed/1skipped，149.97s**；服务器 **886passed/1skipped，140.38s**，均exit0。两平台及子组重叠，不能相加。包含全部stereo、旧oracle/default matcher、OpenCV、privileged/371D/24D与surface/squashed-Gaussian回归。

Windows现有Python3.12.13/Torch2.14 CPU；server现有Python3.10.12/Torch2.7cu128/NumPy1.26.4/OpenCV4.11/SciPy1.15.3，未安装依赖。两端CUDA_VISIBLE_DEVICES为空、未导入Isaac；原528保护文件SHA不变，1131份src/scripts/configs/tests与server实际执行快照逐字节一致。

```text
PYTHONPATH=src;artifacts/stage7_lifecycle_cpu_deps_v01
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
python artifacts/stereo-j2b-20261111/run_cpu.py full
python artifacts/stereo-j2b-20261111/analytic/evaluate_candidates.py --label run_final_v02 --require-qualified
```

```bash
DH116_PYTHON=/mnt/mydisk/xk/envs/dh116/bin/python DH116_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 \
/mnt/mydisk/xk/dh116/scripts/python.sh \
/tmp/dh116-stereo-j2b-20261111-v02/frozen_entry.py pytest
```

实际full选择全部`tests/test_dh116_stereo_*.py`及既定11份privileged/Phase0/surface/squashed回归，runner与完整日志私有保留。公开可复现软件/精度测试命令为`python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_j2b_analytic.py tests/test_dh116_stereo_j2b_costs.py tests/test_dh116_stereo_j2b_pyramid.py tests/test_dh116_stereo_j2b_integration.py`。归档1213项SHA见机器摘要，原始日志/源码快照/运行数据不上传。

## 固定解析参考结果

48case是CPU直接连续辐射采样，不是48个原生初态：16新候选fractional、8旧control、8Census/ZNCC affine、16退化/缺失。d为6.1/6.25/6.5/6.75；每case固定2800参考像素，coverage>=70%、disparity P95<=.2px；coverage与固定参考正确比例的分母不删除无效点；MAE/RMSE/P95仅统计已接受的参考像素，空集为NA。16负例均fail closed，数值runner exit0/source_stable=true。逐case图像/标定/reference/config/output/source SHA保留在private原件；公开处理后摘要发布其processed_metrics总SHA及source/config/test SHA；未使用GT作为matcher输入。

|method|四个fractional coverage|最大disparity P95|最小固定参考正确比例(<=.2px)|
|---|---|---:|---:|
|SAD control|100/100/73.714/100%|.135774px|73.714%|
|reliable_sad control|全100%|.0250001px|100%|
|Census|全100%|.0250001px|99.9286%|
|ZNCC|全100%|.0250001px|100%|
|Gradient|全100%|.0250001px|100%|
|Coarse-to-fine|全100%|.0250001px|100%|

新候选最高轴向Z P95为4.28236mm。这是axial Z，**不是5mm Euclidean正确3D点完整率**。Census仍有极少坏尾，P95通过不等于每点正确。未检验完整距离/方向/线宽因子矩阵、任意重复图案或真实手部遮挡；J2-A保留的27个混合遮挡边缘错点没有在此被宣告解决。

上述解析执行配置为range2..12、block9、LR.5、uniqueness.01，K为fixture的fx80/B.08；四个独立生产候选配置为range8..192、LR1/uniqueness.05，需配实际相机。解析小范围门槛通过不授予这些VGA生产profile的实测资格，也没有测SGBM在本解析矩阵的结果；旧OpenCV路径仅作回归。各层或metric固有参数均由源码SHA绑定。

## 状态与下一阶段

J2-B实现/CPU/上述解析场景通过；原10run native accuracy仍未通过，最终resolution未选，正式BC/DAgger和部署继续blocked。[readiness](../../contracts/stereo_phase0_7_readiness.json)分开记录这些轴。GPU显存、sensor-to-Actor延迟、完整训练吞吐均NA；本轮未执行J2-F完整同cohort原始数据比较。下一阶段按已批准顺序是J2-C真实native-ID标签约束和共享backbone分割，再J2-D拓扑/跟踪与J2-E独立depth head，之后J2-F同数据比较和J3申请；不自动启动原生训练或大规模Isaac。
