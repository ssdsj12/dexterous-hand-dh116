# WireSeg depth20 独立离线物化（2026-11-14）

本包将已完整归档、已通过配对审计的 depth20 / train09 **360 对**物化为独立的
metric_depth_npy 子集。真实离线输出、完整读取及同一身份的续作校验均完成；
**实际训练更新 0、模型推理 0、下载 0、原生运行 0**。它不是新的训练 checkpoint，
也不授予质量、泛化、立体对应或 Teacher 原生资格。

工作树：
/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/wireseg-materialize-20261114
分支 codex/wireseg-materialize-20261114，基线
0e042a95260eb10a9c8e916019af1724fc41bd88。只增加 CLI、测试和本报告；
原始解码/模型、integration、README、Registry 和原 cohort 未修改。

## 输出与实际读取

最终输出目录：

/mnt/mydisk/xk/projects/dh116_data/derived/wireseg-depth20-train09-20261114-verified-metadata

| 项目 | 实际结果 |
| --- | --- |
| NPY / RGB-mask 配对 | 360 / 360 |
| 样本范围（原顺序） | flying_wire/2_cable/001200.png … 001559.png |
| 实例 masks | 713 |
| 有效深度像素 | 6,579,592 |
| NPY 总大小 | 1,509,995,520 bytes |
| 单位与有效性 | 原 float32 米；isfinite 且 >0；无缩放 |
| 物化 | 28.381537 秒，exit 0；外层上限 180 秒，未超时 |
| 已完成目录续作 | 4.408821 秒，内容完全相同 |
| 全 360 个样本通过读取接口 | 15.542091 秒，数量/实例/有效像素与原审计一致 |
| 输出文件修改时间 | 360 NPY + 2 metadata 共 362 个均未被续作/读取改变 |

最终 manifest：
/mnt/mydisk/xk/projects/dh116_data/derived/wireseg-depth20-train09-20261114-verified-metadata/materialization.json

SHA256：
48947c6e3b3f70790f96ba028066d3aebf71296975d72fa5d4c9600ce43f5d33

同目录 binding.json SHA256：
50f5d0e5115247efe93f96fd5cb7feca2a97dda0d8f7fa32570d81ced30d88e3

360 个 ordered pair SHA 的换行序列 SHA256：
88c7d0763fdc5064c5efffd0b7fa7c0a95352887c117bdaef809c72a2d2b3b5e

数据与私有 manifests 都在 /mnt 大盘，不进入 Git。RGB/mask 不另复制；
读取器通过绑定的原 parquet 只读取得它们。此目录是独立 cohort，
仍明确依赖原 pinned RGB blob、审计元数据和对应实现版本，不是假称脱离源文件的完整副本。

## 复用与校验边界

复用原 extract_depth_shard、read_subset、load_wireseg_sample、read_catalogue 和 verify_blob。
新包装器只负责独立派生目录、审计身份绑定和完成性校验：

- 固定 catalogue/revision、原 subset/ledger/admission/heldout 的显式 SHA，
  并校验实际参与的 RGB、validation RGB、depth blob 的完整大小与 SHA。
- 小型元数据只读取一次；**同一份已校验 bytes 用于解析**。
  原 read_subset 接收已校验的只读 snapshot，避免后续重读外部路径改变语义。
- 样本顺序、family/split、depth member、原 pair SHA、PNG SHA、mask JSON SHA、
  NPY SHA、RGB shape、实例数及有效深度像素必须与原审计一致。
- binding 绑定四个实际实现文件 SHA；不同实现、源身份或目标目录不复用旧完成状态。
- flock 独占写入；完整 metadata 通过 exclusive hard-link 发布，不覆盖已有文件。
  原提取器只允许已存在且逐字节一致的 NPY 续作。
- 拒绝未绑定文件、软链接、破坏的 NPY、变更的 binding/manifest、
  非独立输出目录、错序或重复身份；完成 manifest 只在全量解码通过后发布。
- 原提取器若在临时文件写入中被强制终止，未绑定临时文件会被拒绝，
  需单独查验；本包不自动删除可疑文件，也不改写既有数据。

最终 CLI SHA256：
74910afdceea88961466a2affc2d0bd011a7d7d52986b36813531c3f970f7361

读取接口示例（在本工作树、下述 PYTHONPATH 环境中运行）：

~~~python
from scripts.materialize_dh116_wireseg_subset import read_materialized

for sample in read_materialized(
    "/mnt/mydisk/xk/projects/dh116_data/derived/"
    "wireseg-depth20-train09-20261114-verified-metadata",
    expected_sha256="48947c6e3b3f70790f96ba028066d3aebf71296975d72fa5d4c9600ce43f5d33",
):
    # 原 WireSegSample；此处不构建或更新模型。
    assert sample.depth_m is not None
~~~

## 原始身份与 source-family 持出

原 cache：
/mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86

固定 revision：1dadc86f4e988a015cd9fd02b6dfd16c11c9c40b
catalogue 文件 SHA256：
bc9afab256cb247b60d45116b9351810c99d6e0818fd26e3690d1cf9dbe86e93

| 原始来源 | 完整 SHA256 |
| --- | --- |
| depth/depth-00020-of-00100.tar.gz，25,243,688 bytes | 612322216330fedb4d969c7e3b5ee31829571df1c8a045dca06d52c5d128d514 |
| data/train-00009-of-00045.parquet，622,337,468 bytes | b51ef63355390dc333d4139b307bcebd1ecbb40db015046a68921657c1e12307 |
| data/train-00028-of-00045.parquet（validation） | bf983007bc3c087a0f9b48edba2ddd04f0398c3cf595e98c90e517ef8eee3ca3 |

原配对审计目录：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/continuation-paired-depth20-train09-rkBj9Q

| 文件 | SHA256 |
| --- | --- |
| continuation-paired-depth20-train09-subset.json | dafdda5a62a35b401872dd11ead4219888c0928cc7032fcb0bdcc87bb2eab524 |
| continuation-paired-depth20-train09-samples.jsonl | ea07e6b56a4a348b51f086efe19851d341e272df3cb4eddbff076846da32ad31 |
| continuation-paired-depth20-train09-admission.json | eb32001bb97293b72e5339f9dbbf11e2c98641ef42578d066c514c4d9c155995 |

训练来源固定为 flying_wire 的上述 360 身份。原 heldout：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-original-rgb-eQP8ZB/validation/verified_subset_manifest.json
SHA256：5d5cc1331115737bfd039fa8223bd0f23cd67ea385595ffc0f03e71e6115f78f。

现有 reader 重新验证该 heldout 的完整 128 身份与顺序；
其 family 为 wire_on_plane，与训练 family 和样本身份均不重叠。
没有改用训练来源作 validation，也没有执行新的 heldout 模型评价。

原 cache 的 metric_depth_npy 仍为 1,800 文件，depth20 对应原位置仍为 0/360；
本包未向原 cache 增加派生文件。depth18 独立 cohort 保留：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-download-resume-20261113/depth18-train08-subset.json
SHA256：6b0cc32edd28a232f9b2e3db784b117c820569841601e1f66eba4904214e99c7。

旧训练 manifest 本身含 128 行，旧 checkpoint contract 只固定其前 16 个训练样本；
该 manifest SHA 38159e2f94a7528f983a1b9dca3a74f9f8bb91e5576046d31c9413b386ae905d
再次核对不变。两条当前独立 head checkpoint 仍各为历史 128 次更新；本包未恢复或加载模型：

- segmentation：evidence/wireseg-overfit-eb31c79/run01/seg-only/diagnostic.pt，
  SHA 77459b47f26ccfdeea024f298fe59bedbb9c119242cddda305aeea0b71cf5f39。
- depth：evidence/wireseg-overfit-eb31c79/run02/depth-only/diagnostic.pt，
  SHA 56826811b64075446486fc0309a60a616dabcafc738d40700be04ec39ac21adb。

这里 evidence 的绝对前缀均为
/home/xk/.local/share/dh116-autonomous-3c80166/。

## 测试、审查修正与可重放证据

Ubuntu 现有环境，无安装依赖：

~~~bash
cd /mnt/mydisk/xk/projects/dh116-autonomous-3c80166/wireseg-materialize-20261114
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:.:/mnt/mydisk/xk/projects/dh116_data/huggingface/WireSeg-36K-1dadc86/tools \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q \
tests/test_dh116_wireseg_materialize.py \
tests/test_dh116_wireseg_pretraining.py \
tests/test_dh116_wireseg_blob_relay.py
~~~

结果：**33 新测试 + 18 既有回归 = 51 passed / 4.91 秒**。
fixture 使用实际 PNG/RLE/NPY/parquet 路径，仅以小型 fixture catalogue 替代官方完整 catalogue；
不以 fixture 声称真实训练或泛化。第一轮缺新模块的 20 项 setup error 已留档。
另一次回归命令误写既有测试文件名，0 测试运行；更正后通过，日志未覆盖。

root 只读审查发现小元数据 digest/read 的二次读取缺口。
5 个独立换读负例先 **5 failed**，再以同份 bytes 校验解析修复；
额外测试确认 decoder 使用已校验 snapshot，外部路径变动会令后续 resume 拒绝。
两位其他 agent 因主线忙未审查，不计为已审。

首轮真实物化在收到此审查时已完成：30.397027 秒 / exit 0；
旧输出目录为相同前缀但没有 -verified-metadata 后缀，
旧 manifest SHA 133c69e9ed00c65f8ebb6a285f0dfc6b25f8dd65910d1da443a0cadf6acbfe01。
原执行源码及旧输出完整保留，不改其实现 binding，不冒充修正后输出；
最终读取器因实现 SHA 不同拒绝将其复用。修正后在新目录重新物化，
因此本包实际有两次离线数据物化，均 0 优化器更新。

证据根：
/home/xk/.local/share/dh116-autonomous-3c80166/evidence/wireseg-materialize-20261114

| 文件 | SHA256 |
| --- | --- |
| fixed-run-plan.json（含实际完整 argv） | 1f7f37634fe34cc8981bfea2b73f4ccf79b3c4657a782108a401ab8c106a4ad2 |
| fixed-run-result.json（外层退出码/墙钟） | b957de86b9421c102e70e594e7171dbb2eb6c56e689c83a8a58c48a652d0c00f |
| fixed-materialize.log | 67aee2ea551665190af36c97d481025b6ceafad55cf4d44cce98d856b581feea |
| readback-resume-preservation.json | dc7e73d526f73578e04159dc26918b47df4cdf0da3b635e80fa90ea67580a8df |
| tests-metadata-red.log | b1a7d605e64b7be2ae2402fb4512f3173e187913f4ec5505e50e18f28040e012 |
| tests-final.log | 502e9c8d121a56b9f59736d1ad0e3c87bb4a10e1230da42d41047a04b4f18dcc |
| initial-executed-source.py（旧输出实现） | dccb1897fe53328765c0403ca720971e4376e3644c411753f6fb7a5b6312211d |

preservation 证明逐一核对原 train/validation/depth blobs、原配对审计元数据、
旧训练 manifest 及两条旧 checkpoint 的 SHA。完整读取验证没有改写最终输出文件。

## 下一训练阻断

物化缺口已关闭，但这 360 对没有自动加入旧训练 contract，也没有替换 depth18 cohort。
历史 Python/NumPy RNG 缺失仍是显式未知；legacy diagnostic import 不变成严格 native v2 resume，
更不能把新 cohort 与旧 sampler 的身份混用。后续若使用新 cohort，须明确新的数据 contract、
head/lineage、授权更新预算及评价标准。

原 depth20 审计记录中心 192 ROI 有深度支持 292/360、无支持 68/360；
full-field 有效深度集中在 wire 标签，背景监督域差异仍未解决。
这些是原固定审计的已知数据支持，不是本包新增模型评价。
现有 heldout depth 指标差于 constant baseline 的结论也未被物化改变；
应先明确 wire/background 有效标签与 loss 域，再决定是否续训。本包不申请或追加训练预算。

上述视觉数据与训练阻断不阻塞无相机的 Teacher 物理准备。
