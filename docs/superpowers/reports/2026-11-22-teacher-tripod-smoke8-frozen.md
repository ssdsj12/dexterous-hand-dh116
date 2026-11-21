# Tripod fresh17 smoke8 原生 PPO 冻结配方
基线1f4b8385923a5ddf1ef3a579563c8293f01efc25；本包先CPU验证并经Windows发布，再允许新指纹的一次有界原生执行。

## 可执行配方
新增configs/dh116_execution/tripod-actual-smoke8-core-v1.json和独立launch companion。
复用原actual worker、scheduler、NativeContextTeacher、PPO、KL Guard与自然退出审计；没有重写算法。
模型直接读取privileged_deformable_teacher_tripod_v1.json完整原配置（std=0.1），抓法为Tripod，非Pinch模型换标签；旧Pinch配置和checkpoint保留。
真实Tripod Reset manifest SHA为ed086c283c177e52a5d9358c2f4887ccaea3d5c568e4770b19dc517d5f1f6d5c，绑定已自然退出的fresh17 Reset原始证据。
源码闭包558文件，SHA为3f07512d5e35ef4f7eacf88084a4f68770950c737f8e17e315fb83049a23dc45。
训练配置SHA为602f0299625b17207fd0137ac52c84c9abf394679cf8575dec3ce013a60d0f7c，私有路径由core输入显式绑定，不上传原始数据/模型。
语义SHA为8729f2dfffbf1a8b38e9887d8fdc6b4e3cabccdf6c171c17fc8e75e58babab30。
新原生attempt指纹c6b3b2804df1ed5f43c2fb69cd317669055241d424e45637aa34fb9de73d0a79；未复用已消耗指纹。

## 有界范围和所有权
GPU3物理卡，CUDA mask3/logical cuda:0；单Isaac、1环境、8控制步；
2初始化+64测量SDK步、64solver调用×12内部子步=768；仅1次PPO更新。
内层180s、外层240s，不自动增预算、同指纹不重跑、不自动kill。
训练不启用相机/WebRTC；现有固定双RGB配置不改。本次检验真实柔性物理-特权状态-动作-奖励-更新链，视觉独立资格不冒充本诊断结果。
R2/ForceProxy关闭，Teacher/抓取资格false，approach_wiring_smoke_v1不是完整抓取奖励。
后续恢复只允许读取本次真实已接受checkpoint和退出回执，不能事先伪造输出路径或继承旧Pinch身份。

## CPU验证和失败证据
test_native_teacher_tripod_frozen_worker新增一例，先因实际Registry无三指配方而失败，再经真实Repository解析core、companion、源SHA和实际manifest完成验证；不mock native租约，不启动App。
新用例与原actual launch、worker contract、Registry共135通过/0失败（35.19s），Registry60例包含其中，不叠加。
冻结首轮因C06输入必须精确为一个训练配置而拒绝（未启动Isaac）。修正私有冻结脚本：保留单训练输入，manifest仍由该配置SHA绑定；未放松生产validator。
前置只读资源检查：单实例控制目录无阻断、GPU3资源足够、此指纹无既有尝试。
所有日志和私有训练配置留Git外；readiness记录日志SHA。历史Reset证据和CPU green均不等于本次原生PPO通过。
Antigravity UI仍quota reached，无新审查回复；不修改额度或安全设置。

## 下一步
按用户持续授权为此精确身份建立独立私有预算记录，再经scripts/python.sh启动一次实际native worker。
记录全部64个测量SDK行、8个决策/转移、原始来源pack、Adam/RNG/策略身份、Checkpoint、GDB/外层自然退出；失败保留、其余抓法继续。
完整抓取仍需对向接触→固定腕稳定保持→完全离架Lift，原30mm/5s/0.5mm标准不变。


新增Registry条目后复验60通过/0失败（14.68s），不与135项重复相加。528保护文件和3份历史Checkpoint保持原SHA。
