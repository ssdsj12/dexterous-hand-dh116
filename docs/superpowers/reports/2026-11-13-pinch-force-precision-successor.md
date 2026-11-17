# Pinch 后继配方的受控力归约

新增独立 `validate_ledger`，接收原始 Pinch 行。先用原 float32 节点向量求和顺序复验 producer 的标量，独立用 float64 重算实际力差并继续执行原 1e−3 N 守恒门。只把归约后的标量交给原 validator 的内存副本，所有原始数组、样本时钟、关节与接触检查继续经过原 validator。v2/v3 源码与失败记录维持冻结。

在真实 v3 只读原件上，238 行通过新数值路径，原 FP32/FP64 差异 208 行，独立最大残差 1.491979e−6 N，手部对向接触候选 0。旧外层退出码 1 保留。该结果不授予原生通过或训练资格。

Ubuntu CPU **36 passed**，新增 5 项。覆盖源标量篡改、实际超过力残差门、缺节点、非有限值和不改原对象。使用空 CUDA_VISIBLE_DEVICES、2 线程、`bash scripts/python.sh -m pytest -q -p no:cacheprovider`，测试为 force precision、v3 evidence replay、grasp runtime、pair capture 四个文件。

CPU 日志 SHA256 `2d0025f7a0efc6cfaeed95ab06a8cc4e52915785b8e3435231b6a823c71e757d`。实际 v3 离线兼容回执 SHA256 `010db2c3359b281808296d45811340d7c42057debb75c269200cc07a7d92e996`，原始账本 SHA256 `d002b0f8cab4108b91bd92976acbb0e95629aab856e949bffb279c240a7c580a`。0 新原生步、0 PPO 更新。下一步接入冻结的新接触候选并验证自然退出。
