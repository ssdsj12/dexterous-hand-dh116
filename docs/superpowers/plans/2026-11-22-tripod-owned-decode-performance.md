# Tripod Owned Decode Performance Implementation Plan

> **For agentic workers:** use executing-plans with independent spec and quality review, then verify and publish the finite work package.

**Goal:** reduce verified raw-array decode cost without changing any evidence or admission gate.

**Architecture:** independent decoder; bulk only for proven numeric structure, always convert the original payload; tags and unusual structures retain the legacy path. Tripod manifest keeps all provenance and geometry checks.

**Tech Stack:** existing Python3.10 / NumPy / PyTorch on Ubuntu; no new dependency.

---
沿用已批准的现有执行链分包修复与完整来源校验；Ubuntu开发、Windows同commit发布，不改物理/模型。

目的：v5进入更新前只余12.238s；CPU profile发现原始数组解码逐标量递归，manifest一次完整验证17.0127s（含profiling开销）。
本包只尝试CPU纯数据等价优化，不启动Isaac、不扩大原生预算、不缓存资格。

1. 在既有owned codec回归里新增可观测边界：Mask/NaN/负零/端序/大整数混合输入、非法非有限tag、对象dtype与原函数拒绝行为。
2. 先看到红测试，再新增独立fast-owned decoder。原解码器不改；标签或非纯数值数组回退原解码器。
   numpy探测只能判定结构；最终还原从原payload和原dtype进行，防止mixed float/int探测先丢失2**53以上整数。
   bool目标不能把不合法tag字典强制转为True。
3. 使用真实SHA绑定原始native JSON进行CPU重算，验证深度结构/数值/编码字节一致；再比原/新解析耗时。
4. 仅在等价与收益都有证据后，在Tripod manifest _derive直接调用该decoder；每次完整源文件SHA、几何与生命周期复验照常执行。
5. 回归、readiness、报告与恢复点；若改变实际worker闭包，重新冻结当前配方但不自动消耗原生预算。
6. 复验528保护文件、3份旧Checkpoint、固定双RGB；独立commit，Windows发布并核对远端SHA，PR2保持Draft。

## Implementation and verification

Files: tabletop_fast_owned_codec.py, tabletop_tripod_reset_manifest.py; primary tests test_native_teacher_owned_codec_parity.py.

- [x] Read current implementations and actual cProfile data.
- [x] Add5 boundary regressions and capture5 failures before implementation.
- [x] Verify decoder semantics before integrating manifest.
- [x] CPU raw-evidence equivalence and before/after wall measurements.
- [x] Integrate only if qualified, run regressions, independently review.
- [ ] Report/Registry/protection/Windows same-commit publication.

Exact decoder candidate:

```python
def decode_owned(value):
    """Decode numeric JSON arrays without walking every scalar in Python.

    The dtype probe only excludes dictionaries/tags and irregular structure.
    Conversion always uses the original payload, preserving mixed integer/float
    values before the declared dtype is applied. Tagged or unusual inputs keep
    the original decoder's validation and rejection behavior.
    """
    if isinstance(value,dict):
        if set(value)=={"__dh116_array__","dtype"}:
            dtype=np.dtype(value["dtype"])
            if dtype.kind not in "bifu":
                raise ValueError("invalid numeric array type")
            payload=value["__dh116_array__"]
            try:
                probe=np.asarray(payload)
            except (TypeError,ValueError,OverflowError):
                return codec.decode_owned(value)
            if probe.dtype.kind in "bifu":
                return np.array(payload,dtype=dtype,copy=True)
            return codec.decode_owned(value)
        if set(value)=={"__dh116_nonfinite__"}:
            return codec.decode_owned(value)
        return {key:decode_owned(item) for key,item in value.items()}
    if isinstance(value,list):
        return [decode_owned(item) for item in value]
    return value

```

CPU command: /mnt/mydisk/xk/envs/dh116/bin/python -B -m pytest -q -p no:cacheprovider tests/test_native_teacher_owned_codec_parity.py; environment is the existing dh116_wire_joint_pregrasp_v1.json CPU environment. New data remain outside Git; primary function is exercised through native JSON schemas, not flags or source greps.
