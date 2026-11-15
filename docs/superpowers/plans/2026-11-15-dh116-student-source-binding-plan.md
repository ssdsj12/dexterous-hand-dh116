# R04 Student 原始记录绑定与在线来源实施计划（修订稿）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans。按任务逐包执行；本文只修订 Markdown，没有实现。

**Goal:** 绑定真实 Student 数据来源，同时保持在线 DAgger 的“本轮真实采集→验证→optimizer→下一轮”结构。

**Architecture:** 两条来源路径分开：磁盘封存重放使用完整 native proof、外部自然退出回执和原始 records；真实在线更新使用由实际 collector/session 产生、在 optimizer 前验证的 live producer witness。自然退出是整个 run 的最终证据门，不是每轮 optimizer 的永久前置。既有 Teacher 成功资格和 Student 成功标准仍独立验证。

**Tech Stack:** Python、PyTorch、现有原生来源校验器、pytest。当前工作只包含 CPU 修复与计划；不启动新 native、不扩大原生预算。

基线：`5b10a4c2f457e965e24ec79ce7033374d46c1852`。新架构参考：`/home/xk/.local/share/dh116-autonomous-3c80166/evidence/ppo-reward-doc-20261115/extracted.txt` 第56、94–122、147、149、175、208、400、442、570、625行。

## 两条路径及完成边界

| 路径 | 来源门 | 更新时点 | 当前计划状态 |
|---|---|---|---|
| sealed/offline replay | proof→实际 external completion→同run source manifest→原始 records→精确 rows | run 已自然退出后重放 | R04.1、R04.2可执行 |
| live/online DAgger | 当前真实 collector/session→受控owner/epoch→fsync raw chunk/hash→opaque witness→精确 rows | 同run每轮 optimizer 前，不等待最终close | R04-live需独立设计与接口批准，不能当成已实现 |

当前可交付是 sealed-source 修复，不能据此声称整个 R04 或在线来源闭环已完成。不得把 `run_online_dagger` 改成“收集退出后离线 BC”，也不添加要求每轮 native process close 的永久 guard。其现有在线采集与更新顺序保留；正式在线资格仍按现有实际门槛报告，未来由 R04-live 接入真正来源 witness。

公共 `validate_dataset`、`batch_dataset`、`StereoBCTrainer._step` 和 `DAggerDataset.append` 的统一来源接线必须在两条来源接口都明确后完成。不能在这些共享入口无条件调用 sealed-only verifier，从而把活跃 session 错误地要求为已退出。

## 已核对的接口与可信锚点

- `tabletop_stereo_training_cli.py::admit_candidates` 已读取 candidate records 和 teacher_checkpoint_sha256；该转换适合 sealed-source admission。
- `run_online_dagger` 在同一 native callback 内采集并更新；当前 pool.append 前未另外封存原始 records，不存在该run最终外部completion。此事实是 live-source 设计缺口，不是改成离线流程的理由。
- `tabletop_stereo_supervision_dataset.py::rows_from_records` 已有，必须复用实际分拆方式。
- `tabletop_stereo_promotion.py::_read_object`、`_artifact`、`_native_run`、`_process_output` 已有。`_process_output` 从实际外部 completion 的 produced_artifacts 验证路径与 SHA。
- 每份 sealed binding 的 proof 必须属于 records 自己的 native run；历史 Teacher 成功证明不能认证任意新采集 run。
- payload 自己提供的 bool、hash、路径或重算后的新文件都不能创建原生来源资格。旧 formal 文件没有绑定时拒收并保留原件；schema_only fixture 明确豁免。
- live witness 合同尚未实现时，不增加伪造 witness fixture 来宣布真实在线已恢复；独立设计任务必须保留未完成状态。

## Task R04.1：定义并实现可信源重放校验

**新增：**

- `src/go2_pvcnn/control/dh116/tabletop_stereo_source_binding.py`
- `tests/test_dh116_stereo_source_binding.py`

**新增 API（先定义，再接入）：**

1. `verify_stereo_dataset_sources(data: dict) -> None`：验证 formal payload 的每个 source binding、源 manifest、原始 records 和精确 row 对应关系。
2. 模块内部 `_same(a, b) -> bool`：比较 tensor dtype/shape/raw bytes、dict/list/tuple 与标量。
3. 数据集新增 `source_bindings`：非空有序 list，每项恰含 `proof_path`、`proof_sha256`、`row_indices`。它只提供定位信息，资格仍来自实际 native proof 与外部 completion；binding 自带哈希本身不授予资格。

原生源 manifest 新合同：`dh116-stereo-source-manifest-v1`，恰含 `version`、`teacher_checkpoint_sha256`、`native_raw_artifact`、`records_artifact`。native proof 中的 `stereo_source_manifest_artifact` 指向该 manifest；manifest 与 records 都必须属于该 proof 的同一外部 completion。records packet 保留现有 `dh116-stereo-teacher-candidate-records-v1` 格式，仍使用 `records` 与 `teacher_checkpoint_sha256`。Dagger 也使用这一原始记录封装，但不将其称为成功 demonstration。

- [ ] 先创建以下测试，当前 import 因模块缺失失败。这是明确的新能力 RED：

```python
"""Source binding negatives use no native evidence and grant no qualification."""
from copy import deepcopy

import pytest

from test_dh116_stereo_collection import record
from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import build_dataset


def formal_fixture_without_binding():
    data = build_dataset([record()], source_sha256="f" * 64, schema_only=True)
    data["schema_only"] = False
    data["purpose"] = "dagger_visited_states"
    data["source_bindings"] = []
    return data


def test_missing_formal_source_binding_fails_before_native_lookup():
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import (
        verify_stereo_dataset_sources,
    )
    with pytest.raises(ValueError, match="trusted source bindings"):
        verify_stereo_dataset_sources(formal_fixture_without_binding())


def test_missing_native_proof_cannot_be_replaced_with_payload_hash(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import (
        verify_stereo_dataset_sources,
    )
    data = formal_fixture_without_binding()
    data["source_bindings"] = [dict(
        proof_path=str(tmp_path / "absent-native-proof.json"),
        proof_sha256="a" * 64,
        row_indices=[0],
    )]
    with pytest.raises(ValueError, match="native source proof"):
        verify_stereo_dataset_sources(data)


def test_schema_fixture_has_explicit_source_binding_exemption():
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import (
        verify_stereo_dataset_sources,
    )
    data = build_dataset([record()], source_sha256="f" * 64, schema_only=True)
    before = deepcopy(data["rows"][0]["teacher_mean"])
    assert verify_stereo_dataset_sources(data) is None
    assert data["rows"][0]["teacher_mean"].equal(before)
```

- [ ] RED 命令：

```bash
PYTHONPATH=src:.:tests PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_source_binding.py
```

- [ ] 新模块最小实现：

```python
"""Exact row replay from records sealed by the actual native process observer."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import torch

from . import tabletop_stereo_promotion as promotion


def _same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        left = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        right = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return torch.equal(left, right)
    if isinstance(a, dict) or isinstance(b, dict):
        return (
            isinstance(a, dict) and isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_same(a[key], b[key]) for key in a)
        )
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (
            type(a) is type(b) and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    return type(a) is type(b) and a == b


def verify_stereo_dataset_sources(data):
    if data.get("schema_only") is True:
        return
    bindings = data.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("formal data requires trusted source bindings")
    from .tabletop_stereo_supervision_dataset import rows_from_records

    expected_rows = []
    for binding in bindings:
        if not isinstance(binding, dict) or binding.keys() != {
            "proof_path", "proof_sha256", "row_indices",
        }:
            raise ValueError("exact trusted source bindings required")
        try:
            proof_path = Path(binding["proof_path"])
            if not proof_path.is_absolute():
                raise ValueError("absolute native source proof required")
            proof_bytes = proof_path.read_bytes()
            promotion.require_sha(binding["proof_sha256"])
            if hashlib.sha256(proof_bytes).hexdigest() != binding["proof_sha256"]:
                raise ValueError("native source proof SHA mismatch")
            proof = json.loads(proof_bytes)
        except (OSError, TypeError, ValueError) as error:
            raise ValueError("native source proof unavailable or changed") from error

        base = proof_path.parent
        promotion._native_run(proof, base)
        manifest_bytes, manifest_path = promotion._process_output(
            proof, base, proof.get("stereo_source_manifest_artifact"),
        )
        manifest = json.loads(manifest_bytes)
        if (
            not isinstance(manifest, dict)
            or manifest.keys() != {
                "version", "teacher_checkpoint_sha256",
                "native_raw_artifact", "records_artifact",
            }
            or manifest["version"] != "dh116-stereo-source-manifest-v1"
            or manifest["teacher_checkpoint_sha256"] != data["source_sha256"]
            or manifest["native_raw_artifact"] != proof.get("raw_artifact")
        ):
            raise ValueError("native stereo source manifest identity differs")
        records_bytes, _ = promotion._process_output(
            proof, base, manifest["records_artifact"],
            binding_base=manifest_path.parent,
        )
        packet = torch.load(
            io.BytesIO(records_bytes), map_location="cpu", weights_only=True,
        )
        if (
            not isinstance(packet, dict)
            or packet.get("version") != "dh116-stereo-teacher-candidate-records-v1"
            or packet.get("teacher_checkpoint_sha256") != data["source_sha256"]
            or not isinstance(packet.get("records"), list)
        ):
            raise ValueError("actual sealed stereo records required")
        records = deepcopy(packet["records"])
        if data.get("purpose") == "successful_bc":
            for record in records:
                record["mode"] = "teacher_reference"
                record["teacher_qualification"] = deepcopy(data["qualification"])
                record["candidate_only"] = False
        elif data.get("purpose") != "dagger_visited_states":
            raise ValueError("explicit formal source purpose required")
        rows = rows_from_records(records)
        indices = binding["row_indices"]
        if (
            not isinstance(indices, list) or not indices
            or any(type(index) is not int or not 0 <= index < len(rows)
                   for index in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("ordered unique raw record row selection required")
        expected_rows.extend(rows[index] for index in indices)
    if not _same(data.get("rows"), expected_rows):
        raise ValueError("dataset rows differ from sealed native records")
```

这一最小实现不是 source manifest 的生产器。当前原生 proof 无 manifest 时必须拒收；不得修改旧 proof/completion 使其通过。

- [ ] 在同一新测试文件补以下边界测试；均只测试上层精确比较，绝不称 native admission：

```python
@pytest.mark.parametrize("field", ["left_rgb", "right_rgb", "proprio"])
def test_exact_tensor_comparison_rejects_payload_changes(field):
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import _same
    data = build_dataset([record()], source_sha256="f" * 64, schema_only=True)
    original = data["rows"][0]
    changed = deepcopy(original)
    changed["history"][field].reshape(-1)[0] += .125
    assert not _same(original, changed)


@pytest.mark.parametrize(
    "field", ["frame_id", "reset_id", "policy_tick",
              "physical_initial_condition_sha256", "source_episode_group"],
)
def test_exact_row_comparison_rejects_identity_and_split_changes(field):
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import _same
    data = build_dataset([record()], source_sha256="f" * 64, schema_only=True)
    original = data["rows"][0]
    changed = deepcopy(original)
    value = changed["identity"][field]
    changed["identity"][field] = value + 1 if type(value) is int else value + "-changed"
    assert not _same(original, changed)


def test_exact_row_comparison_rejects_reordered_rows():
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import _same
    data = build_dataset([record()], source_sha256="f" * 64, schema_only=True)
    assert not _same(data["rows"], list(reversed(data["rows"])))


def test_exact_row_comparison_rejects_jointly_rewritten_action_and_owner_request():
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import _same
    data = build_dataset([record()], source_sha256="f" * 64, schema_only=True)
    original = data["rows"][0]
    changed = deepcopy(original)
    changed["teacher_mean"].fill_(.9)
    changed["issued_actions"].copy_(changed["teacher_mean"])
    assert not _same(original, changed)
```

- [ ] GREEN 命令同 RED，预期新模块测试全通过；报告实际数量与仅 CPU 范围。

- [ ] 提交范围：

```bash
git diff --check
git add -- src/go2_pvcnn/control/dh116/tabletop_stereo_source_binding.py tests/test_dh116_stereo_source_binding.py
git diff --cached
git commit -m "Add native-sealed stereo source record replay"
```

## Task R04.2：接入封存离线装载与 candidate admission

**修改：**

- `src/go2_pvcnn/control/dh116/tabletop_stereo_supervision_dataset.py`
- `src/go2_pvcnn/control/dh116/tabletop_stereo_training_cli.py`
- `tests/test_dh116_stereo_source_binding.py`
- `tests/test_dh116_stereo_training_cli.py`

新增 API：`validate_sealed_stereo_dataset(data, *, allow_schema_only=False)`。先定义该函数，再用于磁盘 loader 与明确的 sealed candidate admission。它不代替 live dataset/session verifier。

- [ ] 失败测试：先在新 source_binding 测试文件加入以下测试，当前缺少新函数，因此 RED 为 ImportError：

```python
def test_sealed_admission_rejects_missing_source_before_teacher_success_gate():
    from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import (
        validate_sealed_stereo_dataset,
    )
    data = formal_fixture_without_binding()
    with pytest.raises(ValueError, match="trusted source bindings"):
        validate_sealed_stereo_dataset(data)


def test_sealed_schema_path_remains_explicitly_unqualified():
    from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import (
        validate_sealed_stereo_dataset,
    )
    data = build_dataset([record()], source_sha256="f" * 64, schema_only=True)
    checked = validate_sealed_stereo_dataset(data, allow_schema_only=True)
    assert checked["native_qualified"] is False
    assert checked["successful_demonstrations"] is False
```

- [ ] 最小函数补丁，放在 `validate_dataset` 定义之后：

```python
def validate_sealed_stereo_dataset(data, *, allow_schema_only=False):
    if not isinstance(data, dict):
        raise ValueError("versioned stereo dataset required")
    if data.get("schema_only") is not True:
        from .tabletop_stereo_source_binding import verify_stereo_dataset_sources
        verify_stereo_dataset_sources(data)
    return validate_dataset(data, allow_schema_only=allow_schema_only)
```

- [ ] 在 `load_dataset` 中，用以下调用替换现有 `validate_dataset(data, allow_schema_only=allow_schema_only)`：

```python
validate_sealed_stereo_dataset(data, allow_schema_only=allow_schema_only)
```

旧 formal 磁盘数据缺绑定时不能继续作为新版本 formal 输入；不重写旧文件。schema fixture 明示 allow_schema_only 仍可装载。live pool 在采集中的磁盘快照不因此变成已封存来源；正式重放需要最终 source receipt。

- [ ] `build_dataset` 新增 keyword-only 参数 `source_bindings=None`，payload 构造增加：

```python
source_bindings=copy.deepcopy(source_bindings),
```

- [ ] `admit_candidates` 新增 keyword-only 参数 `source_bindings=None`；将其传入 `build_dataset`。在 save_dataset 之前显式调用：

```python
from .tabletop_stereo_supervision_dataset import validate_sealed_stereo_dataset
validate_sealed_stereo_dataset(dataset)
```

当前正式成功池仍先要求真实 Teacher/episode 成功证明。新 source verifier 附加校验，不能用它替换成功证明。CLI 只在 admit 模式增加 `--source-bindings` JSON 文件参数，读取 binding list 后传给 admit_candidates；该文件只是定位清单，不是资格证明。

- [ ] 不修改 `run_online_dagger` 的采集、pool.append、bc.train 顺序；不加入永久 fail-closed online guard；不声称本包已保护所有直接内存 BC 或 live append 入口。它们的来源闭环归 R04-live，未完成项必须写入 readiness/报告。

- [ ] GREEN/回归：

```bash
PYTHONPATH=src:.:tests PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_source_binding.py tests/test_dh116_stereo_collection.py tests/test_dh116_stereo_dagger.py tests/test_dh116_stereo_training_cli.py tests/test_dh116_stereo_training_lineage.py tests/test_dh116_stereo_bc_mode.py
```

预期 CPU 来源负向、schema-only、线上 fixture 原顺序回归通过。真实 sealed 正向必须使用已有通过原生source gate、且外部 completion 已记录 manifest 和 records artifact 的 proof；没有则报告缺真实来源，不能制造替代凭据或发起新 native。

- [ ] 提交范围：

```bash
git diff --check
git add -- src/go2_pvcnn/control/dh116/tabletop_stereo_supervision_dataset.py src/go2_pvcnn/control/dh116/tabletop_stereo_training_cli.py tests/test_dh116_stereo_source_binding.py tests/test_dh116_stereo_training_cli.py
git diff --cached
git commit -m "Verify sealed stereo sources at offline admission"
```

## Task R04-live：真实在线 producer witness 独立设计工作包

**性质：** 补充设计与实施计划；不是当前已有 API，不在 sealed-source CPU包里伪造实现，也不替代真实在线 DAgger。

**候选修改/新增位置：**

- 当前 `tabletop_stereo_native.py::NativeTensorBridge` 的真实 owner/capture 接口。
- 当前 `tabletop_stereo_teacher_collector.py::collect_step` 的实际同步前动作记录。
- 当前 `tabletop_stereo_training_cli.py::run_online_dagger` 的每轮采集完成与 optimizer 前边界。
- 候选新增 `tabletop_stereo_live_source.py`，与新文档中的原生 collector/session 组合。
- 对应新测试 `tests/test_dh116_stereo_live_source.py`；后续计划必须给完整测试fixture，不能引用尚不存在的helper。

先确认以下接口合同，才编写实现计划：

| 拟新增接口 | 创建者/用途 | 必须具备的校验 |
|---|---|---|
| LiveStereoSourceSession | 真实 native collector 工厂，绑定实际 owner/session；不是 payload constructor | 当前 BackendProfile、真实实例、Teacher/Actor源身份、session随机capability、有效reset来源；失败关闭owner，不凭bool准入 |
| seal_round(records, execution_rows) | 当前session独占写入原始chunk并返回opaque witness | 同次真实采集的深拷贝；chunk原始bytes落盘；flush/fsync；序号、byte范围、hash链、来源代码/config摘要 |
| verify_round(witness, rows, optimizer_context) | 每轮optimizer前重新校验当前session registry中的witness | owner/session与epoch不可跨用；env/episode/reset/policy/physics tick、timestamp、实际issued action、行序/行数与精确tensor全对应 |
| finalize_run(external_receipt) | native自然退出后，由外部进程观察器封存整run | 已记录live chunks/witness全覆盖、最终source manifest、自然退出与关键错误；不能倒填成功或删除失败chunk |

拟议对象只有 session 自己登记的opaque identity才能查到真实记录；把对象或hash字段复制进普通dataset不创建producer witness。不得提供 `qualified=True`、`native=True` 或 payload中“存在一个token”作为准入条件。

- [ ] 设计规定 live chunk 在 optimizer 前必须 fsync；后续如要批量同步，需额外批准最大可丢失窗口、关键提交点和崩溃恢复规则。
- [ ] 设计规定 witness 绑定本轮冻结策略版本、Teacher标签来源、完整state/action clocks；其他策略/MPC接管的动作不能冒充当前PPO sample。
- [ ] 设计规定事务顺序为 collect→own/freeze records→seal/fsync→verify witness/exact rows→optimizer→记录更新与checkpoint边界，保持同run在线更新。
- [ ] 设计规定自然退出失败使该run最终native/成功/导出资格失败；不能伪称未发生的优化，但也不能把“必须最终退出”提前为每轮不能更新的循环前置。
- [ ] 设计列出篡改/异session/异owner/跨episode/tick重复或倒退/已消费witness/未fsync/截断chunk/崩溃恢复负向测试，以及两轮实际CPU集成序列和单独有界native验收。
- [ ] 设计明确 in-memory BC/DAgger batch 通过 `source_context` 一类真实session接口验证，sealed disk重放通过外部proof验证；不能把可序列化bool当二者的统一通行证。
- [ ] 与 `tabletop_native_teacher_session.py`、`tabletop_native_teacher_collector.py` 的新架构设计协调后，更新独立规范与计划，交root审查后才实现。

完成口径：本设计工作包未批准/未实现时，R04-live保持未完成；sealed-source修复可以独立发布，但不声明在线来源闭环已完成。

## Task R04-mask：未来 masked 371D 标签账本兼容适配

这是新架构兼容任务，不是将旧全finite 371D基线判成已确认缺陷。新文档要求66项实际可观测字段、其余305项保持NaN与valid=False；不能 `nan_to_num`、补零或称为完整371D真实状态。此包只调整标签边界，旧全finite Critic输入仍受原guard保护。

**修改：**

- `src/go2_pvcnn/control/dh116/tabletop_stereo_supervision_dataset.py::validate_rows`
- `src/go2_pvcnn/control/dh116/tabletop_stereo_teacher_collector.py::PreActionState.validate`
- 新增 `tests/test_dh116_stereo_masked_label_compatibility.py`

- [ ] 写失败测试：

```python
"""Future masked label compatibility; never upgrades native/source qualification."""
import pytest
import torch

from test_dh116_stereo_collection import FakeEnv, record
from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import (
    build_dataset, validate_dataset,
)
from go2_pvcnn.control.dh116.tabletop_stereo_student_cli import _snapshot


def test_unknown_371_labels_stay_nan_and_false_in_dataset():
    data = build_dataset([record()], source_sha256="f" * 64, schema_only=True)
    for row in data["rows"]:
        row["privileged"].fill_(float("nan"))
        row["privileged_valid"].zero_()
    checked = validate_dataset(data, allow_schema_only=True)
    assert all(torch.isnan(row["privileged"]).all() for row in checked["rows"])
    assert all(not row["privileged_valid"].any() for row in checked["rows"])
    assert checked["native_qualified"] is False


def test_valid_nonfinite_label_is_still_rejected():
    data = build_dataset([record()], source_sha256="f" * 64, schema_only=True)
    data["rows"][0]["privileged"][0] = float("nan")
    data["rows"][0]["privileged_valid"][0] = True
    with pytest.raises(ValueError, match="finite|371D"):
        validate_dataset(data, allow_schema_only=True)


def test_pre_action_label_mask_allows_unknown_without_imputation():
    state = FakeEnv().snapshot()
    state.privileged.fill_(float("nan"))
    state.privileged_valid.zero_()
    state.validate()
    assert torch.isnan(state.privileged).all()
    assert not state.privileged_valid.any()


def test_existing_full_371_critic_guard_remains_closed_for_nan():
    state = FakeEnv().snapshot()
    state.privileged.fill_(float("nan"))
    state.privileged_valid.zero_()
    with pytest.raises(ValueError, match="finite unchanged native371"):
        _snapshot(state, "c" * 64)
```

当前预期：未知标签dataset与PreActionState通过测试失败；valid nonfinite仍被拒；最后一个测试当前可能在PreActionState处先失败，适配后必须到达旧Critic明确guard，不能让NaN进入该网络。

- [ ] dataset中用以下完整检查替换现有 privileged/valid 类型与全finite条件：

```python
privileged, valid = row.get("privileged"), row.get("privileged_valid")
if (
    not isinstance(privileged, torch.Tensor)
    or privileged.shape != (371,)
    or not privileged.is_floating_point()
    or not isinstance(valid, torch.Tensor)
    or valid.shape != (371,)
    or valid.dtype != torch.bool
    or valid.device != privileged.device
):
    raise ValueError("371D labels and explicit per-dimension validity required")
if bool((valid & ~torch.isfinite(privileged)).any()):
    raise ValueError("valid 371D labels must be finite")
```

- [ ] `PreActionState.validate` 的 privileged类型检查改为：

```python
_tensor(self.privileged, (n, 371), "privileged", finite=False)
```

保持已有privileged_valid布尔形状检查与共同device检查。在这些检查之后增加：

```python
if bool((self.privileged_valid & ~torch.isfinite(self.privileged)).any()):
    raise ValueError("valid 371D labels must be finite")
```

- [ ] 不删除 `tabletop_stereo_student_cli.py::_snapshot` 的全finite Critic guard；不把masked ledger送入现有固定371D Critic。未来measured66/rod模型必须有独立输入适配、模型与checkpoint身份。

- [ ] 回归命令：

```bash
PYTHONPATH=src:.:tests PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_masked_label_compatibility.py tests/test_dh116_stereo_collection.py tests/test_dh116_stereo_dagger.py tests/test_dh116_stereo_student_cli.py tests/test_dh116_stereo_canonical_field_losses.py
```

预期4个新兼容测试通过，原批次、Critic和源资格门回归通过。仅valid字段贡献loss；未知值保留NaN与false，资格保持原范围。

- [ ] 提交范围：

```bash
git diff --check
git add -- src/go2_pvcnn/control/dh116/tabletop_stereo_supervision_dataset.py src/go2_pvcnn/control/dh116/tabletop_stereo_teacher_collector.py tests/test_dh116_stereo_masked_label_compatibility.py
git diff --cached
git commit -m "Support masked stereo labels without imputing unknown state"
```

## 修订后的发布口径

本稿包含 sealed-source 两项可执行修复、一个独立live-source设计包、一个新架构masked-label兼容包。整run自然退出仍决定最终证据资格；在线每轮优化不能被改成等待自然退出。公共readiness/报告必须分别记录sealed重放是否完成、live witness是否已设计/实现、mask兼容与真实模型适配是否完成。不用离线替代完成在线目标。

## R04.1 完整上层重放回归补充（必须随 Task R04.1 一并实施）

前述 `_same` 测试不代替实际 verifier 接线测试。在同一 `tests/test_dh116_stereo_source_binding.py` 中补以下完整 fixture 和回归。此 fixture 只 mock `_native_run` 一层的 native 资格；文件 SHA、completion produced_artifacts、manifest、records 读取、行重放和比较全部真实执行。它明确只证明上层数据绑定，不能报告 native admission。

```python
import hashlib
import json

import torch

from go2_pvcnn.control.dh116 import tabletop_stereo_promotion as promotion
from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import (
    rows_from_records,
)
from test_dh116_stereo_collection import (
    FakeEnv, TinyStudent, TinyTeacher, module,
)


def sealed_cpu_upper_fixture(tmp_path, monkeypatch):
    original = module("teacher_collector").collect_step(
        FakeEnv(), TinyStudent(), TinyTeacher(),
        teacher_probability=0., schema_only=True,
    )
    data = build_dataset(
        [original], source_sha256="f" * 64, schema_only=True,
        purpose="dagger_visited_states",
    )
    records = [deepcopy(original)]
    records[0]["schema_only"] = False
    records[0]["info"]["evidence_scope"] = "native_runtime_capture_unqualified"

    def binding(path):
        return dict(
            path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text("{}\n", encoding="utf-8")
    records_path = tmp_path / "records.pt"
    torch.save(dict(
        version="dh116-stereo-teacher-candidate-records-v1",
        teacher_checkpoint_sha256="f" * 64,
        records=records,
    ), records_path)
    raw_binding = binding(raw_path)
    manifest = dict(
        version="dh116-stereo-source-manifest-v1",
        teacher_checkpoint_sha256="f" * 64,
        native_raw_artifact=raw_binding,
        records_artifact=binding(records_path),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(json.dumps(dict(
        produced_artifacts=[
            raw_binding, binding(records_path), binding(manifest_path),
        ],
    )), encoding="utf-8")
    proof = dict(
        raw_artifact=raw_binding,
        completion_artifact=binding(completion_path),
        stereo_source_manifest_artifact=binding(manifest_path),
    )
    proof_path = tmp_path / "source-proof.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    data.update(
        schema_only=False,
        rows=rows_from_records(records),
        source_bindings=[dict(
            proof_path=str(proof_path.resolve()),
            proof_sha256=binding(proof_path)["sha256"],
            row_indices=list(range(len(data["rows"]))),
        )],
    )
    monkeypatch.setattr(
        promotion, "_native_run",
        lambda *args, **kwargs: {"assumed_native_gate_only": True},
    )
    return data, proof_path, manifest_path, records_path, binding


def test_actual_upper_verifier_replays_exact_sealed_records(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import (
        verify_stereo_dataset_sources,
    )
    data, _, _, _, _ = sealed_cpu_upper_fixture(tmp_path, monkeypatch)
    assert verify_stereo_dataset_sources(data) is None


@pytest.mark.parametrize(
    "change",
    ["pixel", "action", "frame", "reset", "tick", "initial", "source_group", "order"],
)
def test_actual_upper_verifier_rejects_payload_changes(
    tmp_path, monkeypatch, change,
):
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import (
        verify_stereo_dataset_sources,
    )
    data, _, _, _, _ = sealed_cpu_upper_fixture(tmp_path, monkeypatch)
    row = data["rows"][0]
    if change == "pixel":
        row["history"]["left_rgb"].reshape(-1)[0] += .125
    elif change == "action":
        row["teacher_mean"].fill_(.9)
        row["student_nominal"].fill_(.9)
        row["issued_actions"].fill_(.9)
    elif change == "order":
        data["rows"].reverse()
    else:
        field = {
            "frame": "frame_id",
            "reset": "reset_id",
            "tick": "policy_tick",
            "initial": "physical_initial_condition_sha256",
            "source_group": "source_episode_group",
        }[change]
        value = row["identity"][field]
        row["identity"][field] = value + 1 if type(value) is int else value + "-changed"
    with pytest.raises(ValueError, match="rows differ"):
        verify_stereo_dataset_sources(data)


def test_rehashed_replacement_records_are_not_an_external_output(
    tmp_path, monkeypatch,
):
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import (
        verify_stereo_dataset_sources,
    )
    data, proof_path, manifest_path, records_path, binding = (
        sealed_cpu_upper_fixture(tmp_path, monkeypatch)
    )
    packet = torch.load(records_path, map_location="cpu", weights_only=True)
    packet["records"][0]["teacher_mean"].fill_(.9)
    replacement = tmp_path / "replacement.pt"
    torch.save(packet, replacement)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records_artifact"] = binding(replacement)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["stereo_source_manifest_artifact"] = binding(manifest_path)
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    data["source_bindings"][0]["proof_sha256"] = binding(proof_path)["sha256"]

    with pytest.raises(ValueError, match="absent from actual external process completion"):
        verify_stereo_dataset_sources(data)
```

最后一个测试故意允许重写 payload 提供的 proof/manifest SHA，但不修改原 completion；必须仍被拒收。这是“不能用新 raw 文件自签升级”的直接回归。若真实外部 native gate 不通过，正式数据仍先被该下层 gate 拒收。
