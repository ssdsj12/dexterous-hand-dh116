以下内容可合入 master 实施计划。仅为计划，未写入文件、修改源码或执行提交。核对基线为 `5b10a4c2f457e965e24ec79ce7033374d46c1852`。

# R01 / R02 / R05 Student 执行链修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 统一三种 Actor 模式的训练输入、371D 字段损失和正式 Stereo 准入。

**Architecture:** Actor 从同一次 estimator 结果构造动作特征；BC 和独立状态监督复用字段损失；collection 和 Student PPO 复用 matcher 比较。原生证明校验、输入类型边界、NA 和 Force Proxy 门控保持独立。

**Tech Stack:** Python 3.10、PyTorch、pytest；仅 CPU 合成样本验证。

执行位置为 master 分配的 Ubuntu 隔离 worktree。以下路径相对该 worktree。每包开始检查分支、HEAD、状态，执行 master 中的保护清单核验。测试环境统一为：

```bash
export PYTHONPATH=src:.:tests
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_VISIBLE_DEVICES=
```

R05 属于 A 阶段；R01、R02 属于 B 阶段。R02、R05 都修改 `tabletop_stereo_training_cli.py`，必须串行集成该文件。

---

## R01：BC / DAgger 复用 Actor 的一次估计结果

**修改：**

- `src/go2_pvcnn/control/dh116/tabletop_stereo_state_estimator.py`
- `src/go2_pvcnn/control/dh116/tabletop_stereo_bc.py`

**新增测试：**

- `tests/test_dh116_stereo_bc_actor_inputs.py`

### 1. 写失败测试

- [ ] 新建以下测试文件：

```python
"""CPU composition regressions; no native or successful-data qualification."""
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_stereo_bc import StereoBCTrainer
from go2_pvcnn.control.dh116.tabletop_stereo_dagger_trainer import StereoDAggerTrainer
from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_stereo_actor
from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import (
    batch_dataset, build_dataset,
)
from go2_pvcnn.control.dh116.tabletop_stereo_contract import state_contract
from test_dh116_stereo_collection import record
from test_dh116_stereo_estimator import history

torch.set_num_threads(1)

MODES = ("latent", "estimated_task_state", "hybrid")


def actor_for(mode):
    row = state_contract()["dimensions"][116]
    return make_stereo_actor(dict(
        embed_dim=32,
        history=4,
        num_heads=4,
        num_layers=1,
        actor_input_mode=mode,
        task_state_config=dict(
            allowlist=[116],
            source_qualification={
                key: True for key in row["required_validity_keys"]
            },
        ),
    ))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("trainer_type", [StereoBCTrainer, StereoDAggerTrainer])
def test_supervised_step_uses_selected_actor_inputs_once(mode, trainer_type):
    torch.manual_seed(116)
    data = build_dataset(
        [record()], source_sha256="f" * 64, schema_only=True,
    )
    batch = batch_dataset(data, [0])
    actor = actor_for(mode).eval()
    original_modes = [module.training for module in actor.modules()]
    before = actor.action_net[-1].weight.detach().clone()
    calls = []
    hook = actor.estimator.register_forward_hook(
        lambda module, arguments, output: calls.append(output)
    )
    try:
        report = trainer_type(
            actor, {"lambda_state": 0.}, schema_only=True,
        ).step(batch)
    finally:
        hook.remove()

    assert len(calls) == 1
    assert report["steps"] == 1
    assert report["gradient_norms"]["action"] > 0
    assert report["native_qualified"] is False
    assert report["ppo_storage_modified"] is False
    assert not torch.equal(before, actor.action_net[-1].weight)
    assert [module.training for module in actor.modules()] == original_modes

    if mode != "latent":
        gradient = actor.estimator.heads["privileged_state"].weight.grad
        assert gradient is not None
        assert gradient[116].abs().sum() > 0
        assert gradient[180:200].abs().sum() == 0


def test_latent_feature_refactor_preserves_exact_action_math():
    torch.manual_seed(116)
    actor = actor_for("latent").eval()
    sensors = history(1)
    with torch.no_grad():
        estimates = actor.estimator(sensors)
        expected = actor.action_net(estimates["latent"])
        features = actor.features_from_estimates(estimates, sensors)
        actual = actor.latent_mean(sensors)

    assert torch.equal(features, estimates["latent"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

- [ ] 运行 RED：

```bash
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_bc_actor_inputs.py
```

当前预期：

- 四个非 latent 组合在矩阵乘法失败。
- latent 精确数学测试因缺少 `features_from_estimates` 失败。
- 两个 latent optimizer 组合应已通过。

### 2. 最小补丁

- [ ] 在 `StereoStudentActor` 中增加方法，并替换 `latent_mean`：

```python
def features_from_estimates(self, estimates, history):
    features = estimates["latent"]
    if self.input_mode != "latent":
        selected = {
            key: value for key, value in estimates.items()
            if key != "world_auxiliary"
        }
        task = self.task_state_adapter(selected, history)
        features = task["actor_features"]
        if self.input_mode == "hybrid":
            features = torch.cat((estimates["latent"], features), -1)
    return features


def latent_mean(self, history):
    estimates = self.estimator(history)
    return self.action_net(self.features_from_estimates(estimates, history))
```

公开推理入口仍接收 `StereoHistoryBatch`；该方法只复用内部 estimator 输出，不增加 Critic 或 GT 输入。

- [ ] 将 BC 当前直接读取 latent 的语句替换为：

```python
output = self.actor.estimator(history)
features = self.actor.features_from_estimates(output, history)
nominal = self.actor.action_net(features).tanh()
```

后续 state loss 继续使用同一个 `output`。不重新调用 `actor.deterministic(history)`。

### 3. GREEN 和回归

- [ ] 运行：

```bash
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_bc_actor_inputs.py tests/test_dh116_stereo_bc_mode.py tests/test_dh116_stereo_runtime_actor_modes.py tests/test_dh116_stereo_collection.py tests/test_dh116_stereo_dagger.py tests/test_dh116_stereo_student_ppo.py tests/test_dh116_stereo_runtime_rollout_storage.py
```

预期全部通过；记录实际数量。新增组合测试每次 optimizer 更新只调用一次 estimator，原 latent 动作逐位一致。

### 4. 范围提交

- [ ] 审查并仅暂存：

```bash
git diff --check
git add -- src/go2_pvcnn/control/dh116/tabletop_stereo_state_estimator.py src/go2_pvcnn/control/dh116/tabletop_stereo_bc.py tests/test_dh116_stereo_bc_actor_inputs.py
git diff --cached --stat
git diff --cached
git commit -m "Fix supervised updates for all stereo Actor input modes"
```

使用独立新训练输出；不覆盖或迁移旧 Checkpoint。由 master 补发布报告和公共登记文档。

---

## R02：统一 canonical 371D 分类、回归和旋转损失

**修改：**

- `src/go2_pvcnn/control/dh116/tabletop_stereo_losses.py`
- `src/go2_pvcnn/control/dh116/tabletop_stereo_state_supervision.py`
- `src/go2_pvcnn/control/dh116/tabletop_stereo_bc.py`
- `src/go2_pvcnn/control/dh116/tabletop_stereo_training_cli.py`

**新增测试：**

- `tests/test_dh116_stereo_canonical_field_losses.py`

### 1. 写失败测试

- [ ] 新建以下测试文件：

```python
"""Canonical field-loss mathematics; CPU masks are not native evidence."""
import math

import pytest
import torch
from torch.nn import functional as F

from go2_pvcnn.control.dh116.tabletop_stereo_losses import StereoEstimationLoss
from go2_pvcnn.control.dh116.tabletop_stereo_state_supervision import _prediction_loss

CLASSIFICATION = (
    149, 150, 151, 152, 255, 256, 257, 258, 303, 304,
    309, 310, 311, 312, 313, 314, 315, 316, 317, 318,
)


def sources_for(criterion):
    sources = {
        key: torch.ones(1, dtype=torch.bool)
        for key in criterion.required
    }
    sources["rotation_symmetry_valid"] = torch.ones(1, dtype=torch.bool)
    return sources


def evaluate(criterion, prediction, target, mask, sources):
    return criterion(
        {"privileged_state": prediction},
        {"privileged_state": target},
        {"privileged_state": mask, "state_sources": sources},
    )


@pytest.mark.parametrize("index", CLASSIFICATION)
@pytest.mark.parametrize("label", [0., 1.])
def test_every_canonical_classification_field_uses_bce(index, label):
    criterion = StereoEstimationLoss()
    assert criterion.state_allowed[index]
    prediction = torch.zeros(1, 371, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0, index] = label
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, index] = True

    result = evaluate(
        criterion, prediction, target, mask, sources_for(criterion),
    )
    independent, _ = _prediction_loss(
        {"privileged_state": prediction},
        {"privileged_state": (target, mask)},
        criterion,
    )

    assert float(result["total"]) == pytest.approx(math.log(2.))
    torch.testing.assert_close(result["total"], independent)
    result["total"].backward()
    assert float(prediction.grad[0, index]) == pytest.approx(.5 - label)
    other = prediction.grad.detach().clone()
    other[0, index] = 0
    assert torch.count_nonzero(other) == 0


def test_unknown_nan_fields_have_zero_loss_and_gradient():
    criterion = StereoEstimationLoss()
    prediction = torch.full((1, 371), float("nan"), requires_grad=True)
    target = torch.full_like(prediction, float("nan"))
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, 149] = True
    sources = sources_for(criterion)
    sources["source_valid"].zero_()

    result = evaluate(criterion, prediction, target, mask, sources)
    assert float(result["total"]) == 0.
    assert result["coverage"]["privileged_state"] == 0.
    result["total"].backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) == 0


def test_regression_field_keeps_smooth_l1_scale():
    criterion = StereoEstimationLoss()
    prediction = torch.zeros(1, 371, requires_grad=True)
    target = torch.zeros_like(prediction)
    prediction.data[0, 116] = .2
    target[0, 116] = .8
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, 116] = True
    result = evaluate(
        criterion, prediction, target, mask, sources_for(criterion),
    )
    expected = F.smooth_l1_loss(
        prediction[:, 116], target[:, 116],
    )
    torch.testing.assert_close(result["total"], expected)


def test_shared_quaternion_loss_remains_sign_invariant():
    criterion = StereoEstimationLoss()
    lo, hi = criterion.rotations[0]["slice"]
    prediction = torch.zeros(1, 371, requires_grad=True)
    prediction.data[0, lo:hi] = torch.tensor([1., .2, 0., 0.])
    target = torch.zeros_like(prediction)
    target[0, lo] = 1.
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[0, lo:hi] = True

    positive = criterion.canonical_state_loss(prediction, target, mask)
    negative = criterion.canonical_state_loss(prediction, -target, mask)
    torch.testing.assert_close(positive, negative)
    positive.backward()
    assert torch.isfinite(prediction.grad).all()


def test_new_loss_semantics_has_explicit_version():
    assert (
        StereoEstimationLoss().loss_version
        == "dh116-stereo-canonical-field-loss-v2"
    )
```

- [ ] 运行 RED：

```bash
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_canonical_field_losses.py
```

当前预期：40 个分类组合的 BCE 数值断言失败；共享方法和新版本测试因接口不存在失败。未知值和回归测试用于保持原语义。

### 2. 提取共享数学

- [ ] 在 losses 模块增加：

```python
STATE_LOSS_VERSION = "dh116-stereo-canonical-field-loss-v2"
```

- [ ] 在 `StereoEstimationLoss.__init__` 的 `super().__init__()` 后增加：

```python
self.loss_version = STATE_LOSS_VERSION
```

- [ ] 在该类中增加：

```python
def canonical_state_loss(self, prediction, target, mask):
    if prediction.ndim != 2 or prediction.shape[1] != 371:
        raise ValueError("state prediction shape must be [B,371]")
    mask = _mask(mask, prediction)
    scalar_mask = mask.clone()
    rotation_losses = []
    for group in self.rotations:
        lo, hi = group["slice"]
        group_mask = mask[:, lo:hi].all(-1)
        scalar_mask[:, lo:hi] = False
        rotation_losses.append(quaternion_geodesic_loss(
            prediction[:, lo:hi], target[:, lo:hi], group_mask,
        ))
    classification = self.state_classification.to(
        prediction.device,
    )[None].expand_as(prediction)
    regression_loss = masked_smooth_l1(
        prediction, target, scalar_mask & ~classification,
    )
    classification_loss = masked_bce(
        prediction, target, scalar_mask & classification,
    )
    return (
        regression_loss + classification_loss
        + sum(rotation_losses, prediction.new_zeros(()))
    )
```

此方法只负责数学；调用方先完成来源和对称性资格处理。现有 `_safe_pair` 保持未知 NaN 的安全屏蔽。

- [ ] 将 `forward` 中整个 `privileged_state` 分支替换为：

```python
if name == "privileged_state":
    mask = self.state_mask(pred, validity)
    for group in self.rotations:
        lo, hi = group["slice"]
        group_mask = mask[:, lo:hi].all(-1)
        if group.get("symmetry_mask_required"):
            symmetry = validity.get("state_sources", {}).get(
                "rotation_symmetry_valid",
            )
            if symmetry is None:
                group_mask &= False
            elif (
                not isinstance(symmetry, torch.Tensor)
                or symmetry.shape != (len(pred),)
                or symmetry.dtype != torch.bool
                or symmetry.device != pred.device
            ):
                raise ValueError(
                    "rotation_symmetry_valid must be boolean [B]"
                )
            else:
                group_mask &= symmetry
        mask[:, lo:hi] = group_mask[:, None]
    loss = self.canonical_state_loss(pred, target, mask)
```

保留后续 `force_proxy`、额外 logits、coverage 和 NA 分支。

- [ ] 在 `tabletop_stereo_state_supervision.py::_prediction_loss` 中，将原803–810行替换为：

```python
classification = criterion.state_classification[None].to(pred.device)
terms.append(criterion.canonical_state_loss(pred, target, mask))
for group in criterion.rotations:
    lo, hi = group["slice"]
    scalar[:, lo:hi] = False
```

紧随其后的 sigmoid、旋转误差与 residual 报告逻辑保留。移除该处重复的 geodesic、Smooth L1 和 BCE 训练损失计算，避免加算两次。

### 3. 记录独立损失版本

- [ ] 在 BC `_step` 返回的报告字段中增加：

```python
state_loss_version=self.state_loss.loss_version,
```

- [ ] 将 `train_bc` 的合同构造替换为：

```python
contract = make_contract(
    "bc",
    dict(
        network=network,
        bc=deepcopy(bc_config),
        state_loss_version=trainer.state_loss.loss_version,
        dataset_contract_sha256=data["contract"]["contract_sha256"],
        schema_only=data["schema_only"],
        split=split,
        provenance=provenance,
        preprocessor=preprocessor,
    ),
    calibration_sha256=calibration,
)
```

- [ ] state supervision 模块从 losses 导入 `STATE_LOSS_VERSION`；在 `run_training_mode` 的 `report` 字典和 `main` 保存的 `state_estimator.pt` 字典中分别增加：

```python
state_loss_version=STATE_LOSS_VERSION,
```

新训练使用新目录和合同；不改已有配置、数据或历史模型。

### 4. GREEN 和回归

- [ ] 运行：

```bash
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_canonical_field_losses.py tests/test_dh116_stereo_estimator.py tests/test_dh116_stereo_state_supervision.py tests/test_dh116_stereo_state_training_modes.py tests/test_dh116_stereo_runtime_task_state.py tests/test_dh116_stereo_bc_actor_inputs.py tests/test_dh116_stereo_collection.py tests/test_dh116_stereo_training_cli.py
```

预期全部通过。除新增44项外，重点确认现有 quaternion、NA、未资格化 force、source mask 和原生准入负向测试保持通过；实际总数以运行输出记录。

### 5. 范围提交

- [ ] 审查并仅暂存：

```bash
git diff --check
git add -- src/go2_pvcnn/control/dh116/tabletop_stereo_losses.py src/go2_pvcnn/control/dh116/tabletop_stereo_state_supervision.py src/go2_pvcnn/control/dh116/tabletop_stereo_bc.py src/go2_pvcnn/control/dh116/tabletop_stereo_training_cli.py tests/test_dh116_stereo_canonical_field_losses.py
git diff --cached --stat
git diff --cached
git commit -m "Unify canonical stereo field loss semantics"
```

由 master 补发布报告和公共登记文档。

---

## R05：Student 与 collection 共用 matcher 准入比较

**新增：**

- `src/go2_pvcnn/control/dh116/tabletop_stereo_admission.py`
- `tests/test_dh116_stereo_admission_matcher.py`

**修改：**

- `src/go2_pvcnn/control/dh116/tabletop_stereo_training_cli.py`
- `src/go2_pvcnn/control/dh116/tabletop_stereo_student_cli.py`

使用新模块保存准入比较，不改 `tabletop_stereo_preprocessing.py`，避免仅提取 gate 就改变所有 preprocessing 的源摘要。

### 1. 写失败测试

- [ ] 新建以下测试文件：

```python
"""Shared admission comparison; external proof verification is mocked explicitly."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from go2_pvcnn.control.dh116 import tabletop_stereo_promotion as promotion
from go2_pvcnn.control.dh116 import tabletop_stereo_student_cli as student
from go2_pvcnn.control.dh116 import tabletop_stereo_training_cli as collection
from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116.tabletop_stereo_preprocessing import processing_contract

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def admission(tmp_path, monkeypatch):
    camera = ROOT / "configs/dh116_training/stereo_camera_v1.json"
    calibration = StereoCalibration.from_json(camera)
    teacher_checkpoint = tmp_path / "teacher.pt"
    teacher_checkpoint.write_bytes(b"CPU metadata fixture; no native proof")
    proof = dict(
        calibration_sha256=calibration.sha256,
        depth_estimator_config=deepcopy(
            processing_contract(camera)["depth_estimator"]
        ),
    )
    monkeypatch.setattr(
        promotion, "require_teacher_qualification",
        lambda *args, **kwargs: {"mock_only": True},
    )
    monkeypatch.setattr(
        promotion, "verify_stereo_qualification",
        lambda *args, **kwargs: deepcopy(proof),
    )

    def call(entry):
        if entry == "collection":
            return collection._formal_collection_admission(
                {}, teacher_checkpoint, {}, camera,
            )
        arguments = SimpleNamespace(
            teacher_proof={},
            teacher_checkpoint=teacher_checkpoint,
            stereo_proof={},
            checkpoint=None,
            actor_package=tmp_path / "bound_actor.pt",
            camera_config=camera,
        )
        return student._formal_admission(arguments, calibration)

    return call, proof


@pytest.mark.parametrize("entry", ["student", "collection"])
def test_same_calibration_changed_matcher_is_rejected(admission, entry):
    call, proof = admission
    matcher = proof["depth_estimator_config"]
    matcher["subpixel"] = not matcher["subpixel"]
    with pytest.raises(ValueError, match="matcher differs"):
        call(entry)


@pytest.mark.parametrize("entry", ["student", "collection"])
def test_missing_matcher_cannot_reuse_calibration_only_proof(admission, entry):
    call, proof = admission
    proof.pop("depth_estimator_config")
    with pytest.raises(ValueError, match="matcher differs"):
        call(entry)


@pytest.mark.parametrize("entry", ["student", "collection"])
@pytest.mark.parametrize("resource_batch", [1, 99])
def test_resource_batch_does_not_change_matcher_semantics(
    admission, entry, resource_batch,
):
    call, proof = admission
    proof["depth_estimator_config"]["image_processing_batch_size"] = (
        resource_batch
    )
    result = call(entry)
    assert result["stereo"]["calibration_sha256"] == proof["calibration_sha256"]


@pytest.mark.parametrize("entry", ["student", "collection"])
def test_calibration_mismatch_remains_rejected(admission, entry):
    call, proof = admission
    proof["calibration_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="calibration differs"):
        call(entry)


@pytest.mark.parametrize("entry", ["student", "collection"])
def test_external_stereo_proof_verifier_is_still_required(
    admission, entry, monkeypatch,
):
    call, _ = admission

    def reject(*args, **kwargs):
        raise ValueError("genuine stereo proof is missing")

    monkeypatch.setattr(
        promotion, "verify_stereo_qualification", reject,
    )
    with pytest.raises(ValueError, match="genuine stereo proof is missing"):
        call(entry)
```

- [ ] 运行 RED：

```bash
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_admission_matcher.py
```

当前预期：Student 的“换 matcher”“缺 matcher”两项失败，因为没有抛错；collection 对应负向项和其余保持项通过。测试使用 mock 只隔离配置比较，不授予原生资格。

### 2. 最小共享补丁

- [ ] 新建 admission 模块：

```python
"""Compare an externally verified stereo proof with actual processing semantics.

This comparison does not verify native evidence or grant qualification.
"""
from copy import deepcopy

from .tabletop_stereo_preprocessing import processing_contract


def require_qualified_stereo_processing(verified_stereo, camera_config):
    actual = processing_contract(camera_config)
    if (
        not isinstance(verified_stereo, dict)
        or verified_stereo.get("calibration_sha256")
        != actual["calibration_sha256"]
    ):
        raise ValueError(
            "formal stereo proof calibration differs from selected camera"
        )
    measured = deepcopy(verified_stereo.get("depth_estimator_config"))
    if not isinstance(measured, dict):
        raise ValueError(
            "formal matcher differs from verified raw stereo processing"
        )
    measured.pop("image_processing_batch_size", None)
    if measured != actual["depth_estimator"]:
        raise ValueError(
            "formal matcher differs from verified raw stereo processing"
        )
    return actual
```

- [ ] 将 collection 的 `_formal_collection_admission` 替换为：

```python
def _formal_collection_admission(
    teacher_proof, teacher_checkpoint, stereo_proof, camera_config,
):
    from .tabletop_stereo_promotion import (
        require_teacher_qualification, verify_stereo_qualification,
    )
    from .tabletop_stereo_admission import (
        require_qualified_stereo_processing,
    )

    teacher = require_teacher_qualification(
        teacher_proof, file_sha256(teacher_checkpoint),
    )
    stereo = verify_stereo_qualification(stereo_proof)
    require_qualified_stereo_processing(stereo, camera_config)
    return dict(teacher=teacher, stereo=stereo)
```

- [ ] 将 Student 的 `_formal_admission` 替换为：

```python
def _formal_admission(args, calibration):
    from .tabletop_stereo_promotion import (
        require_teacher_qualification, verify_stereo_qualification,
    )
    from .tabletop_stereo_admission import (
        require_qualified_stereo_processing,
    )

    if (
        args.teacher_proof is None
        or args.teacher_checkpoint is None
        or args.stereo_proof is None
    ):
        raise ValueError(
            "formal Student PPO requires Teacher proof + actual Teacher "
            "checkpoint + stereo qualification proof"
        )
    teacher = require_teacher_qualification(
        args.teacher_proof, file_sha256(args.teacher_checkpoint),
    )
    stereo = verify_stereo_qualification(args.stereo_proof)
    if stereo["calibration_sha256"] != calibration.sha256:
        raise ValueError(
            "formal stereo qualification calibration differs from selected camera"
        )
    require_qualified_stereo_processing(stereo, args.camera_config)
    if args.checkpoint is None and args.actor_package is None:
        raise ValueError(
            "formal Student PPO requires explicit actor-package warm start "
            "or exact Student resume"
        )
    return dict(teacher=teacher, stereo=stereo)
```

Teacher 和 Stereo 的原始 proof verifier 仍先执行；actor package 的处理合同仍由现有 `_warmstart_processing` / `require_raw_processing` 校验。选中 actor package 不能跳过新比较。

### 3. GREEN 和回归

- [ ] 运行：

```bash
/mnt/mydisk/xk/envs/dh116/bin/python -m pytest -q -p no:cacheprovider tests/test_dh116_stereo_admission_matcher.py tests/test_dh116_stereo_training_cli.py tests/test_dh116_stereo_student_cli.py tests/test_dh116_stereo_preprocessing.py tests/test_dh116_stereo_student_ppo.py
```

预期全部通过；新增12项通过。现有正式 gate、绑定预处理、actor-only 推理和诊断资源限制继续通过。记录实际总数，不将 mock 或 CPU 结果写成 native 资格。

### 4. 范围提交

- [ ] 审查并仅暂存：

```bash
git diff --check
git add -- src/go2_pvcnn/control/dh116/tabletop_stereo_admission.py src/go2_pvcnn/control/dh116/tabletop_stereo_training_cli.py src/go2_pvcnn/control/dh116/tabletop_stereo_student_cli.py tests/test_dh116_stereo_admission_matcher.py
git diff --cached --stat
git diff --cached
git commit -m "Bind formal Student admission to verified stereo matcher"
```

由 master 补发布报告和公共登记文档。三个工作包均不启动 native、不消耗原生重试预算、不修改物理合同或历史模型。
