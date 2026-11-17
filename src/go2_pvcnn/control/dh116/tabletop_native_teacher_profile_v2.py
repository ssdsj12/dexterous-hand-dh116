"""Independent six-key fixed-S0 model schemas, separate from physical provenance.

BackendProfile remains the exact original source profile loaded from real files.
These versions describe the new CPU model/input contract and grant no admission.
"""
from dataclasses import asdict

from .tabletop_native_teacher_profile import (
    BackendProfile, digest, INPUT_KEYS,
    MODEL_SCHEMA_VERSION as SOURCE_MODEL_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION as SOURCE_INPUT_SCHEMA_VERSION,
    CONTEXT_SCHEMA_VERSION as SOURCE_CONTEXT_SCHEMA_VERSION,
)
from .tabletop_native_teacher_context_v2 import (
    CONTEXT_SCHEMA_VERSION_V2, CONTEXT_SCHEMA_SHA256_V2, PHASE_CONTRACT_SHA256_V2,
)
from .tabletop_native_phase_contract_v2 import PHASES_V2
from .tabletop_flexible_measured_common_teacher import common_selection

MODEL_SCHEMA_VERSION_V2 = "dh116-native-fixed-s0-context-teacher-tanh-v2"
INPUT_SCHEMA_VERSION_V2 = "dh116-native-fixed-s0-teacher-six-key-input-v2"
PHASE_SCHEMA_VERSION_V2 = "dh116.fixed_s0.execution-phase.v2"
INPUT_KEYS_V2 = tuple(INPUT_KEYS)
CONTEXT_WIDTH_V2 = 51
CPU_FIXTURE_SCHEMA_VERSION_V2 = "dh116.fixed_s0.teacher.cpu-fixture.v2"

PHASE_SCHEMA_SHA256_V2 = digest(dict(version=PHASE_SCHEMA_VERSION_V2,
    phases=list(PHASES_V2), source="execution_phase_owner_v2",
    phase_contract_sha256=PHASE_CONTRACT_SHA256_V2))
INPUT_SCHEMA_SHA256_V2 = digest(dict(version=INPUT_SCHEMA_VERSION_V2, keys=INPUT_KEYS_V2,
    raw_common_shape=["N",371], raw_common_indices=common_selection(),
    raw_common_unknown="NaN_false_305", rod_shape=["N",41,13], all_nodes_required=True,
    context_shape=["N",CONTEXT_WIDTH_V2], context_schema_sha256=CONTEXT_SCHEMA_SHA256_V2,
    phase_schema_sha256=PHASE_SCHEMA_SHA256_V2, raw_context_unknown="NaN_false", owned=True))
MODEL_SCHEMA_SHA256_V2 = digest(dict(version=MODEL_SCHEMA_VERSION_V2,
    input_schema_sha256=INPUT_SCHEMA_SHA256_V2, context_schema_sha256=CONTEXT_SCHEMA_SHA256_V2,
    phase_schema_sha256=PHASE_SCHEMA_SHA256_V2, common=[66,256,256], rod=[13,64,64],
    rod_pool="mean+max", context=[102,64,64], fusion=448, actor=24, critic=1,
    distribution="tanh_latent_gaussian", independent_branches=True))


def validate_backend_profile_v2(profile):
    """Check the original source identity without relabelling its schema fields."""
    from . import tabletop_mesh_observation_source as source
    if (type(profile) is not BackendProfile or profile.binary_sha256 != source.BACKEND_SHA
            or profile.source_tree_sha256 != source.TREE_SHA or profile.patch_sha256 != source.PATCH_SHA
            or profile.model_schema != SOURCE_MODEL_SCHEMA_VERSION
            or profile.input_schema != SOURCE_INPUT_SCHEMA_VERSION
            or profile.context_schema != SOURCE_CONTEXT_SCHEMA_VERSION
            or profile.digest != digest({k:v for k,v in asdict(profile).items() if k != "digest"})):
        raise ValueError("current immutable actual BackendProfile identity required")
    return profile
