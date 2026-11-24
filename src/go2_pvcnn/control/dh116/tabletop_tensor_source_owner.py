"""New flat-candidate source and smoke reference owner, explicitly unqualified.

The original SharedTeacherSource class gate is untouched. This owner accepts
the new TensorRodOwner and a main-program profile with verify()/digest; it
reuses the old boundary precision and actual material validation functions.
Actual parameter snapshots are read only at startup and selected Reset.
Smoke contexts and action history stay on device, with no lane tensor readback.
"""
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import uuid

import numpy as np
import torch

from .tabletop_native_teacher_profile import ROOT, digest
from .tabletop_native_teacher_context import ContextIdentity, SMOKE_EXCLUDED
from .tabletop_shared_teacher_source import SharedTeacherSource, SharedSourceIdentity
from .tabletop_tensor_rod_owner import TensorRodOwner


CANDIDATE_MANIFEST = ROOT / "docs/contracts/dh116_flat_backend_candidate_v1.json"
_EXCLUDED = (*SMOKE_EXCLUDED, 8, 9, 46)
_REQUIRED = tuple(i for i in range(50) if i not in _EXCLUDED)


class TensorSourceOwner:
    """Owner of source identities and GPU smoke reset/action references.

    ``profile`` is an explicit new candidate profile supplied by the driver.
    Its digest is the semantic digest(json_manifest), as in the original shared
    source contract. Profile verification and the original native parameter
    validator run at construction/Reset, never at substep/context capture.

    ``contexts()`` returns (float32[N,50], bool[N,50], ContextIdentity tuple).
    ``reset(env_ids, distance_m)`` accepts K primitive row IDs and a device
    floating [K] tensor measured at each actual fresh Reset. Only those GPU
    reference rows change. Wall time remains external and elapsed time comes
    from the original SDK-derived policy timestamp, not a fitted offset.
    """

    def __init__(self, runtime, sim, profile, env_origins, device, grasp,
                 episode_cap_s, wall_cap_s, *, scope=None, rod_uids=None):
        if type(runtime) is not TensorRodOwner:
            raise ValueError("actual new TensorRodOwner required; legacy source gate is unchanged")
        if (grasp not in ("pinch", "tripod", "power")
                or any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0
                       for x in (episode_cap_s, wall_cap_s))):
            raise ValueError("original grasp and finite source-owned episode/wall caps required")
        self.runtime, self.sim, self.profile = runtime, sim, profile
        self.device = torch.device(device)
        self.scope = ("cpu_fixture" if self.device.type == "cpu" else "owner_source_unqualified") if scope is None else scope
        if self.scope not in ("cpu_fixture", "owner_source_unqualified"):
            raise ValueError("explicit fixture or unqualified owner source scope required")
        if runtime.device != self.device:
            raise ValueError("source and actual tensor rod owner device differ")
        self._profile_sha256 = getattr(profile, "digest", None)
        self._verify_profile()
        n = runtime.solver.num_envs
        self.n = n
        self._parent_solver_id = id(runtime.solver)
        self.rod_uids = tuple("rod" + str(i) for i in range(n)) if rod_uids is None else tuple(rod_uids)
        if (len(self.rod_uids) != n or len(set(self.rod_uids)) != n
                or any(type(x) is not str or not x.strip() for x in self.rod_uids)):
            raise ValueError("one source-owned material rod identity per actual parent row required")
        self._origins = torch.as_tensor(env_origins, device=self.device, dtype=torch.float64).detach().clone()
        if (self._origins.shape != (n, 3) or not bool(torch.isfinite(self._origins).all())
                or not bool((self._origins[:, 2] == 0).all())):
            raise ValueError("source-owned finite horizontal environment origins required")
        if self.scope != "cpu_fixture":
            loaded = sys.modules.get("ropesim._ropesim")
            manifest = json.loads(CANDIDATE_MANIFEST.read_text())
            if (loaded is None or not getattr(loaded, "__file__", None)
                    or Path(loaded.__file__).resolve() != Path(manifest["binary_path"]).resolve()
                    or type(runtime.solver) is not getattr(loaded, "RopeBatch", None)):
                raise ValueError("loaded actual native flat candidate shared solver required")
        self.grasp = grasp
        self.episode_cap_s, self.wall_cap_s = float(episode_cap_s), float(wall_cap_s)
        self.started = perf_counter()
        self.owner_uid = uuid.uuid4().hex
        self._reset_identities = [None] * n
        self._previous_tokens = [None] * n
        self._reset_z = torch.full((n, 2), torch.nan, dtype=torch.float64, device=self.device)
        self._reset_distance = torch.full((n,), torch.nan, dtype=torch.float64, device=self.device)
        self._reference_sdk = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        self._reference_policy_time = torch.full((n,), torch.nan, dtype=torch.float64, device=self.device)
        self._previous = torch.full((n, 24), torch.nan, dtype=torch.float32, device=self.device)
        boundary = self._boundary()
        self._rest, self._rest_total = self._read_parameters(boundary)
        if self._boundary() != boundary:
            raise ValueError("source boundary moved during startup parameter capture")

    @property
    def native_qualified(self): return False
    @property
    def teacher_admission(self): return False
    @property
    def native_qualification(self): return False
    @property
    def origins(self): return self._origins.clone()
    @property
    def rest_lengths_m(self): return self._rest.clone()
    @property
    def rest_length_m(self): return self._rest_total.clone()
    @property
    def reset_z(self): return self._reset_z.clone()
    @property
    def reset_distance_m(self): return self._reset_distance.clone()
    @property
    def distance_ref_m(self): return self._reset_distance.clamp_min(.005)
    @property
    def previous_request(self): return self._previous.clone()
    @property
    def previous_request_tokens(self): return tuple(self._previous_tokens)
    @property
    def reference_sdk(self): return self._reference_sdk.clone()

    def _verify_profile(self):
        verify = getattr(self.profile, "verify", None)
        sha = getattr(self.profile, "digest", None)
        if (not callable(verify) or type(sha) is not str or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError("explicit verifiable flat candidate profile digest required")
        verify()
        manifest = json.loads(CANDIDATE_MANIFEST.read_text())
        if (manifest.get("schema") != "dh116-flat-backend-candidate-v1"
                or manifest.get("teacher_admission") is not False
                or manifest.get("native_mechanical_receipt") is not None
                or manifest.get("physics_kernel_changes") is not False
                or manifest.get("flat_abi") != "contact_pair_samples_flat_cuda_v1"
                or digest(manifest) != sha or sha != self._profile_sha256):
            raise ValueError("strictly unqualified source-locked flat candidate profile required")

    def _boundary(self):
        if (getattr(self.profile, "digest", None) != self._profile_sha256
                or id(self.runtime.solver) != self._parent_solver_id
                or self.runtime.solver.num_envs != self.n):
            raise ValueError("candidate profile or actual parent solver changed")
        # Reuse the original code verbatim: nominal 480Hz, native fp32 callback
        # timer, raw accumulated SDK clock, solver health/time and Reset epochs.
        return SharedTeacherSource._boundary(self)

    def _read_parameters(self, boundary):
        raw = self.runtime.solver.parameter_snapshot()
        # Original code validates actual snapshot provenance, free nodes,
        # gravity, radius, friction, EA/EI and reconstructed device node masses.
        rest = SharedTeacherSource._parameters(self, raw, boundary)
        # Preserve the oracle's np.asarray(raw_rest).sum() precision/order.
        total = torch.as_tensor(np.asarray(raw["rest_lengths_m"]).sum(axis=1),
                                dtype=torch.float64, device=self.device).clone()
        return rest, total

    def _selected(self, env_ids):
        ids = tuple(env_ids)
        if (not ids or any(type(i) is not int or not 0 <= i < self.n for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("unique actual selected Reset row IDs required")
        return ids, torch.tensor(ids, device=self.device, dtype=torch.int64)

    def _floating(self, value, shape, name):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.device != self.device or not value.is_floating_point()):
            raise ValueError("actual same-device " + name + " tensor required")
        return value.detach()

    @staticmethod
    def _identities(boundary):
        return tuple(ContextIdentity(boundary.run_uuid, i, boundary.episode_uids[i],
            boundary.reset_ids[i], boundary.rod_uids[i], boundary.sdk_step,
            boundary.rod_ticks[i], boundary.policy_time_s, boundary.solver_time_s)
            for i in range(boundary.parent_num_envs))

    def reset(self, env_ids, distance_m):
        ids, selected = self._selected(env_ids)
        distance = self._floating(distance_m, (len(ids),), "measured Reset distance").double()
        if bool((~torch.isfinite(distance) | (distance < 0)).any()):
            raise ValueError("actual finite nonnegative measured Reset distance required")
        self._verify_profile()
        boundary = self._boundary()
        identities = self._identities(boundary)
        for lane in ids:
            prior = self._reset_identities[lane]
            current = identities[lane]
            if (current.rod_tick != 0 or prior is not None and (
                    current.reset_id == prior.reset_id or current.episode_uid == prior.episode_uid)):
                raise ValueError("fresh zero-tick nodes and a new native Reset epoch required")
        rest, total = self._read_parameters(boundary)
        keep = torch.ones(self.n, dtype=torch.bool, device=self.device)
        keep[selected] = False
        if not torch.equal(rest[keep], self._rest[keep]) or not torch.equal(total[keep], self._rest_total[keep]):
            raise ValueError("selected Reset changed peer material rest references")
        p = torch.as_tensor(self.runtime.solver.positions(), device=self.device).detach()
        if p.shape != (self.n, 41, 3):
            raise ValueError("actual complete source-owned Reset nodes required")
        positions = p[selected]
        z = torch.stack((positions[:, :, 2].median(dim=1).values,
                         positions[:, :, 2].amin(dim=1)), dim=1).double()
        if not bool(torch.isfinite(positions).all()):
            raise ValueError("actual complete finite selected Reset node positions required")
        if self._boundary() != boundary:
            raise ValueError("source boundary moved during selected Reset capture")
        self._reset_z[selected] = z
        self._reset_distance[selected] = distance
        self._reference_sdk[selected] = boundary.sdk_step
        self._reference_policy_time[selected] = boundary.policy_time_s
        self._previous[selected] = 0.  # Original measured-reset zero request.
        self._rest[selected] = rest[selected]
        self._rest_total[selected] = total[selected]
        for lane in ids:
            self._reset_identities[lane] = identities[lane]
            self._previous_tokens[lane] = "measured-reset:" + identities[lane].episode_uid
        return tuple(identities[i] for i in ids)

    def set_previous_request(self, actions, token):
        values = self._floating(actions, (self.n, 24), "normalized actual request")
        if (type(token) is not str or not token.strip()
                or bool((~torch.isfinite(values) | (values.abs() > 1)).any())):
            raise ValueError("one actual finite bounded normalized request per row and token required")
        if any(c is None for c in self._reset_identities):
            raise ValueError("verified Reset references required before recording requests")
        self._boundary()
        self._previous = values.float().clone()
        self._previous_tokens = [token] * self.n

    def contexts(self):
        boundary = self._boundary()
        identities = self._identities(boundary)
        for current, reset in zip(identities, self._reset_identities):
            if reset is None or current.epoch != reset.epoch:
                raise ValueError("new native Reset epoch requires fresh source references")
        remaining = max(0., self.wall_cap_s - (perf_counter() - self.started)) / self.wall_cap_s
        if not math.isfinite(remaining) or not 0. <= remaining <= 1.:
            raise ValueError("finite original external budget clock required")
        values = torch.full((self.n, 50), torch.nan, dtype=torch.float32, device=self.device)
        valid = torch.zeros((self.n, 50), dtype=torch.bool, device=self.device)
        valid[:, _REQUIRED] = True
        values[:, :4] = values.new_tensor([1., 0., 0., 0.])
        values[:, 10:12] = 0.
        values[:, 12:36] = self._previous
        values[:, 36:38] = self._reset_z.float()
        values[:, 38] = self._reset_distance.clamp_min(.005).float()
        values[:, 39] = self._reset_distance.float()
        values[:, 40:43] = values.new_tensor([.03, 5., .0005])
        values[:, 43] = ((boundary.policy_time_s - self._reference_policy_time) / self.episode_cap_s).float()
        values[:, 44] = remaining
        values[:, 45] = 0.
        values[:, 48:50] = 0.
        if bool(((valid & ~torch.isfinite(values)) | (~valid & ~torch.isnan(values))).any()):
            raise ValueError("finite actual context with explicit NA values required")
        if self._boundary() != boundary:
            raise ValueError("source boundary moved during tensor context capture")
        return values, valid, identities

    def validate_fresh(self, packet):
        source = packet if type(packet) is SharedSourceIdentity else getattr(packet, "source", None)
        if type(source) is not SharedSourceIdentity or source != self._boundary():
            raise ValueError("shared parent/row/episode Reset epoch or clock changed")
        return packet
