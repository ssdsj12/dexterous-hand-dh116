"""Source-locked, opt-in read-only getters in a NEW DeformX worktree.

Generated DeformX/YarnBall delta: GPL-3.0-only. Does not modify the installed backend.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import subprocess
from scripts.prepare_dh116_deformx_pair_readback import PATCHES as PAIR_PATCHES, UPSTREAM

POST_PAIR_BINDINGS_SHA = "89b3928faf49ec26e2d53614c0e30c5d6242892e54c8345c45f4d5702077d8d5"
ANCHOR = '        .def("inv_masses",\n             [](RopeBatch& b) {\n'
GETTER = "        .def(\"parameter_snapshot\",\n             [](RopeBatch& b) {\n                 std::lock_guard<std::mutex> g(b.mutex());\n                 const int N = b.num_envs(), V = b.verts_per_rope();\n                 const auto& m = b.raw()->meta;\n                 if (N < 1 || V < 2 || (size_t)m.numVerts != (size_t)N * V)\n                     throw py::value_error(\"invalid batched parameter layout\");\n                 auto checked = [](cudaError_t status) {\n                     if (status != cudaSuccess)\n                         throw std::runtime_error(cudaGetErrorString(status));\n                 };\n                 // Complete the solver's work, then take owned blocking D2H copies.\n                 // Sim::download() does not copy d_qRests; never use its host mirror.\n                 checked(cudaStreamSynchronize(b.raw()->solverStream()));\n                 std::vector<YarnBall::Vertex> vertices((size_t)N * V);\n                 std::vector<glm::vec4> rest_q((size_t)N * V);\n                 checked(cudaMemcpy(vertices.data(), m.d_verts,\n                     vertices.size() * sizeof(YarnBall::Vertex), cudaMemcpyDeviceToHost));\n                 checked(cudaMemcpy(rest_q.data(), m.d_qRests,\n                     rest_q.size() * sizeof(glm::vec4), cudaMemcpyDeviceToHost));\n                 py::array_t<float> rest({N, V - 1}), stretch({N, V - 1});\n                 py::array_t<float> qnorm({N, V - 1}), ea({N, V - 1}), ei({N, V - 1});\n                 py::array_t<float> invmass({N, V}), gravity(3);\n                 auto r = rest.mutable_unchecked<2>(), k = stretch.mutable_unchecked<2>();\n                 auto q = qnorm.mutable_unchecked<2>(), a = ea.mutable_unchecked<2>();\n                 auto eib = ei.mutable_unchecked<2>(), im = invmass.mutable_unchecked<2>();\n                 auto gv = gravity.mutable_unchecked<1>();\n                 for (int j = 0; j < 3; ++j) gv(j) = m.gravity[j];\n                 for (int e = 0; e < N; ++e) {\n                     for (int v = 0; v < V; ++v) {\n                         const size_t i = (size_t)e * V + v;\n                         im(e, v) = vertices[i].invMass;\n                         if (v == V - 1) continue; // trailing node is not a segment\n                         const float length = vertices[i].lRest;\n                         if (!std::isfinite(length) || !(length > 0))\n                             throw py::value_error(\"nonpositive/nonfinite device rest length\");\n                         r(e, v) = length;\n                         k(e, v) = vertices[i].kStretch;\n                         q(e, v) = glm::length(rest_q[i]);\n                         a(e, v) = k(e, v) / length;\n                         eib(e, v) = q(e, v) * length / 4.f;\n                     }\n                 }\n                 py::dict out;\n                 out[\"schema\"] = \"dh116.native_parameter_snapshot.v1\";\n                 out[\"num_envs\"] = N; out[\"nodes_per_env\"] = V;\n                 out[\"time_s\"] = m.time;\n                 out[\"metadata_source\"] = \"native_host_metadata\";\n                 out[\"segment_source\"] = \"cuda_device_copy\";\n                 out[\"hardware_measured\"] = false;\n                 out[\"contact_radius_m\"] = m.contactRadius;\n                 out[\"contact_mu_static\"] = m.contactMuS;\n                 out[\"contact_mu_kinetic\"] = m.contactMuK;\n                 out[\"contact_v_slip_m_s\"] = m.contactVSlip;\n                 out[\"contact_kappa\"] = m.contactKappa;\n                 out[\"gravity_world_m_s2\"] = gravity;\n                 out[\"rest_lengths_m\"] = rest;\n                 out[\"stored_k_stretch_n_m\"] = stretch;\n                 out[\"rest_quaternion_norm_n_m\"] = qnorm;\n                 out[\"stretch_ea_n\"] = ea;\n                 out[\"bend_ei_n_m2\"] = ei;\n                 out[\"inverse_masses_kg_inv\"] = invmass;\n                 return out;\n             }, \"Owned native parameter copies, no step/upload; EA/EI are decoded from device coefficients.\")\n"

def patch_bindings(text):
    if hashlib.sha256(text.encode()).hexdigest() != POST_PAIR_BINDINGS_SHA:
        raise ValueError("unexpected post-pair bindings SHA; refuse an unreviewed source")
    if text.count(ANCHOR) != 1 or '.def("parameter_snapshot"' in text:
        raise ValueError("expected a single unpatched RopeBatch anchor")
    return text.replace(ANCHOR, GETTER + ANCHOR)

def git_output(root, *args):
    return subprocess.check_output(["git", "-c", f"safe.directory={root}", "-C",
                                    str(root), *args], text=True).strip()

def planned_changes(root):
    """Validate every existing pair anchor and the new getter before any write."""
    root = Path(root).resolve(strict=True)
    pending = {}
    for name, before, after in PAIR_PATCHES:
        path = root / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("source path escapes independent worktree")
        text = pending.get(path)
        if text is None:
            text = path.read_text()
        if text.count(before) != 1:
            raise ValueError(f"source anchor mismatch: {name}")
        pending[path] = text.replace(before, after)
    binding = root / "csrc/bindings.cpp"
    pending[binding] = patch_bindings(pending[binding])
    return pending

def apply(root):
    root = Path(root).resolve(strict=True)
    if not (root / ".git").is_file():
        raise ValueError("a separate Git worktree is required")
    if git_output(root, "rev-parse", "HEAD") != UPSTREAM or git_output(root, "status", "--porcelain"):
        raise ValueError("a clean separate worktree at the pinned upstream commit is required")
    pending = planned_changes(root)
    for path, text in pending.items():
        path.write_text(text)
    return {str(path.relative_to(root)): hashlib.sha256(text.encode()).hexdigest()
            for path, text in pending.items()}

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(apply(args.source), indent=2))
