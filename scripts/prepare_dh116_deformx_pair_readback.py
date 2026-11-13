"""Apply the optional GPL-3.0 DeformX readback delta to a disposable worktree."""
# The DeformX/YarnBall changes produced by this script are licensed GPL-3.0-only.
import argparse
from pathlib import Path
import subprocess

UPSTREAM = "96b8514c7c55c724f1abba1fbf6ed6642b7db72b"

PATCHES = [('third_party/yarnball/KittenEngine/YarnBall/YarnBall.h',
  '\t\tvec3* d_sdfWrench;\n',
  '\t\tvec3* d_sdfWrench;\n'
  '\t\t// Optional diagnostic output only. One sample per solver substep/node/mesh.\n'
  '\t\tstruct SdfPairSample {\n'
  '\t\t\tint substep, node, mesh;\n'
  '\t\t\tvec3 point, normal, forceOnRod, torqueOnMesh;\n'
  '\t\t};\n'
  '\t\tint sdfPairCapacity = 0;\n'
  '\t\tint* d_sdfPairCount = nullptr;\n'
  '\t\tint* d_sdfPairSubstep = nullptr;\n'
  '\t\tSdfPairSample* d_sdfPairs = nullptr;\n'),
 ('third_party/yarnball/KittenEngine/YarnBall/sim/iteration.cu',
  '\t\tif (data->sdfMoving && tid < data->sdfEnvs',
  '\t\tif (tid == 0 && data->d_sdfPairSubstep) *data->d_sdfPairSubstep = sub;\n'
  '\t\tif (data->sdfMoving && tid < data->sdfEnvs'),
 ('third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh',
  '\t\t\tconst vec3 tq = cross(r, -fm);\n',
  '\t\t\tconst vec3 tq = cross(r, -fm);\n'
  '\t\t\tif (data->d_sdfPairs) {\n'
  '\t\t\t\tconst int i = atomicAdd(data->d_sdfPairCount, 1);\n'
  '\t\t\t\tif (i < data->sdfPairCapacity)\n'
  '\t\t\t\t\tdata->d_sdfPairs[i] = {*data->d_sdfPairSubstep,\n'
  '\t\t\t\t\t\ttid % data->sdfEnvVerts, m, P.pos + r, n, fm, tq};\n'
  '\t\t\t}\n'),
 ('csrc/sdf.h',
  '    void enable_forces(bool on);\n',
  '    void enable_forces(bool on);\n'
  '\n'
  '    using PairSample = YarnBall::MetaData::SdfPairSample;\n'
  '    void enable_pair_capture(int capacity);\n'
  '    void invalidate_pair_capture() { pairReady = false; }\n'
  '    struct PairReadback {\n'
  '        int attempted, capacity, substeps, mesh_count;\n'
  '        std::vector<PairSample> samples;\n'
  '    };\n'
  '    PairReadback contact_pairs(cudaStream_t stream) const;\n'),
 ('csrc/sdf.h',
  '    int lastSubsteps = 1;\n',
  '    int lastSubsteps = 1;\n'
  '    int pairCapacity = 0;\n'
  '    int pairMeshCount = 0;\n'
  '    bool pairReady = false;\n'
  '    int* d_pairCount = nullptr;\n'
  '    int* d_pairSubstep = nullptr;\n'
  '    PairSample* d_pairs = nullptr;\n'),
 ('csrc/sdf.cu',
  '    clear();\n    for (void* p',
  '    clear();\n'
  '    cudaFree(d_pairs);\n'
  '    cudaFree(d_pairSubstep);\n'
  '    cudaFree(d_pairCount);\n'
  '    for (void* p'),
 ('csrc/sdf.cu', 'void SdfStore::clear() {\n', 'void SdfStore::clear() {\n    invalidate_pair_capture();\n'),
 ('csrc/sdf.cu',
  'void SdfStore::enable_forces(bool on) { forcesOn = on; }\n',
  'void SdfStore::enable_forces(bool on) {\n'
  '    if (!on && pairCapacity) throw std::invalid_argument("pair capture requires contact forces");\n'
  '    forcesOn = on;\n'
  '    invalidate_pair_capture();\n'
  '}\n'
  '\n'
  'void SdfStore::enable_pair_capture(int capacity) {\n'
  '    if (N != 1 || !forcesOn || !count() || pairCapacity || capacity < 1 || capacity > 1048576)\n'
  '        throw std::invalid_argument("pair capture requires one rope, mesh forces, and a bounded one-time '
  'allocation");\n'
  '    try {\n'
  '        check(cudaMalloc(&d_pairCount, sizeof(int)), "cudaMalloc(pair count)");\n'
  '        check(cudaMalloc(&d_pairSubstep, sizeof(int)), "cudaMalloc(pair substep)");\n'
  '        check(cudaMalloc(&d_pairs, sizeof(PairSample) * capacity), "cudaMalloc(pair samples)");\n'
  '    } catch (...) {\n'
  '        cudaFree(d_pairs); d_pairs = nullptr;\n'
  '        cudaFree(d_pairSubstep); d_pairSubstep = nullptr;\n'
  '        cudaFree(d_pairCount); d_pairCount = nullptr;\n'
  '        throw;\n'
  '    }\n'
  '    pairCapacity = capacity;\n'
  '    invalidate_pair_capture();\n'
  '}\n'
  '\n'
  'SdfStore::PairReadback SdfStore::contact_pairs(cudaStream_t stream) const {\n'
  '    if (!pairReady) throw std::invalid_argument("no completed pair capture after most recent reset or mesh '
  'change");\n'
  '    check(cudaStreamSynchronize(stream), "synchronize pair capture");\n'
  '    int attempted = 0;\n'
  '    check(cudaMemcpy(&attempted, d_pairCount, sizeof(int), cudaMemcpyDeviceToHost), "download pair count");\n'
  '    if (attempted < 0) throw std::runtime_error("pair count overflow");\n'
  '    PairReadback result{attempted, pairCapacity, lastSubsteps, pairMeshCount, {}};\n'
  '    result.samples.resize(std::min(attempted, pairCapacity));\n'
  '    if (!result.samples.empty())\n'
  '        check(cudaMemcpy(result.samples.data(), d_pairs, sizeof(PairSample) * result.samples.size(),\n'
  '                         cudaMemcpyDeviceToHost), "download pair samples");\n'
  '    return result;\n'
  '}\n'),
 ('csrc/sdf.cu',
  '    const int idx = count() - 1;\n',
  '    invalidate_pair_capture();\n    const int idx = count() - 1;\n'),
 ('csrc/sdf.cu',
  '    m.d_sdfWrench = d_wrench;\n',
  '    m.d_sdfWrench = d_wrench;\n'
  '    m.sdfPairCapacity = pairCapacity;\n'
  '    m.d_sdfPairCount = d_pairCount;\n'
  '    m.d_sdfPairSubstep = d_pairSubstep;\n'
  '    m.d_sdfPairs = d_pairs;\n'),
 ('csrc/sdf.cu',
  'void SdfStore::begin_step(YarnBall::MetaData& m, cudaStream_t s) {\n',
  'void SdfStore::begin_step(YarnBall::MetaData& m, cudaStream_t s) {\n'
  '    invalidate_pair_capture();\n'
  '    if (pairCapacity) {\n'
  '        if (m.numPlanes || !count()) throw std::invalid_argument("capture supports mesh-only contact");\n'
  '        check(cudaMemsetAsync(d_pairCount, 0, sizeof(int), s), "zero(pair count)");\n'
  '    }\n'),
 ('csrc/sdf.cu',
  '        check(cudaGetLastError(), "average(contact forces)");\n    }\n',
  '        check(cudaGetLastError(), "average(contact forces)");\n'
  '    }\n'
  '    pairMeshCount = count();\n'
  '    pairReady = pairCapacity != 0;\n'),
 ('csrc/batch.cu',
  'void RopeBatch::reset_envs(const int* ids_host, int count) {\n'
  '    if (count <= 0) return;\n'
  '    std::lock_guard<std::mutex> g(mtx_);\n',
  'void RopeBatch::reset_envs(const int* ids_host, int count) {\n'
  '    if (count <= 0) return;\n'
  '    std::lock_guard<std::mutex> g(mtx_);\n'
  '    sdf.invalidate_pair_capture();\n'),
 ('csrc/bindings.cpp',
  '        .def("contact_forces_view",   // device [N, V, 3]',
  '        .def("enable_contact_pair_capture",\n'
  '             [](RopeBatch& b, int capacity) {\n'
  '                 std::lock_guard<std::mutex> g(b.mutex());\n'
  '                 if (b.stepped || b.raw()->meta.numPlanes)\n'
  '                     throw py::value_error("pair capture needs mesh-only configuration before the first '
  'step");\n'
  '                 b.sdf.enable_pair_capture(capacity);\n'
  '                 b.sdf.write_meta(b.raw()->meta, b.verts_per_rope());\n'
  '             }, py::arg("capacity"))\n'
  '        .def("contact_pair_samples",\n'
  '             [](RopeBatch& b) {\n'
  '                 std::lock_guard<std::mutex> g(b.mutex());\n'
  '                 auto capture = b.sdf.contact_pairs(b.raw()->solverStream());\n'
  '                 const py::ssize_t size = capture.samples.size();\n'
  '                 py::array_t<int> ids({size, py::ssize_t(3)});\n'
  '                 py::array_t<float> values({size, py::ssize_t(12)});\n'
  '                 auto idx = ids.mutable_unchecked<2>();\n'
  '                 auto val = values.mutable_unchecked<2>();\n'
  '                 for (py::ssize_t i = 0; i < size; ++i) {\n'
  '                     const auto& r = capture.samples[i];\n'
  '                     idx(i, 0) = r.substep; idx(i, 1) = r.node; idx(i, 2) = r.mesh;\n'
  '                     for (int j = 0; j < 3; ++j) {\n'
  '                         val(i, j) = r.point[j]; val(i, 3 + j) = r.normal[j];\n'
  '                         val(i, 6 + j) = r.forceOnRod[j]; val(i, 9 + j) = r.torqueOnMesh[j];\n'
  '                     }\n'
  '                 }\n'
  '                 py::dict result;\n'
  '                 result["substeps"] = capture.substeps;\n'
  '                 result["mesh_count"] = capture.mesh_count;\n'
  '                 result["capacity"] = capture.capacity;\n'
  '                 result["attempted"] = capture.attempted;\n'
  '                 result["overflow"] = capture.attempted > capture.capacity;\n'
  '                 result["indices"] = ids;\n'
  '                 result["vectors"] = values;\n'
  '                 return result;\n'
  '             }, "Owned host copies of raw mesh contact at each substep. Overflow means incomplete.")\n'
  '        .def("contact_forces_view",   // device [N, V, 3]'),
 ('csrc/sdf.h',
  '    void invalidate_pair_capture() { pairReady = false; }\n',
  '    void invalidate_pair_capture() { pairReady = false; }\n'
  '    bool pair_capture_enabled() const { return pairCapacity != 0; }\n'
  '    void require_bounded_pair_step(int vertices, int substeps) const;\n'),
 ('csrc/sdf.cu',
  'SdfStore::PairReadback SdfStore::contact_pairs(cudaStream_t stream) const {',
  'void SdfStore::require_bounded_pair_step(int vertices, int substeps) const {\n'
  '    if (pairCapacity && (substeps < 1 || substeps > 64 ||\n'
  '            (int64_t)vertices * count() * substeps > 2147483647))\n'
  '        throw std::invalid_argument("pair counter would exceed the bounded capture range");\n'
  '}\n'
  '\n'
  'SdfStore::PairReadback SdfStore::contact_pairs(cudaStream_t stream) const {'),
 ('csrc/batch.cu',
  'void RopeBatch::step_free(float dt, int substeps) {\n    std::lock_guard<std::mutex> g(mtx_);\n',
  'void RopeBatch::step_free(float dt, int substeps) {\n'
  '    std::lock_guard<std::mutex> g(mtx_);\n'
  '    sdf.require_bounded_pair_step(V, substeps);\n'),
 ('csrc/bindings.cpp',
  '                 if (!b.rootsPinned)\n'
  '                     throw py::value_error("step() drives every rope\'s root; ropes built with a free root ',
  '                 if (b.sdf.pair_capture_enabled()) throw py::value_error("pair capture requires step_free");\n'
  '                 if (!b.rootsPinned)\n'
  '                     throw py::value_error("step() drives every rope\'s root; ropes built with a free root '),
 ('csrc/bindings.cpp',
  '                 result["vectors"] = values;\n                 return result;',
  '                 result["vectors"] = values;\n'
  '                 py::array_t<float> node_forces({py::ssize_t(b.verts_per_rope()), py::ssize_t(3)});\n'
  '                 py::array_t<float> mesh_wrenches({py::ssize_t(capture.mesh_count), py::ssize_t(6)});\n'
  '                 b.sdf.contact_forces(node_forces.mutable_data());\n'
  '                 b.sdf.wrenches(mesh_wrenches.mutable_data());\n'
  '                 result["node_forces_on_rod_world_n"] = node_forces;\n'
  '                 result["mesh_wrenches_on_mesh_world"] = mesh_wrenches;\n'
  '                 return result;')]

def apply(root):
    """Check the immutable source before writing a separate diagnostic checkout."""
    root = Path(root).resolve(strict=True)
    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    if (subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != UPSTREAM
            or subprocess.check_output([*git, "status", "--porcelain"], text=True).strip()
            or not (root / ".git").is_file()):
        raise ValueError("a clean separate Git worktree at the locked DeformX commit is required")
    pending = {}

    def edit(path, old, new):
        file = root / path
        text = pending.get(file)
        if text is None:
            text = file.read_text()
        if text.count(old) != 1:
            raise ValueError(f"{path}: expected one pinned source anchor, got {text.count(old)}")
        pending[file] = text.replace(old, new)

    for path, old, new in PATCHES:
        edit(path, old, new)

    for file, text in pending.items():
        file.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    apply(args.source)
