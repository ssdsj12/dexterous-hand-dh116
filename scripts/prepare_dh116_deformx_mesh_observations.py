"""Prepare a separately source-locked observation-only DeformX delta; never build/run it.

Generated DeformX/YarnBall delta: GPL-3.0-only. Old backend/source are read-only.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path

VERSION="dh116-deformx-mesh-observations-preparation-v1"
OLD_ROOT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/parameter-getters-build-e96wtu")
OLD_SOURCE=OLD_ROOT/"source"
OLD_BACKEND=OLD_ROOT/"build/python/ropesim/_ropesim.cpython-310-x86_64-linux-gnu.so"
OLD_BACKEND_SHA="2fea8036dc6630bcbc2a2b13a5699764f43cfdffef6036432ab812d4aaa96dec"
TREE_SHA="3cc86d9864d0ffd05da64a2097378a5c8c2b779b2bf7003659ec1fa8e4306472"

PATCHES = [
    ('third_party/yarnball/KittenEngine/YarnBall/YarnBall.h',
     '''			vec3 point, normal, forceOnRod, torqueOnMesh;''',
     '''			vec3 point, normal, forceOnRod, torqueOnMesh;
			// Diagnostic fields from the SAME end-substep contact evaluation.
			vec3 meshPosition;
			vec4 meshQuaternion;
			float gap, normalForce;'''),
    ('third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh',
     '''const float k, const float h, const float muS, const float muK, vec3& f, hess3& H) {''',
     '''const float k, const float h, const float muS, const float muK, vec3& f, hess3& H,
		float* normalForceOut = nullptr) {'''),
    ('third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh',
     '''		const float N = -k * gap;''',
     '''		const float N = -k * gap;
		if (normalForceOut) *normalForceOut = N;'''),
    ('third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh',
     '''				vec3 fm(0);
				surfaceContact(data, n, gap, dxRel, k, h, data->sdf[m].muS, data->sdf[m].muK, fm, H);''',
     '''				vec3 fm(0);
				float nativeNormalForce = 0.f;
				surfaceContact(data, n, gap, dxRel, k, h, data->sdf[m].muS, data->sdf[m].muK, fm, H,
					data->d_sdfPairs ? &nativeNormalForce : nullptr);'''),
    ('third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh',
     '''tid % data->sdfEnvVerts, m, P.pos + r, n, fm, tq};''',
     '''tid % data->sdfEnvVerts, m, P.pos + r, n, fm, tq,
						P.pos, P.q, gap, nativeNormalForce};'''),
    ('csrc/sdf.h',
     '''    PairReadback contact_pairs(cudaStream_t stream) const;''',
     '''    PairReadback contact_pairs(cudaStream_t stream) const;

    // Owned snapshot only: synchronize the solver stream and copy device state.
    // Host flags are identified separately; no target staging or commit occurs.
    struct MeshState {
        int num_envs, mesh_count, pair_capacity;
        bool pending_set, moving, pair_ready;
        std::vector<Pose7> current, target;
        std::vector<uint8_t> teleport;
        std::vector<Pose> solver;
    };
    MeshState mesh_state_snapshot(cudaStream_t stream) const;
'''),
    ('csrc/sdf.cu',
     '''void SdfStore::wrenches(float* out) const {
    if (!forcesOn || !d_wrench) throw std::invalid_argument("enable_contact_forces(True) first");
    std::vector<glm::vec3> w(2 * (size_t)N * MAX);
    check(cudaMemcpy(w.data(), d_wrench, sizeof(glm::vec3) * w.size(), cudaMemcpyDeviceToHost), "download(mesh wrenches)");
    for (int env = 0; env < N; env++)
        for (int m = 0; m < count(); m++) {
            const glm::vec3 F = w[((size_t)env * MAX + m) * 2], T = w[((size_t)env * MAX + m) * 2 + 1];
            float* o = out + ((size_t)env * count() + m) * 6;
            o[0] = F.x; o[1] = F.y; o[2] = F.z; o[3] = T.x; o[4] = T.y; o[5] = T.z;
        }
}
''',
     '''void SdfStore::wrenches(float* out) const {
    if (!forcesOn || !d_wrench) throw std::invalid_argument("enable_contact_forces(True) first");
    std::vector<glm::vec3> w(2 * (size_t)N * MAX);
    check(cudaMemcpy(w.data(), d_wrench, sizeof(glm::vec3) * w.size(), cudaMemcpyDeviceToHost), "download(mesh wrenches)");
    for (int env = 0; env < N; env++)
        for (int m = 0; m < count(); m++) {
            const glm::vec3 F = w[((size_t)env * MAX + m) * 2], T = w[((size_t)env * MAX + m) * 2 + 1];
            float* o = out + ((size_t)env * count() + m) * 6;
            o[0] = F.x; o[1] = F.y; o[2] = F.z; o[3] = T.x; o[4] = T.y; o[5] = T.z;
        }
}

SdfStore::MeshState SdfStore::mesh_state_snapshot(cudaStream_t stream) const {
    check(cudaStreamSynchronize(stream), "synchronize mesh state snapshot");
    MeshState result{};
    result.num_envs = N; result.mesh_count = count();
    result.pair_capacity = pairCapacity;
    result.pending_set = pendingSet; result.moving = moving; result.pair_ready = pairReady;
    const size_t total = (size_t)N * MAX;
    std::vector<Pose7> current(total), target(total);
    std::vector<uint8_t> teleport(total);
    std::vector<Pose> solver(total);
    check(cudaMemcpy(current.data(), d_cur, sizeof(Pose7) * total, cudaMemcpyDeviceToHost), "download mesh current");
    check(cudaMemcpy(target.data(), d_tgt, sizeof(Pose7) * total, cudaMemcpyDeviceToHost), "download mesh target");
    check(cudaMemcpy(teleport.data(), d_tele, total, cudaMemcpyDeviceToHost), "download mesh teleport flags");
    check(cudaMemcpy(solver.data(), d_poses, sizeof(Pose) * total, cudaMemcpyDeviceToHost), "download mesh solver pose");
    const size_t size = (size_t)N * count();
    result.current.reserve(size); result.target.reserve(size);
    result.teleport.reserve(size); result.solver.reserve(size);
    for (int e = 0; e < N; ++e)
        for (int m = 0; m < count(); ++m) {
            const size_t slot = (size_t)e * MAX + m;
            result.current.push_back(current[slot]); result.target.push_back(target[slot]);
            result.teleport.push_back(teleport[slot]); result.solver.push_back(solver[slot]);
        }
    return result;
}
'''),
    ('csrc/bindings.cpp',
     '''        .def("contact_pair_samples",''',
     '''        .def("contact_pair_samples_v2",
             [](RopeBatch& b) {
                 std::lock_guard<std::mutex> g(b.mutex());
                 const float before = b.raw()->meta.time;
                 auto capture = b.sdf.contact_pairs(b.raw()->solverStream());
                 const py::ssize_t size = capture.samples.size();
                 py::array_t<int> ids({size, py::ssize_t(3)});
                 py::array_t<float> vectors({size, py::ssize_t(12)});
                 py::array_t<float> positions({size, py::ssize_t(3)});
                 py::array_t<float> quaternions({size, py::ssize_t(4)});
                 py::array_t<float> gaps(size), normal_forces(size);
                 auto idx = ids.mutable_unchecked<2>();
                 auto v = vectors.mutable_unchecked<2>();
                 auto p = positions.mutable_unchecked<2>();
                 auto q = quaternions.mutable_unchecked<2>();
                 auto gap = gaps.mutable_unchecked<1>();
                 auto nf = normal_forces.mutable_unchecked<1>();
                 for (py::ssize_t i = 0; i < size; ++i) {
                     const auto& r = capture.samples[i];
                     idx(i, 0) = r.substep; idx(i, 1) = r.node; idx(i, 2) = r.mesh;
                     for (int j = 0; j < 3; ++j) {
                         v(i, j) = r.point[j]; v(i, 3+j) = r.normal[j];
                         v(i, 6+j) = r.forceOnRod[j]; v(i, 9+j) = r.torqueOnMesh[j];
                         p(i, j) = r.meshPosition[j];
                     }
                     for (int j = 0; j < 4; ++j) q(i, j) = r.meshQuaternion[j];
                     gap(i) = r.gap; nf(i) = r.normalForce;
                 }
                 py::dict out;
                 out["schema"] = "dh116.sdf_pair_samples.v2";
                 out["substeps"] = capture.substeps; out["mesh_count"] = capture.mesh_count;
                 out["capacity"] = capture.capacity; out["attempted"] = capture.attempted;
                 out["overflow"] = capture.attempted > capture.capacity;
                 out["indices"] = ids; out["vectors"] = vectors;
                 out["mesh_positions_world_m"] = positions;
                 out["mesh_quaternions_xyzw"] = quaternions;
                 out["surface_gap_m"] = gaps; out["normal_force_n"] = normal_forces;
                 out["sample_source"] = "cuda_surfaceReadout_same_end_substep_evaluation";
                 out["normal_force_semantics"] = "unaveraged_penalty_normal_force_before_friction";
                 out["owner_time_s_before"] = before; out["owner_time_s_after"] = b.raw()->meta.time;
                 out["owner_time_source"] = "native_host_owner_metadata_at_readback";
                 out["substep_timestamp_s"] = py::none();
                 out["last_error"] = b.raw()->lastErrorCode;
                 py::array_t<float> node_forces({py::ssize_t(b.verts_per_rope()), py::ssize_t(3)});
                 py::array_t<float> mesh_wrenches({py::ssize_t(capture.mesh_count), py::ssize_t(6)});
                 b.sdf.contact_forces(node_forces.mutable_data());
                 b.sdf.wrenches(mesh_wrenches.mutable_data());
                 out["node_forces_on_rod_world_n"] = node_forces;
                 out["mesh_wrenches_on_mesh_world"] = mesh_wrenches;
                 return out;
             }, "Owned native pose/gap/normal-force samples; substep indices have no measured timestamp.")
        .def("mesh_state_snapshot",
             [](RopeBatch& b) {
                 std::lock_guard<std::mutex> g(b.mutex());
                 const float before = b.raw()->meta.time;
                 auto state = b.sdf.mesh_state_snapshot(b.raw()->solverStream());
                 const py::ssize_t N = state.num_envs, M = state.mesh_count;
                 py::array_t<float> current_pos({N,M,py::ssize_t(3)}), current_q({N,M,py::ssize_t(4)});
                 py::array_t<float> target_pos({N,M,py::ssize_t(3)}), target_q({N,M,py::ssize_t(4)});
                 py::array_t<float> solver_pos({N,M,py::ssize_t(3)}), solver_q({N,M,py::ssize_t(4)});
                 py::array_t<uint8_t> teleport({N,M});
                 auto cp = current_pos.mutable_unchecked<3>(), cq = current_q.mutable_unchecked<3>();
                 auto tp = target_pos.mutable_unchecked<3>(), tq = target_q.mutable_unchecked<3>();
                 auto sp = solver_pos.mutable_unchecked<3>(), sq = solver_q.mutable_unchecked<3>();
                 auto tele = teleport.mutable_unchecked<2>();
                 for (py::ssize_t e=0; e<N; ++e)
                     for (py::ssize_t m=0; m<M; ++m) {
                         const size_t i = (size_t)e*M+m;
                         for (int j=0; j<3; ++j) {
                             cp(e,m,j)=state.current[i].pos[j]; tp(e,m,j)=state.target[i].pos[j];
                             sp(e,m,j)=state.solver[i].pos[j];
                         }
                         for (int j=0; j<4; ++j) {
                             cq(e,m,j)=state.current[i].q[j]; tq(e,m,j)=state.target[i].q[j];
                             sq(e,m,j)=state.solver[i].q[j];
                         }
                         tele(e,m)=state.teleport[i];
                     }
                 py::dict out;
                 out["schema"]="dh116.mesh_state_snapshot.v1";
                 out["num_envs"]=state.num_envs; out["mesh_count"]=state.mesh_count;
                 out["current_positions_world_m"]=current_pos; out["current_quaternions_xyzw"]=current_q;
                 out["target_positions_world_m"]=target_pos; out["target_quaternions_xyzw"]=target_q;
                 out["pending_teleport"]=teleport;
                 out["solver_positions_world_m"]=solver_pos; out["solver_quaternions_xyzw"]=solver_q;
                 out["device_state_source"]="cuda_device_copy";
                 out["pending_set"]=state.pending_set; out["moving"]=state.moving;
                 out["pair_capacity"]=state.pair_capacity; out["pair_ready"]=state.pair_ready;
                 out["flags_source"]="native_host_sdf_store";
                 out["owner_time_s_before"]=before; out["owner_time_s_after"]=b.raw()->meta.time;
                 out["owner_time_source"]="native_host_owner_metadata_at_readback";
                 out["last_error"]=b.raw()->lastErrorCode;
                 return out;
             }, "Owned current/target/teleport/solver-pose copies; no step, staging, or commit.")
        .def("contact_pair_samples",'''),
]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bytes(value):
    return value if isinstance(value,bytes) else value.encode()


def locked_sources(project):
    lock=json.loads((Path(project)/"docs/contracts/deformx_parameter_getter_source_lock_v1.json").read_text())
    inventory=lock["files"]
    digest=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if (digest!=TREE_SHA or len(inventory)!=72
            or lock["source_commit"]!="96b8514c7c55c724f1abba1fbf6ed6642b7db72b"):
        raise ValueError("locked source inventory differs")
    if sha256(OLD_BACKEND)!=OLD_BACKEND_SHA:
        raise ValueError("locked source backend bytes differ")
    result={}
    for name,expected in inventory.items():
        path=OLD_SOURCE/name
        if (path.is_symlink() or not path.resolve(strict=True).is_relative_to(OLD_SOURCE)
                or sha256(path)!=expected):
            raise ValueError("locked source bytes differ: "+name)
        payload=path.read_bytes()
        result[name]=payload if path.suffix==".png" else payload.decode()
    return result


def planned_sources(project):
    result=locked_sources(project)
    for name,before,after in PATCHES:
        if result[name].count(before)!=1:
            raise ValueError("exact single source anchor required: "+name)
        result[name]=result[name].replace(before,after)
    return result


def build_commands(destination):
    destination=Path(destination).resolve()
    tools=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/flexible-f1-20261112-883RS4")
    cmake=str(tools/"build-tools/cmake/data/bin/cmake")
    source=str(destination/"source"); build=str(destination/"build/cmake")
    return dict(
        configure=[cmake,"-S",source,"-B",build,"-G","Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_MAKE_PROGRAM="+str(tools/"build-tools/bin/ninja"),
            "-DCMAKE_CUDA_COMPILER="+str(tools/"sdk/bin/nvcc"),
            "-DCUDAToolkit_ROOT="+str(tools/"sdk"),
            "-DCMAKE_CUDA_ARCHITECTURES=89","-DPYBIND11_FINDPYTHON=ON",
            "-DPython_EXECUTABLE=/mnt/mydisk/xk/envs/dh116/bin/python",
            "-Dpybind11_DIR="+str(tools/"build-tools/pybind11/share/cmake/pybind11")],
        build=[cmake,"--build",build,"--parallel","2"],
        install=[cmake,"--install",build,"--prefix",str(destination/"build/python")],
        environment=dict(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",DH116_NUM_THREADS="2"),
        time_limits_s=dict(configure=60,build=180,install=30),max_attempts=1)


def prepare(project,destination):
    """Copy exact inputs and write reviewable commands only; no subprocess exists here."""
    destination=Path(destination).resolve()
    if (destination.exists() or destination.is_relative_to(OLD_ROOT)
            or OLD_ROOT.is_relative_to(destination)):
        raise ValueError("a new empty independent destination is required")
    original=locked_sources(project)
    prepared=planned_sources(project)  # All validation precedes mkdir.
    changed={name:hashlib.sha256(_bytes(value)).hexdigest()
             for name,value in prepared.items() if value!=original[name]}
    commands=build_commands(destination)
    for name,value in prepared.items():
        path=destination/"source"/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(_bytes(value))
    build=destination/"build";build.mkdir()
    # CMake installs only the new extension later; preserve the exact Python facade.
    for name,value in prepared.items():
        if name.startswith("ropesim/") and name.endswith(".py"):
            path=build/"python"/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(_bytes(value))
    delta="".join("".join(difflib.unified_diff(
        original[name].splitlines(keepends=True),prepared[name].splitlines(keepends=True),
        fromfile="a/"+name,tofile="b/"+name)) for name in sorted(changed))
    (build/"observation-only.patch").write_text(delta)
    result=dict(version=VERSION,original_source_tree_sha256=TREE_SHA,
        original_backend_path=str(OLD_BACKEND),original_backend_sha256=OLD_BACKEND_SHA,
        prepared_source_sha256={name:hashlib.sha256(_bytes(value)).hexdigest() for name,value in prepared.items()},
        changed_sources=changed,delta_sha256=hashlib.sha256(delta.encode()).hexdigest(),
        commands=commands,approval_required_before_build=True,
        compiled=False,solver_executed=False,isaac_executed=False,
        native_qualification=False,reset_qualified=False,contact_qualified=False,teacher_qualified=False)
    (build/"preparation.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    # Recheck original input provenance, including the old binary, after copying.
    locked_sources(project)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1],args.destination),indent=2))
