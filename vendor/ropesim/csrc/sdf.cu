// SdfStore: device-side pose state machine for rod-mesh contact (see sdf.h).
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <string>

#include "sdf.h"
#include "YarnBall/sim/contact.cuh"

namespace {

using Pose7 = SdfStore::Pose7;
using Box = SdfStore::Box;
using Pose = SdfStore::Pose;
using Motion = SdfStore::Motion;
constexpr int MAX = SdfStore::MAX;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Rotation vector (world frame, shortest arc) taking unit q0 to unit q1:
// q1 = exp(rot) * q0, i.e. rot = log(q1 * conj(q0)).
__device__ glm::vec3 rotvec(const glm::vec4& q0, const glm::vec4& q1) {
    const glm::vec3 a(q1.x, q1.y, q1.z), b(-q0.x, -q0.y, -q0.z);
    glm::vec3 v = q1.w * b + q0.w * a + glm::cross(a, b);
    float w = q1.w * q0.w - glm::dot(a, b);
    if (w < 0.f) { v = -v; w = -w; }
    const float s = glm::length(v);
    if (!(s > 1e-9f)) return glm::vec3(0.f);
    return v * (2.f * atan2f(s, w) / s);
}

// Targets for mesh `index` (rows = 1 or N: one thread per env) or, with index < 0,
// for every mesh from a dense [N, M] block (one thread per (env, mesh)).
__global__ void scatterTargetsK(Pose7* tgt, uint8_t* tele, int index, const float* pos, const float* q, int rows, int M,
                                const uint8_t* teleFlags, int teleAll, int N) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int per = index < 0 ? M : 1;
    if (i >= N * per) return;
    const int e = i / per, m = index < 0 ? i % per : index;
    const int r = index < 0 ? i : (rows == 1 ? 0 : e);
    glm::vec4 qq(q[4 * r], q[4 * r + 1], q[4 * r + 2], q[4 * r + 3]);
    const float l = glm::length(qq);
    qq = l > 0.f ? qq / l : glm::vec4(0, 0, 0, 1);
    const size_t slot = (size_t)e * MAX + m;
    tgt[slot].pos = glm::vec3(pos[3 * r], pos[3 * r + 1], pos[3 * r + 2]);
    tgt[slot].q = qq;
    tele[slot] = teleAll ? 1 : (teleFlags ? teleFlags[e] : 0);
}

// One thread per (env, mesh): this frame's motion from the current pose to the
// target; a teleported slot becomes the target at once (no motion).
__global__ void stageMotionK(Pose7* cur, const Pose7* tgt, uint8_t* tele, Motion* motion, int total) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= total) return;
    if (tele[i]) { cur[i] = tgt[i]; tele[i] = 0; }
    motion[i].p0 = cur[i].pos;
    motion[i].q0 = cur[i].q;
    motion[i].dp = tgt[i].pos - cur[i].pos;
    motion[i].rot = rotvec(cur[i].q, tgt[i].q);
}

// The target of slot i is now its current pose; the static pose (velocity 0)
// and its world AABB go to what the kernels read.
__device__ void commitSlot(Pose7* cur, const Pose7* tgt, uint8_t* tele, Pose* poses, const Box* box, size_t i, int m) {
    cur[i] = tgt[i];
    tele[i] = 0;
    Pose P;
    P.pos = cur[i].pos; P.q = cur[i].q; P.vel = glm::vec3(0.f); P.omega = glm::vec3(0.f);
    YarnBall::sdfWorldBox(box[m].lo, box[m].hi, P.pos, Kit::Rotor(P.q), P.lo, P.hi);
    poses[i] = P;
}
// One thread per (env, mesh): after a moving step.
__global__ void commitK(Pose7* cur, const Pose7* tgt, uint8_t* tele, Pose* poses, const Box* box, int numSdf, int total) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= total) return;
    const int m = i % MAX;
    if (m < numSdf) commitSlot(cur, tgt, tele, poses, box, i, m);
}
// One thread per env: a newly added mesh only (other meshes' pending targets stay pending).
__global__ void commitMeshK(Pose7* cur, const Pose7* tgt, uint8_t* tele, Pose* poses, const Box* box, int index, int N) {
    const int e = threadIdx.x + blockIdx.x * blockDim.x;
    if (e >= N) return;
    commitSlot(cur, tgt, tele, poses, box, (size_t)e * MAX + index, index);
}

// Explicit zero-time reset, touching only one selected environment.
__global__ void resetEnvMeshesK(Pose7* cur, Pose7* tgt, uint8_t* tele, Motion* motion,
                                Pose* poses, const Box* box, const float* positions,
                                const float* quaternions, int env, int source, int meshes) {
    const int mesh = threadIdx.x + blockIdx.x * blockDim.x;
    if (mesh >= meshes) return;
    const size_t input = (size_t)source * meshes + mesh;
    const size_t slot = (size_t)env * MAX + mesh;
    tgt[slot].pos = glm::vec3(positions[3*input], positions[3*input+1], positions[3*input+2]);
    tgt[slot].q = glm::normalize(glm::vec4(quaternions[4*input], quaternions[4*input+1],
                                         quaternions[4*input+2], quaternions[4*input+3]));
    commitSlot(cur, tgt, tele, poses, box, slot, mesh);
    motion[slot].p0 = cur[slot].pos;
    motion[slot].q0 = cur[slot].q;
    motion[slot].dp = glm::vec3(0.f);
    motion[slot].rot = glm::vec3(0.f);
}

__global__ void clearEnvContactK(glm::vec3* force, glm::vec3* wrench, int* candidates,
                                 int* counts, int* pairCounts, int env, int vertices) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < vertices && force) {
        force[(size_t)env*vertices+i] = glm::vec3(0.f);
        counts[(size_t)env*vertices+i] = -1;
        for (int j=0; j<SdfStore::CAND; ++j)
            candidates[((size_t)env*vertices+i)*SdfStore::CAND+j] = -1;
    }
    if (i < MAX && wrench) {
        wrench[((size_t)env*MAX+i)*2] = glm::vec3(0.f);
        wrench[((size_t)env*MAX+i)*2+1] = glm::vec3(0.f);
    }
    if (i == 0 && pairCounts) pairCounts[env] = 0;
}

__global__ void scaleK(float* a, int n, float s) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] *= s;
}

inline int blocks(int n) { return (n + 255) / 256; }

}  // namespace

SdfStore::SdfStore(int num_envs) : N(num_envs) {
    const size_t total = (size_t)N * MAX;
    check(cudaMalloc(&d_cur, sizeof(Pose7) * total), "cudaMalloc(sdf cur)");
    check(cudaMalloc(&d_tgt, sizeof(Pose7) * total), "cudaMalloc(sdf tgt)");
    check(cudaMalloc(&d_tele, total), "cudaMalloc(sdf tele)");
    check(cudaMalloc(&d_motion, sizeof(Motion) * total), "cudaMalloc(sdf motion)");
    check(cudaMalloc(&d_poses, sizeof(Pose) * total), "cudaMalloc(sdf poses)");
    check(cudaMalloc(&d_box, sizeof(Box) * MAX), "cudaMalloc(sdf boxes)");
    check(cudaMalloc(&d_stage, sizeof(float) * 7 * (size_t)N * MAX), "cudaMalloc(sdf stage)");
    check(cudaMalloc(&d_stageTele, N), "cudaMalloc(sdf stage flags)");
    std::vector<Pose7> ident(total, Pose7{glm::vec3(0.f), glm::vec4(0, 0, 0, 1)});
    check(cudaMemcpy(d_cur, ident.data(), sizeof(Pose7) * total, cudaMemcpyHostToDevice), "upload(sdf cur)");
    check(cudaMemcpy(d_tgt, ident.data(), sizeof(Pose7) * total, cudaMemcpyHostToDevice), "upload(sdf tgt)");
    check(cudaMemset(d_tele, 0, total), "memset(sdf tele)");
    check(cudaMemset(d_motion, 0, sizeof(Motion) * total), "memset(sdf motion)");
    check(cudaMemset(d_poses, 0, sizeof(Pose) * total), "memset(sdf poses)");
    check(cudaMemset(d_box, 0, sizeof(Box) * MAX), "memset(sdf boxes)");
}

SdfStore::~SdfStore() {
    clear();
    cudaFree(d_pairs);
    cudaFree(d_pairSubstep);
    cudaFree(d_pairCount);
    for (void* p : {(void*)d_cur, (void*)d_tgt, (void*)d_tele, (void*)d_motion, (void*)d_poses, (void*)d_box,
                    (void*)d_stage, (void*)d_stageTele, (void*)d_cand, (void*)d_numCand, (void*)d_force, (void*)d_wrench})
        cudaFree(p);
}

void SdfStore::alloc_verts(int nv) {
    if (nv == numVerts) return;
    cudaFree(d_cand); cudaFree(d_numCand); cudaFree(d_force); cudaFree(d_wrench);
    d_cand = nullptr; d_numCand = nullptr; d_force = nullptr; d_wrench = nullptr;
    numVerts = nv;
    check(cudaMalloc(&d_cand, sizeof(int) * (size_t)nv * CAND), "cudaMalloc(sdf candidates)");
    check(cudaMalloc(&d_numCand, sizeof(int) * nv), "cudaMalloc(sdf candidate counts)");
    check(cudaMemset(d_numCand, 0xFF, sizeof(int) * nv), "memset(sdf candidate counts)");   // -1: every mesh
    check(cudaMalloc(&d_force, sizeof(glm::vec3) * nv), "cudaMalloc(contact forces)");
    check(cudaMalloc(&d_wrench, sizeof(glm::vec3) * 2 * (size_t)N * MAX), "cudaMalloc(mesh wrenches)");
    check(cudaMemset(d_force, 0, sizeof(glm::vec3) * nv), "memset(contact forces)");
    check(cudaMemset(d_wrench, 0, sizeof(glm::vec3) * 2 * (size_t)N * MAX), "memset(mesh wrenches)");
}

int SdfStore::add(const float* values, int nx, int ny, int nz, glm::vec3 origin, float spacing, float muS, float muK,
                  const float* pos, const float* q, int rows, cudaStream_t s) {
    if (count() >= MAX) throw std::invalid_argument("at most " + std::to_string(MAX) + " meshes per sim");
    const size_t n = (size_t)nx * ny * nz;
    float* d = nullptr;
    check(cudaMalloc(&d, sizeof(float) * n), "cudaMalloc(sdf grid)");
    const cudaError_t e = cudaMemcpy(d, values, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        cudaFree(d);
        check(e, "upload(sdf grid)");
    }
    grids.push_back(Grid{d, nx, ny, nz, origin, spacing, 1.f / spacing, muS, muK});
    boxes.push_back(Box{origin, origin + glm::vec3(float(nx - 1), float(ny - 1), float(nz - 1)) * spacing});
    invalidate_pair_capture();
    const int idx = count() - 1;
    check(cudaMemcpyAsync(d_box, boxes.data(), sizeof(Box) * boxes.size(), cudaMemcpyHostToDevice, s), "upload(sdf boxes)");
    const bool pending = pendingSet;                           // other meshes' targets stay pending (review 3)
    set_targets_host(idx, pos, q, rows, nullptr, true, s);     // the new mesh starts at its initial pose
    commitMeshK<<<blocks(N), 256, 0, s>>>(d_cur, d_tgt, d_tele, d_poses, d_box, idx, N);
    check(cudaStreamSynchronize(s), "add_mesh");
    pendingSet = pending;
    return idx;
}

void SdfStore::set_friction(int index, float muS, float muK) {
    if (index < 0 || index >= count()) throw std::invalid_argument("mesh index out of range");
    grids[index].muS = muS; grids[index].muK = muK;
}

void SdfStore::set_targets(int index, const float* d_pos, const float* d_q, int rows, const uint8_t* tele, bool teleAll,
                           cudaStream_t s) {
    if (index >= count() || (index < 0 && count() == 0)) throw std::invalid_argument("mesh index out of range");
    const int threads = index < 0 ? N * count() : N;
    scatterTargetsK<<<blocks(threads), 256, 0, s>>>(d_tgt, d_tele, index, d_pos, d_q, rows, count(), tele, teleAll ? 1 : 0, N);
    check(cudaGetLastError(), "set_mesh_poses");
    pendingSet = true;
}

const uint8_t* SdfStore::stage_flags(const uint8_t* tele, bool teleAll, cudaStream_t s) {
    if (!tele || teleAll) return nullptr;
    check(cudaMemcpyAsync(d_stageTele, tele, N, cudaMemcpyHostToDevice, s), "upload(sdf teleport flags)");
    return d_stageTele;
}

void SdfStore::set_targets_host(int index, const float* pos, const float* q, int rows, const uint8_t* tele, bool teleAll,
                                cudaStream_t s) {
    if (index >= count() || (index < 0 && count() == 0)) throw std::invalid_argument("mesh index out of range");
    const int n = index < 0 ? N * count() : rows;
    float* dq = d_stage + 3 * (size_t)N * MAX;
    // pageable sources: the driver stages them before returning, so the caller's arrays may die right after
    check(cudaMemcpyAsync(d_stage, pos, sizeof(float) * 3 * n, cudaMemcpyHostToDevice, s), "upload(sdf target pos)");
    check(cudaMemcpyAsync(dq, q, sizeof(float) * 4 * n, cudaMemcpyHostToDevice, s), "upload(sdf target quat)");
    set_targets(index, d_stage, dq, rows, stage_flags(tele, teleAll, s), teleAll, s);
}

void SdfStore::reset_env_mesh_poses(const std::vector<int>& envs, const float* positions,
                                     const float* quaternions, cudaStream_t stream) {
    if (pendingSet || moving || !count() || envs.empty())
        throw std::invalid_argument("mesh Reset requires a completed mesh boundary");
    std::vector<bool> seen(N, false);
    for (int env : envs) {
        if (env < 0 || env >= N || seen[env])
            throw std::invalid_argument("distinct in-range mesh Reset environments required");
        seen[env] = true;
    }
    const size_t rows = envs.size() * count();
    float* stagedQ = d_stage + 3 * (size_t)N * MAX;
    check(cudaMemcpyAsync(d_stage, positions, sizeof(float)*3*rows, cudaMemcpyHostToDevice, stream),
          "upload selected Reset positions");
    check(cudaMemcpyAsync(stagedQ, quaternions, sizeof(float)*4*rows, cudaMemcpyHostToDevice, stream),
          "upload selected Reset quaternions");
    const int vertices = numVerts / N;
    for (size_t source=0; source<envs.size(); ++source) {
        resetEnvMeshesK<<<blocks(count()), 256, 0, stream>>>(
            d_cur, d_tgt, d_tele, d_motion, d_poses, d_box, d_stage, stagedQ,
            envs[source], (int)source, count());
        clearEnvContactK<<<blocks(vertices > MAX ? vertices : MAX), 256, 0, stream>>>(
            d_force, d_wrench, d_cand, d_numCand, d_pairCount, envs[source], vertices);
    }
    check(cudaGetLastError(), "selected mesh Reset kernels");
    check(cudaStreamSynchronize(stream), "complete selected mesh Reset");
    invalidate_pair_capture();
}

void SdfStore::clear() {
    invalidate_pair_capture();
    for (Grid& g : grids) cudaFree(const_cast<float*>(g.v));
    grids.clear();
    boxes.clear();
    pendingSet = false;
    moving = false;
}

void SdfStore::enable_forces(bool on) {
    if (!on && pairCapacity) throw std::invalid_argument("pair capture requires contact forces");
    forcesOn = on;
    invalidate_pair_capture();
}

void SdfStore::enable_pair_capture(int capacity, bool batched) {
    if ((!batched && N != 1) || (batched && (N < 2 || N > 512 || (int64_t)N * capacity > 8388608)) || !forcesOn || !count() || pairCapacity || capacity < 1 || capacity > 1048576)
        throw std::invalid_argument("pair capture requires one rope, mesh forces, and a bounded one-time allocation");
    try {
        check(cudaMalloc(&d_pairCount, sizeof(int) * N), "cudaMalloc(pair count)");
        check(cudaMalloc(&d_pairSubstep, sizeof(int)), "cudaMalloc(pair substep)");
        check(cudaMalloc(&d_pairs, sizeof(PairSample) * (size_t)capacity * N), "cudaMalloc(pair samples)");
    } catch (...) {
        cudaFree(d_pairs); d_pairs = nullptr;
        cudaFree(d_pairSubstep); d_pairSubstep = nullptr;
        cudaFree(d_pairCount); d_pairCount = nullptr;
        throw;
    }
    pairCapacity = capacity;
    invalidate_pair_capture();
}

void SdfStore::require_bounded_pair_step(int vertices, int substeps) const {
    if (pairCapacity && (substeps < 1 || substeps > 64 ||
            (int64_t)vertices * count() * substeps > 2147483647))
        throw std::invalid_argument("pair counter would exceed the bounded capture range");
}

SdfStore::PairReadback SdfStore::contact_pairs(cudaStream_t stream, int env) const {
    if (env < 0 || env >= N) throw std::out_of_range("pair capture environment");
    if (!pairReady) throw std::invalid_argument("no completed pair capture after most recent reset or mesh change");
    check(cudaStreamSynchronize(stream), "synchronize pair capture");
    int attempted = 0;
    check(cudaMemcpy(&attempted, d_pairCount + env, sizeof(int), cudaMemcpyDeviceToHost), "download pair count");
    if (attempted < 0) throw std::runtime_error("pair count overflow");
    PairReadback result{attempted, pairCapacity, lastSubsteps, pairMeshCount, {}};
    result.samples.resize(std::min(attempted, pairCapacity));
    if (!result.samples.empty())
        check(cudaMemcpy(result.samples.data(), d_pairs + (size_t)env * pairCapacity, sizeof(PairSample) * result.samples.size(),
                         cudaMemcpyDeviceToHost), "download pair samples");
    return result;
}

void SdfStore::write_meta(YarnBall::MetaData& m, int env_verts) {
    alloc_verts(N * (env_verts > 0 ? env_verts : 1));
    m.numSdf = count();
    for (int i = 0; i < MAX; i++)
        m.sdf[i] = i < count() ? grids[i] : Grid{nullptr, 0, 0, 0, glm::vec3(0.f), 0.f, 0.f, 0.f, 0.f};
    m.d_sdfPoses = d_poses;
    m.d_sdfMotion = d_motion;
    m.sdfEnvs = N;
    m.sdfEnvVerts = env_verts > 0 ? env_verts : 1;
    m.d_sdfCand = d_cand;
    m.d_sdfNumCand = d_numCand;
    m.sdfForces = forcesOn ? 1 : 0;
    m.d_contactForce = d_force;
    m.d_sdfWrench = d_wrench;
    m.sdfPairCapacity = pairCapacity;
    m.d_sdfPairCount = d_pairCount;
    m.d_sdfPairSubstep = d_pairSubstep;
    m.d_sdfPairs = d_pairs;
}

void SdfStore::begin_step(YarnBall::MetaData& m, cudaStream_t s) {
    invalidate_pair_capture();
    if (pairCapacity) {
        if (m.numPlanes || !count()) throw std::invalid_argument("capture supports mesh-only contact");
        check(cudaMemsetAsync(d_pairCount, 0, sizeof(int) * N, s), "zero(pair count)");
    }
    if (numVerts == 0) return;   // write_meta never called: no meshes, nothing published
    if (pendingSet) {
        stageMotionK<<<blocks(N * MAX), 256, 0, s>>>(d_cur, d_tgt, d_tele, d_motion, N * MAX);
        m.sdfMoving = 1;
        moving = true;
    } else {
        m.sdfMoving = 0;
    }
    // the first substep of every frame visits every mesh (poses may have jumped)
    if (count() > 0) check(cudaMemsetAsync(d_numCand, 0xFF, sizeof(int) * numVerts, s), "reset(sdf candidates)");
    if (forcesOn) {
        check(cudaMemsetAsync(d_force, 0, sizeof(glm::vec3) * numVerts, s), "zero(contact forces)");
        check(cudaMemsetAsync(d_wrench, 0, sizeof(glm::vec3) * 2 * (size_t)N * MAX, s), "zero(mesh wrenches)");
    }
}

void SdfStore::commit(cudaStream_t s) {
    commitK<<<blocks(N * MAX), 256, 0, s>>>(d_cur, d_tgt, d_tele, d_poses, d_box, count(), N * MAX);
    check(cudaGetLastError(), "commit(sdf poses)");
}

void SdfStore::end_step(int substeps, cudaStream_t s) {
    lastSubsteps = substeps > 0 ? substeps : 1;
    if (numVerts == 0) return;
    if (moving) {
        commit(s);
        moving = false;
        pendingSet = false;
    }
    if (forcesOn) {
        const float k = 1.f / (float)lastSubsteps;
        scaleK<<<blocks(3 * numVerts), 256, 0, s>>>((float*)d_force, 3 * numVerts, k);
        scaleK<<<blocks(6 * N * MAX), 256, 0, s>>>((float*)d_wrench, 6 * N * MAX, k);
        check(cudaGetLastError(), "average(contact forces)");
    }
    pairMeshCount = count();
    pairReady = pairCapacity != 0;
}

void SdfStore::contact_forces(float* out) const {
    if (!forcesOn || !d_force) throw std::invalid_argument("enable_contact_forces(True) first");
    check(cudaMemcpy(out, d_force, sizeof(glm::vec3) * numVerts, cudaMemcpyDeviceToHost), "download(contact forces)");
}

void SdfStore::wrenches(float* out) const {
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
    std::vector<Motion> motion(total);
    check(cudaMemcpy(current.data(), d_cur, sizeof(Pose7) * total, cudaMemcpyDeviceToHost), "download mesh current");
    check(cudaMemcpy(target.data(), d_tgt, sizeof(Pose7) * total, cudaMemcpyDeviceToHost), "download mesh target");
    check(cudaMemcpy(teleport.data(), d_tele, total, cudaMemcpyDeviceToHost), "download mesh teleport flags");
    check(cudaMemcpy(solver.data(), d_poses, sizeof(Pose) * total, cudaMemcpyDeviceToHost), "download mesh solver pose");
    check(cudaMemcpy(motion.data(), d_motion, sizeof(Motion)*total, cudaMemcpyDeviceToHost), "download mesh motion");
    if (pairCapacity) {
        result.pair_counts.resize(N);
        check(cudaMemcpy(result.pair_counts.data(), d_pairCount, sizeof(int)*N, cudaMemcpyDeviceToHost), "download pair counts");
    }
    const size_t size = (size_t)N * count();
    result.current.reserve(size); result.target.reserve(size);
    result.teleport.reserve(size); result.solver.reserve(size);
    for (int e = 0; e < N; ++e)
        for (int m = 0; m < count(); ++m) {
            const size_t slot = (size_t)e * MAX + m;
            result.current.push_back(current[slot]); result.target.push_back(target[slot]);
            result.teleport.push_back(teleport[slot]); result.solver.push_back(solver[slot]);
            result.motion.push_back(motion[slot]);
        }
    return result;
}
