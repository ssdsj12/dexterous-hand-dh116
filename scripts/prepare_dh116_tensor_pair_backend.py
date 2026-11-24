#!/usr/bin/env python3
"""Prepare an independent source-locked device contact-view backend; no install."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

HEADER = r"""
    struct PairDeviceView {
        uintptr_t samples, counts;
        int capacity, substeps, mesh_count;
    };
    PairDeviceView pair_device_view() const {
        if (!pairReady || !pairCapacity || !d_pairs || !d_pairCount)
            throw std::invalid_argument("completed pair device buffer required");
        return {(uintptr_t)d_pairs,(uintptr_t)d_pairCount,pairCapacity,lastSubsteps,pairMeshCount};
    }
"""
STRUCT = r"""
struct PairCudaView {
    uintptr_t ptr;
    std::vector<py::ssize_t> shape, strides;
    std::string typestr;
    py::object owner;
};
"""
BINDING = r"""
    py::class_<PairCudaView>(m,"PairCudaView")
        .def_property_readonly("__cuda_array_interface__",[](const PairCudaView& v) {
            py::dict out;
            out["shape"]=py::cast(v.shape);out["strides"]=py::cast(v.strides);
            out["typestr"]=v.typestr;out["data"]=py::make_tuple(v.ptr,false);
            out["version"]=3;out["stream"]=1;
            return out;
        });
"""
FLAT_METHOD = r"""
        .def("contact_pair_samples_flat_cuda_v1",[](py::object self) {
            RopeBatch& b=self.cast<RopeBatch&>();
            std::lock_guard<std::mutex> g(b.mutex());
            const float time=b.raw()->meta.time;
            if (b.flat_read_valid && b.flat_read_time==time)
                throw py::value_error("completed device pair step already consumed");
            auto v=b.sdf.pair_device_view();
            if (b.raw()->lastErrorCode) throw py::value_error("native solver error");
            const auto N=py::ssize_t(b.num_envs()),C=py::ssize_t(v.capacity);
            auto torch=py::module_::import("torch");
            auto cuda=torch.attr("cuda");
            auto current=cuda.attr("current_stream")().attr("cuda_stream").cast<uintptr_t>();
            auto expected=cuda.attr("default_stream")().attr("cuda_stream").cast<uintptr_t>();
            if (current!=expected) throw py::value_error("flat contact ABI requires legacy/default stream");
            auto tensor=[&](PairCudaView view) {
                return torch.attr("as_tensor")(py::cast(std::move(view)),py::arg("device")="cuda");
            };
            auto raw=tensor(PairCudaView{v.samples,{N,C,24},{C*96,96,4},"<f4",self});
            auto count=tensor(PairCudaView{v.counts,{N},{4},"<i4",self}).attr("clone")();
            auto node=tensor(PairCudaView{b.sdf.contact_forces_ptr(),{N,b.verts_per_rope(),3},
                {b.verts_per_rope()*12,12,4},"<f4",self}).attr("clone")();
            auto mesh=tensor(PairCudaView{b.sdf.wrenches_ptr(),{N,64,6},{64*24,24,4},"<f4",self});
            auto valid=torch.attr("lt")(torch.attr("arange")(C,py::arg("device")="cuda").attr("unsqueeze")(0),
                                        count.attr("unsqueeze")(1));
            // Boolean compaction sizes the owned result from the actual completed GPU count.
            // Torch may synchronize a scalar for exact dynamic output shape; no pair rows go to host.
            auto compact=raw.attr("__getitem__")(valid);
            py::slice all{py::none(),py::none(),py::none()};
            auto ijk=compact.attr("__getitem__")(py::make_tuple(all,py::slice(0,3,1)))
                .attr("contiguous")().attr("view")(torch.attr("int32"));
            auto env=torch.attr("arange")(N,py::arg("device")="cuda",py::arg("dtype")=torch.attr("int32"))
                .attr("unsqueeze")(1).attr("expand")(N,C).attr("__getitem__")(valid).attr("unsqueeze")(1);
            py::dict out;
            out["schema"]="dh116.sdf_pair_flat_cuda.v1";
            out["indices_gpu"]=torch.attr("cat")(py::make_tuple(env,ijk),1).attr("contiguous")();
            out["vectors_gpu"]=compact.attr("__getitem__")(py::make_tuple(all,py::slice(3,15,1))).attr("contiguous")().attr("clone")();
            out["lane_meta_gpu"]=torch.attr("stack")(py::make_tuple(
                torch.attr("zeros_like")(count),
                torch.attr("gt")(count,v.capacity).attr("to")(torch.attr("int32")),
                torch.attr("full_like")(count,v.capacity),count),1);
            out["node_forces_gpu"]=node;
            out["mesh_wrenches_gpu"]=mesh.attr("__getitem__")(py::make_tuple(all,py::slice(0,v.mesh_count,1),all)).attr("contiguous")().attr("clone")();
            out["num_envs"]=N;out["mesh_count"]=v.mesh_count;out["substeps"]=v.substeps;
            out["owner_time_s_before"]=time;out["owner_time_s_after"]=b.raw()->meta.time;
            out["sample_source"]="cuda_surfaceReadout_same_end_substep_evaluation";
            b.flat_read_time=time;b.flat_read_valid=true;
            return out;
        },"Owned compact GPU pair tensors; exact dynamic shape, read once per step.")
"""

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(profile_path,destination):
    profile=json.loads(profile_path.read_text())
    if destination.exists():raise ValueError("fresh independent output required")
    for path,sha in profile["source_inventory"].items():
        if digest(path)!=sha:raise ValueError("qualified backend source changed: "+path)
    original=Path(profile["binary_path"]).parents[2]/"source"
    destination.mkdir()
    shutil.copytree(original,destination/"source")
    source=destination/"source"
    h=source/"csrc/sdf.h";text=h.read_text()
    anchor="    using PairSample = YarnBall::MetaData::SdfPairSample;"
    if text.count(anchor)!=1:raise ValueError("source header anchor changed")
    h.write_text(text.replace(anchor,anchor+HEADER))
    p=source/"csrc/bindings.cpp";text=p.read_text()
    a="PYBIND11_MODULE(_ropesim, m) {"
    text=text.replace(a,STRUCT+chr(10)+a+chr(10)+BINDING)
    a='        .def("contact_pair_samples_batched_v1",'
    if text.count(a)!=1:raise ValueError("source binding anchor changed")
    p.write_text(text.replace(a,FLAT_METHOD+a))
    batch=source/"csrc/batch.h"
    text=batch.read_text()
    anchor="class RopeBatch {"+chr(10)+"public:"
    if text.count(anchor)!=1:raise ValueError("batch owner class anchor changed")
    batch.write_text(text.replace(anchor,anchor+chr(10)+"    bool flat_read_valid=false; float flat_read_time=0;"))
    (destination/"input_profile.json").write_bytes(profile_path.read_bytes())
    (destination/"patch_inventory.json").write_text(json.dumps({
        "upstream_commit":profile["upstream_commit"],
        "input_profile_sha256":digest(profile_path),
        "base_binary_sha256":profile["binary_sha256"],
        "source_patch":{str(p.relative_to(source)):digest(p) for p in (h,p,batch)},
        "physics_kernel_changes":False,"system_install":False},indent=2))
    return destination

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(prepare(args.profile,args.output))

if __name__=="__main__":main()
