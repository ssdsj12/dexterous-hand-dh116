"""Device-only reductions of the source-qualified packed native pair view."""
import math
import torch

def tensor_pair_capture(raw,*,n,vertices,meshes,substeps,device):
    if (raw.get("schema")!="dh116.sdf_pair_device.v1" or raw["substeps"]!=substeps
            or raw["num_envs"]!=n or raw["mesh_count"]!=meshes or not 2<=n<=512
            or raw["last_error"]!=0 or raw["owner_time_s_before"]!=raw["owner_time_s_after"]
            or not math.isfinite(raw["owner_time_s_before"])):
        raise ValueError("source-qualified current device contact ABI required")
    counts=torch.as_tensor(raw["counts"],device=device)
    samples=torch.as_tensor(raw["samples"],device=device)
    if (counts.shape!=(n,) or counts.dtype!=torch.int32 or samples.dtype!=torch.float32
            or samples.ndim!=3 or samples.shape!=(n,raw["capacity"],24)
            or bool(((counts<0)|(counts>raw["capacity"])).any())):
        raise ValueError("bounded complete native per-lane pair buckets required")
    mask=torch.arange(samples.shape[1],device=device)[None,:]<counts[:,None]
    values=samples[mask].clone()
    owners=torch.arange(n,device=device)[:,None].expand_as(mask)[mask]
    ids=values[:,:3].contiguous().view(torch.int32).long()
    vectors=values[:,3:15].double()
    limits=ids.new_tensor([substeps,vertices,meshes])
    if (bool(((ids<0)|(ids>=limits)).any()) or not bool(torch.isfinite(values[:,3:]).all())):
        raise ValueError("actual pair indices or floating fields invalid")
    key=((owners*substeps+ids[:,0])*vertices+ids[:,1])*meshes+ids[:,2]
    if torch.unique(key).numel()!=key.numel():raise ValueError("duplicate actual contact pair")
    normal=vectors[:,3:6];force=vectors[:,6:9]
    normal_force=(normal*force).sum(-1)
    if (not bool(torch.isclose(normal.norm(dim=-1),torch.ones_like(normal_force),atol=2e-5,rtol=0).all())
            or bool((normal_force< -1e-5).any())):
        raise ValueError("outward unit normal and compressive actual force required")
    nodes=torch.zeros(n*vertices,3,dtype=torch.float64,device=device)
    wrench=torch.zeros(n*meshes,6,dtype=torch.float64,device=device)
    nodes.index_add_(0,owners*vertices+ids[:,1],force/substeps)
    wrench.index_add_(0,owners*meshes+ids[:,2],torch.cat((-force,vectors[:,9:12]),-1)/substeps)
    nodes=nodes.reshape(n,vertices,3);wrench=wrench.reshape(n,meshes,6)
    reference_nodes=torch.as_tensor(raw["node_forces"],device=device).double()
    reference_wrench=torch.as_tensor(raw["mesh_wrenches"],device=device)[:,:meshes].double()
    if (not torch.allclose(nodes,reference_nodes,atol=3e-4,rtol=5e-5)
            or not torch.allclose(wrench,reference_wrench,atol=3e-4,rtol=5e-5)):
        raise ValueError("actual pair reductions disagree with native totals")
    return dict(counts=counts.long().clone(),env_ids=owners,indices=ids,vectors=vectors,
                normal_force=normal_force,node_forces=nodes,mesh_wrenches=wrench)

def flat_cuda_pair_capture(raw,*,n,vertices,meshes,substeps,device,expected_owner_time_s):
    """Validate owned compact tensors from the read-once flat ABI."""
    if (raw.get("schema")!="dh116.sdf_pair_flat_cuda.v1" or raw["substeps"]!=substeps
            or raw["num_envs"]!=n or raw["mesh_count"]!=meshes
            or raw["owner_time_s_before"]!=expected_owner_time_s
            or raw["owner_time_s_after"]!=expected_owner_time_s):
        raise ValueError("current complete flat CUDA boundary required")
    ids=raw["indices_gpu"];vectors=raw["vectors_gpu"];meta=raw["lane_meta_gpu"]
    if (not all(isinstance(x,torch.Tensor) and x.device==torch.device(device)
                for x in (ids,vectors,meta,raw["node_forces_gpu"],raw["mesh_wrenches_gpu"]))
            or ids.ndim!=2 or ids.shape[1]!=4 or ids.dtype!=torch.int32
            or vectors.shape!=(len(ids),12) or vectors.dtype!=torch.float32
            or meta.shape!=(n,4) or meta.dtype!=torch.int32
            or bool(((meta[:,0]!=0)|(meta[:,1]!=0)|(meta[:,2]<1)|(meta[:,3]<0)|(meta[:,3]>meta[:,2])).any())
            or not bool(meta[:,3].sum()==len(ids))):
        raise ValueError("owned finite-capacity actual flat CUDA tensors required")
    index=ids.long();env_id=index[:,0];pair=index[:,1:]
    limits=index.new_tensor([n,substeps,vertices,meshes])
    val=vectors.double()
    if bool(((index<0)|(index>=limits)).any()) or not bool(torch.isfinite(val).all()):
        raise ValueError("invalid flat actual pair")
    counts=torch.bincount(env_id,minlength=n)
    if not torch.equal(counts,meta[:,3].long()):raise ValueError("flat pair lane count differs")
    key=((env_id*substeps+pair[:,0])*vertices+pair[:,1])*meshes+pair[:,2]
    if torch.unique(key).numel()!=len(key):raise ValueError("duplicate flat actual pair")
    normal=val[:,3:6];force=val[:,6:9];nf=(normal*force).sum(-1)
    if (not bool(torch.isclose(normal.norm(dim=-1),torch.ones_like(nf),atol=2e-5,rtol=0).all())
            or bool((nf< -1e-5).any())):raise ValueError("invalid flat contact normal/force")
    nodes=val.new_zeros(n*vertices,3);wrench=val.new_zeros(n*meshes,6)
    nodes.index_add_(0,env_id*vertices+pair[:,1],force/substeps)
    wrench.index_add_(0,env_id*meshes+pair[:,2],torch.cat((-force,val[:,9:12]),-1)/substeps)
    nodes=nodes.reshape(n,vertices,3);wrench=wrench.reshape(n,meshes,6)
    if (not torch.allclose(nodes,raw["node_forces_gpu"].double(),atol=3e-4,rtol=5e-5)
            or not torch.allclose(wrench,raw["mesh_wrenches_gpu"].double(),atol=3e-4,rtol=5e-5)):
        raise ValueError("flat pair native totals differ")
    return dict(counts=counts,env_ids=env_id,indices=pair,vectors=val,
        normal_force=nf,node_forces=nodes,mesh_wrenches=wrench)
