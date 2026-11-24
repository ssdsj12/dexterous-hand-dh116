// Independent measurement candidate: every original vertex, FP64 only.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <limits>

constexpr int THREADS = 256;
__device__ __forceinline__ double minimum(double a, double b) {
  return fmin(a,b);
}
__device__ __forceinline__ double maximum(double a, double b) {
  return fmax(a,b);
}
__device__ __forceinline__ double warp_min(double v) {
  #pragma unroll
  for(int delta=16;delta;delta/=2) v=minimum(v,__shfl_down_sync(0xffffffff,v,delta));
  return v;
}
__device__ __forceinline__ double warp_max(double v) {
  #pragma unroll
  for(int delta=16;delta;delta/=2) v=maximum(v,__shfl_down_sync(0xffffffff,v,delta));
  return v;
}
// XYZ reductions retain NaN propagation so original aggregate output-finite
// guard rejects any invalid transform even though support minima are finite-only.
__device__ __forceinline__ double minimum_nan(double a,double b) {
  return (isnan(a)||isnan(b))?CUDART_NAN:fmin(a,b);
}
__device__ __forceinline__ double maximum_nan(double a,double b) {
  return (isnan(a)||isnan(b))?CUDART_NAN:fmax(a,b);
}
__device__ __forceinline__ double warp_min_nan(double v) {
  #pragma unroll
  for(int delta=16;delta;delta/=2)v=minimum_nan(v,__shfl_down_sync(0xffffffff,v,delta));
  return v;
}
__device__ __forceinline__ double warp_max_nan(double v) {
  #pragma unroll
  for(int delta=16;delta;delta/=2)v=maximum_nan(v,__shfl_down_sync(0xffffffff,v,delta));
  return v;
}
__global__ void transform_extrema(const double* vertices,const int64_t* offsets,
    const int64_t* sides,const int64_t* links,const double* rotations,
    const double* translations,int hulls,int64_t total,double* world,
    double* low,double* high) {
  const int hull=blockIdx.x%hulls, env=blockIdx.x/hulls;
  const int64_t begin=offsets[hull],end=offsets[hull+1];
  const int64_t body=env*46+sides[hull]*23+links[hull];
  const double* r=rotations+body*9;
  const double* t=translations+body*3;
  double mn[3]={CUDART_INF,CUDART_INF,CUDART_INF};
  double mx[3]={-CUDART_INF,-CUDART_INF,-CUDART_INF};
  for(int64_t v=begin+threadIdx.x;v<end;v+=THREADS) {
    double x=vertices[v*3],y=vertices[v*3+1],z=vertices[v*3+2];
    #pragma unroll
    for(int axis=0;axis<3;++axis) {
      double value=((x*r[axis*3]+y*r[axis*3+1])+z*r[axis*3+2])+t[axis];
      world[(env*total+v)*3+axis]=value;
      mn[axis]=minimum_nan(mn[axis],value);mx[axis]=maximum_nan(mx[axis],value);
    }
  }
  __shared__ double scratch[6][8];
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  #pragma unroll
  for(int axis=0;axis<3;++axis) {
    mn[axis]=warp_min_nan(mn[axis]);mx[axis]=warp_max_nan(mx[axis]);
    if(lane==0) {scratch[axis][warp]=mn[axis];scratch[axis+3][warp]=mx[axis];}
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int axis=0;axis<3;++axis) {
      double a=warp_min_nan(lane<8?scratch[axis][lane]:CUDART_INF);
      double b=warp_max_nan(lane<8?scratch[axis+3][lane]:-CUDART_INF);
      if(lane==0) {low[(env*hulls+hull)*3+axis]=a;high[(env*hulls+hull)*3+axis]=b;}
    }
  }
}

// One block per original hull; every world x/z tile is read from global once.
// Eight warps each track eight normals, covering all 64 without approximation.
__global__ void support_extrema(const double* world,const int64_t* offsets,
    const double* normals,int hulls,int64_t total,double* support) {
  const int hull=blockIdx.x%hulls,env=blockIdx.x/hulls;
  const int lane=threadIdx.x%32,warp=threadIdx.x/32;
  const int64_t begin=offsets[hull],end=offsets[hull+1];
  __shared__ double xs[512],zs[512];
  double nx[8],nz[8],value[8];
  #pragma unroll
  for(int k=0;k<8;++k) {
    const int normal=warp+k*8;
    nx[k]=normals[normal*3];nz[k]=normals[normal*3+2];value[k]=CUDART_INF;
  }
  for(int64_t tile=begin;tile<end;tile+=512) {
    int count=static_cast<int>(min(static_cast<int64_t>(512),end-tile));
    for(int i=threadIdx.x;i<count;i+=THREADS) {
      const double* point=world+(env*total+tile+i)*3;
      xs[i]=point[0];zs[i]=point[2];
    }
    __syncthreads();
    for(int i=lane;i<count;i+=32) {
      double x=xs[i],z=zs[i];
      #pragma unroll
      for(int k=0;k<8;++k) value[k]=minimum(value[k],x*nx[k]+z*nz[k]);
    }
    __syncthreads();
  }
  #pragma unroll
  for(int k=0;k<8;++k) {
    double v=warp_min(value[k]);
    if(lane==0) support[(env*hulls+hull)*64+warp+k*8]=v;
  }
}

extern "C" int dh116_hull_support(const double* vertices,const int64_t* offsets,
    const int64_t* sides,const int64_t* links,const double* rotations,
    const double* translations,const double* normals,int envs,int hulls,
    int64_t total,double* world,double* low,double* high,double* support,
    void* stream_ptr) {
  cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_ptr);
  transform_extrema<<<envs*hulls,THREADS,0,stream>>>(vertices,offsets,sides,links,
    rotations,translations,hulls,total,world,low,high);
  cudaError_t err=cudaGetLastError();if(err!=cudaSuccess)return static_cast<int>(err);
  support_extrema<<<envs*hulls,THREADS,0,stream>>>(world,offsets,normals,hulls,total,support);
  return static_cast<int>(cudaGetLastError());
}
extern "C" const char* dh116_hull_error(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}
