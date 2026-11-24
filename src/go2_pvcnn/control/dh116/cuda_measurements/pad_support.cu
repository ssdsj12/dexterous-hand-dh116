// Original FP64 pad sampling and unchanged FP32-routing envelope, standalone.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

// torch clamp/amax propagate NaN. fmin/fmax alone intentionally suppress it.
__device__ __forceinline__ double pad_maximum_nan(double a,double b) {
  return (isnan(a)||isnan(b))?CUDART_NAN:fmax(a,b);
}
__device__ __forceinline__ double pad_minimum_nan(double a,double b) {
  return (isnan(a)||isnan(b))?CUDART_NAN:fmin(a,b);
}
__device__ __forceinline__ double dot3(double a,double b,double c,double x,double y,double z) {
  return (a*x+b*y)+c*z;
}
__device__ __forceinline__ double grid_read(const double* v,int64_t base,int64_t ny,int64_t nz,
    int64_t ix,int64_t iy,int64_t iz) {
  return v[base+(ix*ny+iy)*nz+iz];
}
__global__ void pad_metrics_kernel(const double* values,const int64_t* offsets,const int64_t* dims,
    const int64_t* body_ids,const double* origins,const double* spacings,const double* roundings,
    const double* rotations,const double* translations,const double* nodes,const double* radii,
    const double* thresholds,int pads,int threshold_count,double* gaps,double* norms,
    unsigned char* valid_out,unsigned char* fallback) {
  const int pad=blockIdx.x%pads,env=blockIdx.x/pads,node=threadIdx.x;
  __shared__ double candidate_gap[64],candidate_error[64];
  __shared__ unsigned char route[64],has_valid[64];
  candidate_gap[node]=CUDART_INF;candidate_error[node]=0;route[node]=0;has_valid[node]=0;
  if(node<41) {
    const int64_t body=env*46+body_ids[pad];
    const double* r=rotations+body*9;const double* t=translations+body*3;
    const double* point=nodes+(env*41+node)*3;
    const double* origin=origins+pad*3;const double spacing=spacings[pad],inv=1./spacing;
    const double* rounding=roundings+pad*9;
    const int64_t* shape=dims+pad*3;
    const double eps=0x1p-23;
    double relative[3],relative32[3],local[3],local_error[3],guard_u[3],u_error[3],frac[3];
    int64_t index[3];bool valid=true,possible=true,face=false;
    #pragma unroll
    for(int axis=0;axis<3;++axis) {
      relative[axis]=point[axis]-t[axis];
      relative32[axis]=static_cast<double>(__fsub_rn(__double2float_rn(point[axis]),__double2float_rn(t[axis])));
    }
    #pragma unroll
    for(int axis=0;axis<3;++axis) {
      local[axis]=dot3(relative[0],relative[1],relative[2],r[axis],r[3+axis],r[6+axis]);
      const double sample_u=(local[axis]-origin[axis])*inv;
      valid=valid && sample_u>=0 && sample_u<static_cast<double>(shape[axis]-1);
      // Form indices only after full xyz validity below, as in the original masked sampler.
      guard_u[axis]=(local[axis]-origin[axis])/spacing;
      const double a=static_cast<double>(__double2float_rn(r[axis]));
      const double b=static_cast<double>(__double2float_rn(r[3+axis]));
      const double c=static_cast<double>(__double2float_rn(r[6+axis]));
      const double center=dot3(relative32[0],relative32[1],relative32[2],a,b,c);
      const double magnitude=dot3(fabs(relative32[0]),fabs(relative32[1]),fabs(relative32[2]),fabs(a),fabs(b),fabs(c));
      local_error[axis]=(fabs(center-local[axis])+8*eps*magnitude)+1e-12;
      const double u=guard_u[axis];
      u_error[axis]=((local_error[axis]+rounding[5+axis])/spacing+fabs(u)*rounding[8]/spacing)
                      +8*eps*(fabs(u)+1)+1e-10;
      const double band=pad_maximum_nan(u_error[axis],1e-4);
      possible=possible && u>=-band && u<=static_cast<double>(shape[axis]-1)+band;
      face=face || fabs(u-nearbyint(u))<=band;
    }
    #pragma unroll
    for(int axis=0;axis<3;++axis) {
      index[axis]=valid?static_cast<int64_t>((local[axis]-origin[axis])*inv):0;
      const double safe=valid?local[axis]:origin[axis];
      frac[axis]=pad_minimum_nan(pad_maximum_nan((safe-(origin[axis]+static_cast<double>(index[axis])*spacing))*inv,0.),1.);
    }
    const int64_t ix=index[0],iy=index[1],iz=index[2],ny=shape[1],nz=shape[2],base=offsets[pad];
    double c000=grid_read(values,base,ny,nz,ix,iy,iz),c001=grid_read(values,base,ny,nz,ix,iy,iz+1);
    double c010=grid_read(values,base,ny,nz,ix,iy+1,iz),c011=grid_read(values,base,ny,nz,ix,iy+1,iz+1);
    double c100=grid_read(values,base,ny,nz,ix+1,iy,iz),c101=grid_read(values,base,ny,nz,ix+1,iy,iz+1);
    double c110=grid_read(values,base,ny,nz,ix+1,iy+1,iz),c111=grid_read(values,base,ny,nz,ix+1,iy+1,iz+1);
    const double fx=frac[0],fy=frac[1],fz=frac[2];
    const double dx00=c100-c000,dx01=c101-c001,dx10=c110-c010,dx11=c111-c011;
    const double c00=c000+fx*dx00,c01=c001+fx*dx01,c10=c010+fx*dx10,c11=c011+fx*dx11;
    const double dy0=c10-c00,dy1=c11-c01,c0=c00+fy*dy0,c1=c01+fy*dy1;
    const double phi=c0+fz*(c1-c0);
    const double gx0=dx00+fy*(dx10-dx00),gx1=dx01+fy*(dx11-dx01);
    const double gx=(gx0+fz*(gx1-gx0))*inv,gy=(dy0+fz*(dy1-dy0))*inv,gz=(c1-c0)*inv;
    const double norm=sqrt((gx*gx+gy*gy)+gz*gz);
    const double gap=phi-radii[env];
    const double radius_error=fabs(static_cast<double>(__double2float_rn(radii[env]))-radii[env]);
    const double err0=u_error[0]*spacing*rounding[0],err1=u_error[1]*spacing*rounding[1],err2=u_error[2]*spacing*rounding[2];
    const double gap_error=pad_maximum_nan((((err0+err1)+err2)+rounding[3])+64*eps*rounding[4]+radius_error,2e-6);
    const double slope=pad_maximum_nan(pad_maximum_nan(rounding[0],rounding[1]),rounding[2]);
    const double error_sum=(local_error[0]+local_error[1])+local_error[2];
    const double norm_error=pad_maximum_nan(16*(rounding[3]+eps*rounding[4])/spacing+4*slope/spacing*error_sum,2e-6);
    bool near=false;
    for(int k=0;k<threshold_count;++k)near=near || (valid && fabs(gap-thresholds[k])<=gap_error);
    const bool flat=valid && norm<=norm_error;
    const int64_t out=(env*pads+pad)*41+node;
    gaps[out]=valid?gap:CUDART_NAN;norms[out]=valid?norm:CUDART_NAN;valid_out[out]=valid;
    candidate_gap[node]=valid?gap:CUDART_INF;candidate_error[node]=gap_error;
    route[node]=(possible && face)||near||flat;has_valid[node]=valid;
  }
  __syncthreads();
  if(node==0) {
    double first=CUDART_INF,second=CUDART_INF,max_error=0;bool any=false,flag=false;
    for(int k=0;k<41;++k) {
      const double v=candidate_gap[k];
      if(v<first){second=first;first=v;}else if(v<second){second=v;}
      max_error=pad_maximum_nan(max_error,candidate_error[k]);any=any||has_valid[k];flag=flag||route[k];
    }
    const bool tied=any && fabs(second-first)<=2*max_error;
    fallback[env*pads+pad]=flag||tied;
  }
}
extern "C" int dh116_pad_metrics(const double* values,const int64_t* offsets,const int64_t* dims,
    const int64_t* body_ids,const double* origins,const double* spacings,const double* roundings,
    const double* rotations,const double* translations,const double* nodes,const double* radii,
    const double* thresholds,int envs,int pads,int threshold_count,double* gaps,double* norms,
    unsigned char* valid,unsigned char* fallback,void* stream_ptr) {
  const cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_ptr);
  pad_metrics_kernel<<<envs*pads,64,0,stream>>>(values,offsets,dims,body_ids,origins,spacings,roundings,
    rotations,translations,nodes,radii,thresholds,pads,threshold_count,gaps,norms,valid,fallback);
  return static_cast<int>(cudaGetLastError());
}
