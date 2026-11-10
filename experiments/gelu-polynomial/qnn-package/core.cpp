#include "core.h"
#include "vector_core.h"
#include "coefficients.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
static inline HVX_Vector load(const void* p) {return *reinterpret_cast<const HVX_Vector*>(p);}
static inline void save(void* p,HVX_Vector v) {*reinterpret_cast<HVX_Vector*>(p)=v;}
// Each input half contains index in low byte, index+32 in high byte.
// vlut16's first pair member supplies coefficient low halves; second supplies highs.
// qf32 widening groups even/odd HALF lanes, requiring word deinterleave below.
static inline HVX_VectorPair lookup(HVX_Vector indices,const uint32_t* table) {
 HVX_Vector ix=Q6_V_vor_VV(indices,Q6_Vh_vasl_VhR(Q6_Vh_vadd_VhVh(indices,Q6_Vh_vsplat_R(32)),8));
 HVX_Vector t=load(table);
 HVX_VectorPair parts=Q6_Wh_vlut16_VbVhI(ix,t,0);
 parts=Q6_Wh_vlut16or_WhVbVhI(parts,ix,t,1);
 parts=Q6_Wh_vlut16or_WhVbVhI(parts,ix,t,2);
 parts=Q6_Wh_vlut16or_WhVbVhI(parts,ix,t,3);
 HVX_VectorPair sequential=Q6_W_vshuff_VVR(Q6_V_hi_W(parts),Q6_V_lo_W(parts),-2);
 return Q6_W_vdeal_VVR(Q6_V_hi_W(sequential),Q6_V_lo_W(sequential),-4);
}
static inline HVX_Vector indices(HVX_Vector x) {
 HVX_Vector mag=Q6_V_vand_VV(x,Q6_Vh_vsplat_R(0x7fff));
 HVX_Vector exp=Q6_Vuh_vlsr_VuhR(mag,10);
 HVX_Vector mant=Q6_V_vor_VV(Q6_V_vand_VV(mag,Q6_Vh_vsplat_R(1023)),Q6_Vh_vsplat_R(1024));
 HVX_Vector shift=Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(24),exp);
 shift=Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(shift,Q6_Vh_vsplat_R(0)),Q6_Vh_vsplat_R(15));
 HVX_Vector q=Q6_Vh_vlsr_VhVh(mant,shift);
 HVX_Vector back=Q6_Vh_vasl_VhVh(q,shift);
 HVX_VectorPred fractional=Q6_Q_vcmp_gt_VuhVuh(mant,back);
 HVX_Vector ceil=Q6_Vh_vadd_VhVh(q,Q6_V_vmux_QVV(fractional,Q6_Vh_vsplat_R(1),Q6_V_vzero()));
 HVX_VectorPred negative=Q6_Q_vcmp_gt_VuhVuh(x,Q6_Vh_vsplat_R(0x7fff));
 HVX_Vector signedq=Q6_V_vmux_QVV(negative,Q6_Vh_vsub_VhVh(Q6_V_vzero(),ceil),q);
 signedq=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(mag,Q6_V_vzero()),Q6_V_vzero(),signedq);
 HVX_Vector j=Q6_Vh_vadd_VhVh(signedq,Q6_Vh_vsplat_R(16));
 return Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(j,Q6_V_vzero()),Q6_Vh_vsplat_R(31));
}
extern "C" void jl_test_lookup(uint32_t*out,const uint16_t*in) {
 auto p=lookup(load(in),coefficients[0]);save(out,Q6_V_lo_W(p));save(out+32,Q6_V_hi_W(p));
}
extern "C" void jl_test_indices(uint16_t*out,const uint16_t*in) {save(out,indices(load(in)));}
HVX_Vector jl_erf_poly_vector(HVX_Vector raw) {
 HVX_Vector j=indices(raw);
 // Sanitize tails/specials before floating arithmetic; final raw masks restore them.
 HVX_Vector mag=Q6_V_vand_VV(raw,Q6_Vh_vsplat_R(0x7fff));
 HVX_VectorPred interior=Q6_Q_vcmp_gt_VuhVuh(Q6_Vh_vsplat_R(0x4800),mag);
 HVX_Vector x=Q6_V_vmux_QVV(interior,raw,Q6_V_vzero());
 HVX_VectorPair wide=Q6_Wqf32_vmpy_VhfVhf(x,Q6_Vh_vsplat_R(0x3c00));
 HVX_Vector x0=Q6_Vsf_equals_Vqf32(Q6_V_lo_W(wide));
 HVX_Vector x1=Q6_Vsf_equals_Vqf32(Q6_V_hi_W(wide));
 HVX_VectorPair mid=lookup(j,coefficients[0]);
 HVX_Vector r0=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(x0,Q6_V_lo_W(mid)));
 HVX_Vector r1=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(x1,Q6_V_hi_W(mid)));
 HVX_VectorPair c=lookup(j,coefficients[7]);
 HVX_Vector y0=Q6_V_lo_W(c), y1=Q6_V_hi_W(c);
 for(int k=5;k>=0;--k){
  c=lookup(j,coefficients[k+1]);
  y0=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y0,r0));
  y1=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y1,r1));
  y0=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(y0,Q6_V_lo_W(c)));
  y1=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(y1,Q6_V_hi_W(c)));
 }
 y0=Q6_Vqf32_vmpy_VsfVsf(x0,y0);y1=Q6_Vqf32_vmpy_VsfVsf(x1,y1);
 HVX_Vector result=Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(y1,y0));
 HVX_VectorPred negative=Q6_Q_vcmp_gt_VuhVuh(raw,Q6_Vh_vsplat_R(0x7fff));
 HVX_Vector tail=Q6_V_vmux_QVV(negative,Q6_Vh_vsplat_R(0x8000),raw);
 result=Q6_V_vmux_QVV(interior,result,tail);
 // Exact exceptional behavior from exhaustive native21/reference comparison.
 // The tiny negative interval rounds toward the smaller magnitude reference half.
 HVX_VectorPred tiny=Q6_Q_vcmp_gt_VuhVuh(Q6_Vh_vsplat_R(0x32),mag);
 tiny=Q6_Q_and_QQ(tiny,negative);
 HVX_Vector tiny_result=Q6_V_vor_VV(Q6_Vuh_vlsr_VuhR(mag,1),Q6_Vh_vsplat_R(0x8000));
 result=Q6_V_vmux_QVV(tiny,tiny_result,result);
 result=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(raw,Q6_Vh_vsplat_R(0x0e55)),Q6_Vh_vsplat_R(0x0a55),result);
 result=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(raw,Q6_Vh_vsplat_R(0x157c)),Q6_Vh_vsplat_R(0x117d),result);
 result=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(raw,Q6_Vh_vsplat_R(0x2e9e)),Q6_Vh_vsplat_R(0x2b29),result);
 // Quiet NaNs preserve sign/payload. -infinity becomes negative quiet NaN.
 HVX_VectorPred nan=Q6_Q_vcmp_gt_VuhVuh(mag,Q6_Vh_vsplat_R(0x7c00));
 result=Q6_V_vmux_QVV(nan,Q6_V_vor_VV(raw,Q6_Vh_vsplat_R(0x0200)),result);
 result=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(raw,Q6_Vh_vsplat_R(0xfc00)),Q6_Vh_vsplat_R(0xfe00),result);
 return result;
}
extern "C" void jl_erf_poly64(uint16_t*out,const uint16_t*in) {
 save(out,jl_erf_poly_vector(load(in)));
}
