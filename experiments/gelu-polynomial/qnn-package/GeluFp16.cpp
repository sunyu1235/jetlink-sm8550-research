// Erf uses exhaustively corrected HVX polynomial; tanh retains exact table.
// Whole-model parity and real device timing remain separate acceptance gates.
#include <cstring>
#include <type_traits>
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "compact_gelu.h"
#include "vector_core.h"
#ifdef JL_POLY_COUNT
#include <atomic>
static std::atomic<unsigned long> poly_vectors{0};
extern "C" __attribute__((visibility("default"))) unsigned long jl_polynomial_vectors() {return poly_vectors.load();}
#endif
static inline HVX_Vector poly(HVX_Vector x) {
#ifdef JL_POLY_COUNT
 poly_vectors.fetch_add(1,std::memory_order_relaxed);
#endif
 return jl_erf_poly_vector(x);
}
BEGIN_PKG_OP_DEFINITION(PKG_GeluFp16);
template<unsigned Mode, typename OutTensor, typename InTensor>
static int geluFp16(OutTensor &out, const InTensor &in) {
  out.set_dims(in);
  static_assert(sizeof(*in.get_raw_addr(0,0,0,0))==sizeof(uint16_t), "Expected half payload");
  // QNN 2.42 R4Crouton2Layout interleaves two width positions:
  // each depth step advances two half elements. Never walk beyond a depth32
  // chunk, and never read padded elements at an incomplete width/depth edge.
  constexpr bool ci = std::is_same_v<InTensor, F16CroutonTensor>;
  constexpr bool co = std::is_same_v<OutTensor, F16CroutonTensor>;
  constexpr size_t si = ci ? 2 : 1, so = co ? 2 : 1;
  // Padding may shift logical width/depth origins within a physical stripe.
  // Fall back to logical addressing unless both origins preserve our alignment.
  bool aligned_origins=true;
  if constexpr (ci) aligned_origins &= (in.shape->pad[2]%2==0 && in.shape->pad[3]%32==0);
  if constexpr (co) aligned_origins &= (out.shape->pad[2]%2==0 && out.shape->pad[3]%32==0);
#ifdef JL_GELU_FORCE_FALLBACK
  aligned_origins=false;
#endif
  if (!aligned_origins) {
    for(size_t b=0;b<in.dim(0);++b)for(size_t h=0;h<in.dim(1);++h)
      for(size_t w=0;w<in.dim(2);++w)for(size_t d=0;d<in.dim(3);++d) {
        uint16_t x,y;std::memcpy(&x,in.get_raw_addr(b,h,w,d),2);
        y=jl_gelu_half(x,Mode);std::memcpy(out.get_raw_addr(b,h,w,d),&y,2);
      }
    return GraphStatus::Success;
  }
  for (size_t b=0; b<in.dim(0); ++b)
    for (size_t h=0; h<in.dim(1); ++h)
      for (size_t w=0; w<in.dim(2); ++w)
        for (size_t d=0; d<in.dim(3);) {
          if constexpr (Mode==0) {
            if constexpr (!ci && !co) {
              if (d+64<=in.dim(3)) {
                HVX_Vector x;
                std::memcpy(&x,in.get_raw_addr(b,h,w,d),128);
                HVX_Vector y=poly(x);
                std::memcpy(out.get_raw_addr(b,h,w,d),&y,128);
                d+=64;continue;
              }
            } else {
              if (w%2==0 && w+1<in.dim(2) && d+32<=in.dim(3)) {
                HVX_Vector x;
                if constexpr (ci) {
                  std::memcpy(&x,in.get_raw_addr(b,h,w,d),128);
                } else {
                  // Two independent logical depth runs, no inter-row adjacency assumption.
                  std::memcpy(&x,in.get_raw_addr(b,h,w,d),64);
                  std::memcpy(reinterpret_cast<unsigned char*>(&x)+64,in.get_raw_addr(b,h,w+1,d),64);
                  x=Q6_Vh_vshuff_Vh(x);
                }
                HVX_Vector y=poly(x);
                if constexpr (co) {
                  std::memcpy(out.get_raw_addr(b,h,w,d),&y,128);
                } else {
                  y=Q6_Vh_vdeal_Vh(y);
                  // Exactly64 bytes per destination. Never write a padded128-byte row.
                  std::memcpy(out.get_raw_addr(b,h,w,d),&y,64);
                  std::memcpy(out.get_raw_addr(b,h,w+1,d),reinterpret_cast<const unsigned char*>(&y)+64,64);
                }
                d+=32;continue;
              }
              if (w%2==1 && d+32<=in.dim(3)) {d+=32;continue;}
            }
          }
          // Adjacent width positions share one contiguous 64-half stripe.
          // Process both together only when the entire stripe is logical data.
          if constexpr (ci && co) {
            if (w%2==0 && w+1<in.dim(2) && d+32<=in.dim(3)) {
              const auto *ip=in.get_raw_addr(b,h,w,d);
              auto *op=out.get_raw_addr(b,h,w,d);
              for (size_t k=0;k<64;++k) {
                uint16_t x,y; std::memcpy(&x,ip+k,2);
                y=jl_gelu_half(x,Mode); std::memcpy(op+k,&y,2);
              }
              d+=32; continue;
            }
            if (w%2==1 && d+32<=in.dim(3)) { d+=32; continue; }
          }
          size_t n=in.dim(3)-d;
          if constexpr (ci || co) { if (n>32) n=32; }
          const auto *ip=in.get_raw_addr(b,h,w,d);
          auto *op=out.get_raw_addr(b,h,w,d);
          for(size_t k=0; k<n; ++k) {
            uint16_t x,y;
            std::memcpy(&x,ip+k*si,2);
            y=jl_gelu_half(x,Mode);
            std::memcpy(op+k*so,&y,2);
          }
          d+=n;
        }
  return GraphStatus::Success;
}
DEF_TENSOR_PROPERTIES(Op("GeluErfFp16", "in0"), MainMemory("*", "in0"))
DEF_TENSOR_PROPERTIES(Op("GeluTanhFp16", "in0"), MainMemory("*", "in0"))
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((geluFp16<0, PlainFloat16Tensor, PlainFloat16Tensor>), "GeluErfFp16", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((geluFp16<0, PlainFloat16Tensor, F16CroutonTensor>), "GeluErfFp16", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((geluFp16<0, F16CroutonTensor, PlainFloat16Tensor>), "GeluErfFp16", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((geluFp16<0, F16CroutonTensor, F16CroutonTensor>), "GeluErfFp16", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((geluFp16<1, PlainFloat16Tensor, PlainFloat16Tensor>), "GeluTanhFp16", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((geluFp16<1, PlainFloat16Tensor, F16CroutonTensor>), "GeluTanhFp16", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((geluFp16<1, F16CroutonTensor, PlainFloat16Tensor>), "GeluTanhFp16", SNAIL, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((geluFp16<1, F16CroutonTensor, F16CroutonTensor>), "GeluTanhFp16", SNAIL, Flags::RESOURCE_HVX)
END_PKG_OP_DEFINITION(PKG_GeluFp16);
