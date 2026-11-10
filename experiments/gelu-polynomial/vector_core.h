#pragma once
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
// 64 independent raw FP16 lanes. No alignment, layout, memory, or VTCM contract.
// This helper owns no mutable state. The caller supplies vector load/store/layout.
HVX_Vector jl_erf_poly_vector(HVX_Vector raw);
