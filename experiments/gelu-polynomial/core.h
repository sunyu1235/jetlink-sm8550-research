#pragma once
#include <stdint.h>
// Aligned 64-half blocks only. Exact in-place allowed. No partial overlap.
extern "C" void jl_erf_poly64(uint16_t *out, const uint16_t *in);
extern "C" void jl_test_lookup(uint32_t *out, const uint16_t *indices);
extern "C" void jl_test_indices(uint16_t *out, const uint16_t *in);
