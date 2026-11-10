#ifndef JL_COMPACT_GELU_H
#define JL_COMPACT_GELU_H
#include <stddef.h>
#include "gelu_tables.h"
/* Modes: 0 = erf vision, 1 = tanh policy. Return error for invalid mode.
 * Raw uint16 half payloads, NOT QNN tensor-layout or ABI assumptions.
 * Exact in-place permitted. Partially overlapping input/output not supported.
 * Tables are immutable and initialization-free. No heap or libm at runtime. */
static inline uint16_t jl_gelu_half(uint16_t x, unsigned mode) {
  uint16_t page=jl_gelu_pages[mode][x >> JL_GELU_PAGE_SHIFT];
  if (page==65535) return x;
  if (page==65534) return 0;
  if (page==65533) return 32768;
  return jl_gelu_bank[page][x & JL_GELU_PAGE_MASK];
}
static inline int jl_gelu_run(uint16_t *out, const uint16_t *in, size_t n, unsigned mode) {
  if (mode>1 || (n && (!out || !in))) return -1;
  for (size_t i=0;i<n;++i) out[i]=jl_gelu_half(in[i],mode);
  return 0;
}
#endif
