/* Reference equations copied from reports/jetlink-gelu-repair-20261101/exact_gelu.c.
 * Host-only generator: do not compile/run with fast-math. */
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
int main(void) {
  for (unsigned mode=0; mode<2; ++mode) {
    for (unsigned i=0; i<65536; ++i) {
      uint16_t bits=(uint16_t)i, result;
      _Float16 half; memcpy(&half,&bits,2);
      double x=(double)half;
      double f=mode ? tanh(0.79788456080286535588*(x+0.044715*x*x*x))
                    : erf(x*0.70710678118654752440);
      half=(_Float16)(0.5*x*(1.0+f)); memcpy(&result,&half,2);
      unsigned char le[2]={(unsigned char)result,(unsigned char)(result>>8)};
      if (fwrite(le,1,2,stdout)!=2) return 1;
    }
  }
  return 0;
}
