#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compact_gelu.h"
#define CHECK(x) do {if (!(x)) {fprintf(stderr,"failed line %d\n",__LINE__);return 1;}} while(0)
int main(int argc,char **argv) {
  CHECK(argc==2); FILE *f=fopen(argv[1],"rb"); CHECK(f);
  uint16_t *in=malloc(65536*2),*out=malloc(65536*2),*ref=malloc(65536*2); CHECK(in&&out&&ref);
  size_t count=0;
  for(unsigned mode=0;mode<2;++mode) {
    for(unsigned i=0;i<65536;++i) { unsigned char b[2]; CHECK(fread(b,1,2,f)==2); ref[i]=(uint16_t)(b[0]|((unsigned)b[1]<<8)); in[i]=(uint16_t)i; }
    CHECK(jl_gelu_run(out,in,65536,mode)==0);
    for(unsigned i=0;i<65536;++i) {CHECK(out[i]==ref[i]); ++count;}
    CHECK(jl_gelu_run(in,in,65536,mode)==0); CHECK(memcmp(in,ref,131072)==0);
    for(size_t n=0;n<260;++n) {
      uint16_t a[262],b[262]; for(size_t i=0;i<262;++i) {a[i]=(uint16_t)(i*251);b[i]=12345;}
      CHECK(jl_gelu_run(b+1,a+1,n,mode)==0); CHECK(b[0]==12345&&b[n+1]==12345);
      for(size_t i=0;i<n;++i) CHECK(b[i+1]==ref[a[i+1]]);
    }
  }
  CHECK(fgetc(f)==EOF); CHECK(jl_gelu_run(NULL,NULL,0,0)==0); CHECK(jl_gelu_run(out,in,1,2)==-1); CHECK(jl_gelu_run(NULL,in,1,0)==-1);
  printf("PASS %zu bit-pattern comparisons, two full in-place passes, 520 tail/canary cases, invalid arguments\n",count);
  fclose(f); free(in);free(out);free(ref);return 0;
}
