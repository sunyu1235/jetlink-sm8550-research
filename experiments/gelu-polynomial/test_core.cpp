// Authored standalone HVX/libnative test, no QNN runtime required.
#include "core.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#ifndef JL_NO_THREADS
#include <thread>
#include <atomic>
#endif
alignas(128) static uint16_t reference[65536];
static float half_value(uint16_t u){
 unsigned e=(u>>10)&31, m=u&1023;
 float v=e?std::ldexp(float(1024+m),int(e)-25):std::ldexp(float(m),-24);
 return (u&0x8000)?-v:v;
}
static int mechanics(){
 alignas(128) uint16_t in[64],out[64];alignas(128) uint32_t words[64];
 for(unsigned pass=0;pass<8;++pass){
  for(unsigned i=0;i<64;++i)in[i]=(i*(2*pass+1)+pass*7)%32;
  jl_test_lookup(words,in);
  for(unsigned i=0;i<64;++i){
   float expected=-7.75f+.5f*in[i];uint32_t bits;std::memcpy(&bits,&expected,4);
   unsigned word=(i%2)*32+i/2;
   if(words[word]!=bits){std::printf("lookup lane failure %u\n",i);return 1;}
  }
 }
 for(unsigned start=0;start<65536;start+=64){
  for(unsigned i=0;i<64;++i)in[i]=start+i;
  jl_test_indices(out,in);
  for(unsigned i=0;i<64;++i)if((in[i]&0x7c00)!=0x7c00){
   int j=int(std::floor(2*half_value(in[i])))+16;j=j<0?0:j>31?31:j;
   if(out[i]!=j){std::printf("index failure %04x got%u expect%d\n",in[i],out[i],j);return 2;}
  }
 }
 return 0;
}
static unsigned test_pass(unsigned pass,FILE*dump=nullptr){
 alignas(128) uint16_t in[64],storage[192],saved[64];
 uint16_t*out=storage+64;unsigned bad=0;
 for(unsigned i=0;i<192;++i)storage[i]=0xa55a;
 const unsigned multiplier[]={1,40503,25173,10007,3,5,7,65521};
 for(unsigned start=0;start<65536;start+=64){
  for(unsigned i=0;i<64;++i)in[i]=uint16_t((start+i)*multiplier[pass%8]+pass*7919);
  std::memcpy(saved,in,128);jl_erf_poly64(out,in);
  if(std::memcmp(in,saved,128)){std::printf("input mutated\n");return 65536;}
  if(dump)std::fwrite(out,2,64,dump);
  for(unsigned i=0;i<64;++i){
   if(out[i]!=reference[in[i]]){if(bad<10)std::printf("BAD pass%u x%04x got%04x want%04x\n",pass,in[i],out[i],reference[in[i]]);++bad;}
   if(storage[i]!=0xa55a||storage[i+128]!=0xa55a){std::printf("canary failure\n");return 65536;}
  }
  jl_erf_poly64(in,in);
  for(unsigned i=0;i<64;++i)if(in[i]!=reference[saved[i]])++bad;
 }
 return bad;
}
int main(int argc,char**argv){
 if(argc!=3)return 2;
 FILE*f=std::fopen(argv[1],"rb");if(!f)return 3;
 if(std::fread(reference,2,65536,f)!=65536)return 4;std::fclose(f);
 if(mechanics())return 5;
 f=std::fopen(argv[2],"wb");if(!f)return 6;
 unsigned bad=test_pass(0,f);std::fclose(f);
 for(unsigned p=1;p<4;++p)bad+=test_pass(p);
#ifndef JL_NO_THREADS
 std::atomic<unsigned> parallel_bad{0};
 std::thread workers[4];
 for(unsigned p=0;p<4;++p)workers[p]=std::thread([&,p]{parallel_bad.fetch_add(test_pass(p+4));});
 for(auto&t:workers)t.join();bad+=parallel_bad.load();
 const unsigned comparisons=1048576;
#else
 const unsigned comparisons=524288;
#endif
 std::printf("%s %u raw comparisons, %u mismatches; exhaustive lookup/index, permutations, in-place and canaries. Functional execution, NOT device timing.\n",bad?"FAIL":"PASS",comparisons,bad);
 return bad?1:0;
}
