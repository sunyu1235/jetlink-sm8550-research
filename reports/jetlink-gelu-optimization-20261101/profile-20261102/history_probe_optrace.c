// Backend diagnostic runner. Variable inputs are mandatory and size checked.
// Accelerator sessions forbid silent CPU fallback; captures are not warmed up.
#include "jl_ort.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
static long readval(const char *path){FILE *f=fopen(path,"r");long v=-1;if(f){if(fscanf(f,"%ld",&v)!=1)v=-1;fclose(f);}return v;}
static double tm(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int cmp_double(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static void ck(char *e,const char *where){if(e){fprintf(stderr,"%s: %s\n",where,e);jl_free(e);exit(1);}}
extern char *jl_probe_shared_alloc(jl_session *, size_t, void **);
extern void jl_probe_shared_free(void *);
extern void jl_probe_shared_close(void);
int main(int argc,char **argv){
 if(argc<2){fprintf(stderr,"usage: probe context.onnx [shared 0|1] [rpc latency us]\n");return 1;}
 jl_option opt[9]={{"backend_type","htp"},{"htp_performance_mode","burst"},{"enable_htp_fp16_precision","1"},{"qnn_context_priority","high"},{"enable_htp_shared_memory_allocator",argc>2?argv[2]:"0"},{"rpc_control_latency",argc>3?argv[3]:"0"},{"htp_graph_finalization_optimization_mode","3"},{"profiling_level",argc>4?argv[4]:"basic"},{"profiling_file_path","/data/local/tmp/jetlink_qnn_optrace_20261102.csv"}};
 int cpu=argc>9 && !strcmp(argv[9],"cpu");if(argc>9 && !strcmp(argv[9],"gpu"))opt[0].value="gpu";
 const char *ctx_path=getenv("JL_CONTEXT_OUTPUT");jl_option cfg[4]={{"session.disable_cpu_ep_fallback",getenv("JL_CUSTOM_OP_LIBRARY")?"0":"1"},{"ep.context_enable","1"},{"ep.context_file_path",ctx_path?ctx_path:""},{"ep.context_embed_mode","0"}};jl_env *env=0;jl_session *session=0;jl_binding *binding[2]={0};ck(jl_env_create(2,&env),"env");fprintf(stderr,"ORT %s\n",jl_version());
 ck(jl_session_create(env,argv[1],cpu?NULL:"QNN",opt,cpu?0:((argc>4 && strcmp(argv[4],"none"))?9:7),cpu?NULL:cfg,cpu?0:(ctx_path?4:1),(cpu||getenv("JL_CUSTOM_OP_LIBRARY"))?4:1,&session),"session");
 size_t ni=jl_session_io_count(session,0),no=jl_session_io_count(session,1);if(ni>32||no>32)return 2;
 jl_tensor in[32]={0},out[32]={0};char names[2][32][256];int64_t dims[2][32][16];
 for(int o=0;o<2;o++){size_t count=o?no:ni;for(size_t i=0;i<count;i++){
  int typ;size_t rank;ck(jl_session_io_info(session,o,i,names[o][i],256,&typ,dims[o][i],16,&rank),"io");if(rank>16)return 2;
  size_t num=1;for(size_t j=0;j<rank;j++){if(dims[o][i][j]<1)return 2;num*=dims[o][i][j];}
  size_t elt=typ==10?2:typ==1?4:typ==7?8:typ==2?1:0;if(!elt){fprintf(stderr,"unsupported dtype %d\n",typ);return 2;}
  jl_tensor *t=o?&out[i]:&in[i];void *data=NULL;size_t bytes=num*elt;
  if(!strcmp(opt[4].value,"1")){ck(jl_probe_shared_alloc(session,bytes,&data),"shared alloc");memset(data,0,bytes);}else data=calloc(num,elt);
  *t=(jl_tensor){names[o][i],typ,dims[o][i],rank,data,bytes};if(!t->data)return 2;
  fprintf(stderr,"%s %s %zu bytes\n",o?"out":"in",t->name,t->nbytes);
 }}
 if(argc>6 && argc<=8){for(size_t i=0;i<ni;i++){
  if(!strcmp(in[i].name,"new_img")){FILE *f=fopen(argv[6],"rb");if(!f || fread(in[i].data,1,in[i].nbytes,f)!=in[i].nbytes){fprintf(stderr,"image load failed\n");return 2;}fclose(f);}
  else if(!strcmp(in[i].name,"traffic_convention") && in[i].elem_type==JL_FLOAT)((float*)in[i].data)[0]=1.0f;
  else if(!strcmp(in[i].name,"action_t") && in[i].elem_type==JL_FLOAT){((float*)in[i].data)[0]=0.25f;((float*)in[i].data)[1]=0.45f;}
 }}
 jl_tensor in_alt[32],out_alt[32];memcpy(in_alt,in,sizeof(in));memcpy(out_alt,out,sizeof(out));
 for(size_t i=0;i<ni;i++)for(size_t j=0;j<no;j++)if(!strncmp(in[i].name,"state_",6) && !strncmp(out[j].name,"next_",5) && !strcmp(in[i].name,out[j].name+5) && in[i].nbytes==out[j].nbytes){in_alt[i].data=out[j].data;out_alt[j].data=in[i].data;}
 ck(jl_binding_create(session,in,ni,out,no,&binding[0]),"bind 0");ck(jl_binding_create(session,in_alt,ni,out_alt,no,&binding[1]),"bind 1");int parity=0;if(argc<=7)for(int i=0;i<3;i++){ck(jl_binding_run(binding[parity]),"warmup");parity^=1;}
 int runs=argc>5?atoi(argv[5]):40;if(runs<1||runs>1200)return 2;double all=0,min=10000,max=0,values[1200];int over50=0;FILE *trace=NULL;const char *tracepath=getenv("JL_FRAME_CSV");int modulo=getenv("JL_INPUT_MODULO")?atoi(getenv("JL_INPUT_MODULO")):0;if(tracepath){trace=fopen(tracepath,"w");if(!trace)return 6;fprintf(trace,"frame,elapsed_s,execute_ms,battery_deci_c,gpu_milli_c,gpu_clock_hz,nonfinite\n");}double started=tm();for(int i=0;i<runs;i++){if(argc>8){for(size_t j=0;j<ni;j++){char path[512];snprintf(path,sizeof(path),"%s/%d_%s.bin",argv[6],modulo>0?i%modulo:i,in[j].name);FILE *f=fopen(path,"rb");if(!f){if(strncmp(in[j].name,"state_",6)){fprintf(stderr,"missing required input: %s\n",path);return 4;}}else{size_t got=fread(parity?in_alt[j].data:in[j].data,1,in[j].nbytes,f);int extra=fgetc(f);fclose(f);if(got!=in[j].nbytes||extra!=EOF){fprintf(stderr,"input size mismatch: %s expected=%zu got=%zu extra=%d\n",path,in[j].nbytes,got,extra);return 4;}}}}double a=tm();ck(jl_binding_run(binding[parity]),"run");parity^=1;double x=(tm()-a)*1000;values[i]=x;all+=x;if(x<min)min=x;if(x>max)max=x;if(x>50)over50++;if(argc>7){for(size_t j=0;j<no;j++)if(!strcmp(out[j].name,"outputs") || !strcmp(out[j].name,"view_39")){char path[512];snprintf(path,sizeof(path),"%s/out_%d.bin",argv[7],i);FILE *f=fopen(path,"wb");if(!f||fwrite(out[j].data,1,out[j].nbytes,f)!=out[j].nbytes)return 3;fclose(f);}}if(trace){size_t bad=0;for(size_t j=0;j<no;j++)if(!strcmp(out[j].name,"outputs") && out[j].elem_type==JL_FLOAT){float *data=out[j].data;for(size_t k=0;k<out[j].nbytes/4;k++)if(!isfinite(data[k]))bad++;}fprintf(trace,"%d,%.6f,%.6f,%ld,%ld,%ld,%zu\n",i,tm()-started,x,readval("/sys/class/power_supply/battery/temp"),readval("/sys/class/kgsl/kgsl-3d0/temp"),readval("/sys/class/kgsl/kgsl-3d0/gpuclk"),bad);fflush(trace);if(bad){fclose(trace);fprintf(stderr,"nonfinite at frame %d\n",i);return 5;}}if(x<50)usleep((useconds_t)((50-x)*1000));}if(trace)fclose(trace);
 if(argc>7)for(size_t i=0;i<no;i++)if(!strncmp(out[i].name,"layer_norm",10)){char path[512];snprintf(path,sizeof(path),"%s/%s.bin",argv[7],out[i].name);FILE *f=fopen(path,"wb");if(!f||fwrite(out[i].data,1,out[i].nbytes,f)!=out[i].nbytes)return 3;fclose(f);}
 for(size_t i=0;i<no;i++)if(!strcmp(out[i].name,"outputs")){size_t bad=0,n=0;if(out[i].elem_type==JL_FLOAT16){uint16_t *v=out[i].data;n=out[i].nbytes/2;for(size_t j=0;j<n;j++)if((v[j]&0x7c00)==0x7c00)bad++;}else if(out[i].elem_type==JL_FLOAT){float *v=out[i].data;n=out[i].nbytes/4;for(size_t j=0;j<n;j++)if(!isfinite(v[j]))bad++;}printf("output_type=%d nonfinite=%zu/%zu\n",out[i].elem_type,bad,n);}
 qsort(values,runs,sizeof(double),cmp_double);printf("shared=%s rpc_us=%s n=%d mean_ms=%.3f p50_ms=%.3f p99_ms=%.3f min_ms=%.3f max_ms=%.3f over50=%d\n",opt[4].value,opt[5].value,runs,all/runs,values[(runs-1)/2],values[(runs-1)*99/100],min,max,over50);
 jl_binding_release(binding[0]);jl_binding_release(binding[1]);if(!strcmp(opt[4].value,"1")){for(size_t i=0;i<ni;i++)jl_probe_shared_free(in[i].data);for(size_t i=0;i<no;i++)jl_probe_shared_free(out[i].data);jl_probe_shared_close();}jl_session_release(session);jl_env_release(env);return 0;
}
