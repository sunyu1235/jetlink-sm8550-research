// Exact cached vision (HTP + CPU GELU) -> LiteRT GPU policy.
// State timing: both sessions read the same old frame. Only GPU owns desire/feat
// feedback; only vision owns image feedback. Vision's unused desire/feat states stay 0.
#include "jl_ort.h"
#include "jl_litert.h"
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

#define MAX_IO 256
extern char *jl_probe_shared_alloc(jl_session *, size_t, void **);
extern void jl_probe_shared_free(void *);
extern void jl_probe_shared_close(void);
extern char *jl_binding_sync_outputs(jl_binding *);

typedef struct {
  jl_session *session;
  jl_binding *binding[2];
  size_t count[2];
  jl_tensor tensors[2][MAX_IO];
  jl_tensor alternate[2][MAX_IO];
  char names[2][MAX_IO][256];
  int64_t dims[2][MAX_IO][16];
  int shared;
} stage;
static stage vision, policy;
static jl_env *environment;
static FILE *csv;
#define GPU_IO 16
static jl_litert_model *gpu_model;
static jl_litert_buffer *gpu_owned[GPU_IO*3];
static size_t gpu_owned_count;
static void *gpu_hosts[2][GPU_IO];
static int cleaned;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec+t.tv_nsec*1e-9; }
static long sensor(const char *path) { FILE *f=fopen(path,"r"); long v=-1; if(f){if(fscanf(f,"%ld",&v)!=1)v=-1;fclose(f);}return v; }
static void clean_stage(stage *s) {
  jl_binding_release(s->binding[0]); jl_binding_release(s->binding[1]);
  for(int o=0;o<2;o++) for(size_t i=0;i<s->count[o];i++) {
    if(s->shared) jl_probe_shared_free(s->tensors[o][i].data);
    else free(s->tensors[o][i].data);
  }
}
static void cleanup(void) {
  if(cleaned)return;
  cleaned=1;
  // Run while runtime DSOs still own their globals. An atexit handler registered
  // before dlopen runs after those DSOs' C++ destructors and cannot free ORT RPC
  // buffers safely (their allocator mutex has already been destroyed).
  fprintf(stderr,"CLEANUP_BEGIN\n");
  if(csv) { fclose(csv); csv=NULL; }
  fprintf(stderr,"CLEANUP_GPU_BUFFERS_BEGIN\n");
  for(size_t i=0;i<gpu_owned_count;i++)jl_litert_buffer_release(gpu_owned[i]);
  for(int o=0;o<2;o++)for(size_t i=0;i<GPU_IO;i++)free(gpu_hosts[o][i]);
  fprintf(stderr,"CLEANUP_GPU_MODEL_BEGIN\n");
  jl_litert_model_release(gpu_model);
  fprintf(stderr,"CLEANUP_HTP_BINDINGS_BUFFERS_BEGIN\n");
  clean_stage(&policy); clean_stage(&vision);
  fprintf(stderr,"CLEANUP_HTP_ALLOCATOR_BEGIN\n");
  jl_probe_shared_close();
  fprintf(stderr,"CLEANUP_HTP_SESSIONS_BEGIN\n");
  jl_session_release(policy.session); jl_session_release(vision.session);
  fprintf(stderr,"CLEANUP_ORT_ENV_BEGIN\n");
  jl_env_release(environment);
  fprintf(stderr,"CLEANUP_END\n");
}
static void fail(const char *where,const char *reason) { fprintf(stderr,"%s: %s\n",where,reason);cleanup();exit(1); }
static void check(char *e,const char *where) { if(e){fprintf(stderr,"%s: %s\n",where,e);jl_free(e);cleanup();exit(1);} }
static size_t width(int type) { return type==JL_FLOAT16?2:type==JL_FLOAT||type==JL_INT32?4:type==JL_INT64?8:type==JL_UINT8?1:0; }
static jl_tensor *tensor(stage *s,int output,int parity,const char *name) {
  jl_tensor *t=parity?s->alternate[output]:s->tensors[output];
  for(size_t i=0;i<s->count[output];i++) if(!strcmp(t[i].name,name))return &t[i];
  return NULL;
}
static void allocate(stage *s,const char *label) {
  for(int o=0;o<2;o++) {
    s->count[o]=jl_session_io_count(s->session,o);
    if(s->count[o]>MAX_IO)fail(label,"too many tensors");
    for(size_t i=0;i<s->count[o];i++) {
      int type;size_t rank;
      check(jl_session_io_info(s->session,o,i,s->names[o][i],256,&type,s->dims[o][i],16,&rank),"get IO");
      if(rank>16)fail(label,"rank > 16");
      size_t n=1, element=width(type);
      if(!element)fail(label,"unsupported tensor type");
      for(size_t j=0;j<rank;j++) { int64_t d=s->dims[o][i][j];if(d<1||n>SIZE_MAX/(size_t)d)fail(label,"invalid shape");n*=(size_t)d; }
      if(n>SIZE_MAX/element)fail(label,"tensor size overflow");
      size_t bytes=n*element;void *data=NULL;
      if(s->shared){check(jl_probe_shared_alloc(s->session,bytes,&data),"HTP shared allocation");memset(data,0,bytes);}
      else data=calloc(1,bytes);
      if(!data)fail(label,"allocation failed");
      s->tensors[o][i]=(jl_tensor){s->names[o][i],type,s->dims[o][i],rank,data,bytes};
      fprintf(stderr,"%s %s %s type=%d bytes=%zu shape=",label,o?"out":"in",s->names[o][i],type,bytes);
      for(size_t j=0;j<rank;j++)fprintf(stderr,"%s%lld",j?",":"",(long long)s->dims[o][i][j]);
      fputc('\n',stderr);
    }
    memcpy(s->alternate[o],s->tensors[o],s->count[o]*sizeof(jl_tensor));
  }
}
static void feedback(stage *s,const char *state) {
  char name[256];snprintf(name,sizeof(name),"next_%s",state);
  jl_tensor *in=tensor(s,0,0,state),*out=tensor(s,1,0,name);
  if(!in||!out||in->elem_type!=out->elem_type||in->nbytes!=out->nbytes)fail("feedback",state);
  tensor(s,0,1,state)->data=out->data;
  tensor(s,1,1,name)->data=in->data;
}
static void bind(stage *s,const char *name) {
  check(jl_binding_create(s->session,s->tensors[0],s->count[0],s->tensors[1],s->count[1],&s->binding[0]),name);
  check(jl_binding_create(s->session,s->alternate[0],s->count[0],s->alternate[1],s->count[1],&s->binding[1]),name);
}
static void convert(jl_tensor *dst,const jl_tensor *src) {
  if(!src||!dst)fail("convert","missing tensor");
  size_t n=src->nbytes/width(src->elem_type);
  if(n!=dst->nbytes/width(dst->elem_type))fail("convert","element count mismatch");
  if(src->elem_type==dst->elem_type){memcpy(dst->data,src->data,src->nbytes);return;}
  if(src->elem_type==JL_FLOAT16&&dst->elem_type==JL_FLOAT) {
    const __fp16 *a=src->data;float *b=dst->data;for(size_t i=0;i<n;i++)b[i]=(float)a[i];return;
  }
  if(src->elem_type==JL_FLOAT&&dst->elem_type==JL_FLOAT16) {
    const float *a=src->data;__fp16 *b=dst->data;for(size_t i=0;i<n;i++)b[i]=(__fp16)a[i];return;
  }
  fail("convert","unsupported types");
}
static void input_files(const char *dir,int frame,int parity) {
  for(size_t i=0;i<vision.count[0];i++) {
    jl_tensor *t=parity?&vision.alternate[0][i]:&vision.tensors[0][i];
    if(!strncmp(t->name,"state_",6))continue;
    char path[1024];int n=snprintf(path,sizeof(path),"%s/%d_%s.bin",dir,frame,t->name);
    if(n<0||(size_t)n>=sizeof(path))fail("input","path too long");
    FILE *f=fopen(path,"rb");if(!f)fail(path,strerror(errno));
    size_t got=fread(t->data,1,t->nbytes,f);int extra=fgetc(f);fclose(f);
    if(got!=t->nbytes||extra!=EOF)fail(path,"input file size mismatch");
  }
}
static size_t nonfinite(const jl_tensor *t) {
  size_t bad=0;
  if(t->elem_type==JL_FLOAT){const float *v=t->data;for(size_t i=0;i<t->nbytes/4;i++)if(!isfinite(v[i]))bad++;}
  else if(t->elem_type==JL_FLOAT16){const uint16_t *v=t->data;for(size_t i=0;i<t->nbytes/2;i++)if((v[i]&0x7c00)==0x7c00)bad++;}
  return bad;
}
static int compare_double(const void *a,const void *b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
static void gpu_check(char *error,const char *where) {
  if(error){fprintf(stderr,"%s: %s\n",where,error);jl_litert_free(error);cleanup();exit(1);}
}
typedef struct {
  char name[256]; int type; int64_t dims[16]; size_t rank,bytes;
  int pair;
  jl_litert_buffer *buffers[2];
} gpu_io;
static gpu_io gi[GPU_IO],go[GPU_IO];
static size_t gni,gno;
static void gpu_describe(int output,size_t index,gpu_io *io) {
  gpu_check(jl_litert_model_io_info(gpu_model,output,index,io->name,sizeof(io->name),&io->type,io->dims,16,&io->rank),"GPU IO");
  if(io->rank>16||!width(io->type))fail("GPU IO","unsupported type/rank");
  size_t n=1;
  for(size_t j=0;j<io->rank;j++){int64_t d=io->dims[j];if(d<1||n>SIZE_MAX/(size_t)d)fail("GPU IO","invalid shape");n*=(size_t)d;}
  if(n>SIZE_MAX/width(io->type))fail("GPU IO","size overflow");
  io->bytes=n*width(io->type);io->pair=-1;
  if((output&&!strcmp(io->name,"outputs"))||(!output&&strncmp(io->name,"state_",6))){
    gpu_hosts[output][index]=calloc(1,io->bytes);
    if(!gpu_hosts[output][index])fail("GPU IO","allocation failed");
  }
  fprintf(stderr,"GPU %s %s type=%d bytes=%zu shape=",output?"out":"in",io->name,io->type,io->bytes);
  for(size_t j=0;j<io->rank;j++)fprintf(stderr,"%s%lld",j?",":"",(long long)io->dims[j]);fputc('\n',stderr);
}
int main(int argc,char **argv) {
  if(argc!=9){fprintf(stderr,"usage: mixed_litert_probe vision.onnx policy.tflite nativeLibDir inputdir outdir frames csv cacheDir\n");return 2;}
  char *end=NULL;long frames=strtol(argv[6],&end,10);
  if(!end||end==argv[6]||*end||frames<1||frames>1200)fail("frames","expected 1..1200");
  if(mkdir(argv[5],0755)&&errno!=EEXIST)fail(argv[5],strerror(errno));
  if(mkdir(argv[8],0755)&&errno!=EEXIST)fail(argv[8],strerror(errno));
  check(jl_env_create(2,&environment),"environment");fprintf(stderr,"ORT %s\n",jl_version());
  jl_option htp[]={{"backend_type","htp"},{"htp_performance_mode","burst"},{"enable_htp_fp16_precision","1"},{"qnn_context_priority","high"},{"enable_htp_shared_memory_allocator","1"},{"rpc_control_latency","0"},{"htp_graph_finalization_optimization_mode","3"},{"soc_model","43"}};
  jl_option vcfg[]={{"session.disable_cpu_ep_fallback","0"}};
  fprintf(stderr,"VISION_SESSION_CREATE_BEGIN\n");double start=now();alarm(120);
  check(jl_session_create(environment,argv[1],"QNN",htp,sizeof(htp)/sizeof(htp[0]),vcfg,1,4,&vision.session),"vision session");
  alarm(0);fprintf(stderr,"VISION_SESSION_CREATE_END seconds=%.3f\n",now()-start);
  vision.shared=1;allocate(&vision,"vision");feedback(&vision,"state_img_q");bind(&vision,"vision binding");
  gpu_check(jl_litert_open(argv[3]),"LiteRT open");
  const char *key=getenv("JL_LITERT_CACHE_KEY");
  jl_litert_options options={.gpu=1,.cpu_threads=4,.cache_dir=argv[8],.cache_key=key?key:"mixed_policy_sqrt_eps_20261109",.npu_cache_dir=NULL};
  fprintf(stderr,"GPU_POLICY_SESSION_CREATE_BEGIN\n");start=now();alarm(120);
  gpu_check(jl_litert_model_create(argv[2],&options,&gpu_model),"GPU policy model");
  alarm(0);fprintf(stderr,"GPU_POLICY_SESSION_CREATE_END seconds=%.3f\n",now()-start);
  int fully=0;gpu_check(jl_litert_model_fully_accelerated(gpu_model,&fully),"GPU acceleration");
  if(!fully)fail("GPU policy","refusing partial acceleration");
  fprintf(stderr,"GPU fully_accelerated=1; managed state pingpong; no GPU state readback\n");
  gni=jl_litert_model_io_count(gpu_model,0);gno=jl_litert_model_io_count(gpu_model,1);
  if(!gni||!gno||gni>GPU_IO||gno>GPU_IO)fail("GPU IO","invalid count");
  int feature=-1,main_output=-1;
  for(size_t i=0;i<gni;i++){gpu_describe(0,i,&gi[i]);if(!strcmp(gi[i].name,"view_39"))feature=(int)i;}
  for(size_t i=0;i<gno;i++){gpu_describe(1,i,&go[i]);if(!strcmp(go[i].name,"outputs"))main_output=(int)i;}
  jl_tensor *vf=tensor(&vision,1,0,"view_70");
  if(!vf||vf->elem_type!=JL_FLOAT16||vf->nbytes!=32768||feature<0||gi[feature].type!=JL_FLOAT||gi[feature].bytes!=65536)fail("feature seam","unexpected view_70/view_39");
  if(main_output<0||go[main_output].type!=JL_FLOAT||go[main_output].bytes!=18452*4)fail("GPU output","expected outputs float[18452]");
  for(size_t i=0;i<gni;i++) {
    gpu_check(jl_litert_buffer_create(gpu_model,0,i,&gi[i].buffers[0]),"GPU input buffer");gpu_owned[gpu_owned_count++]=gi[i].buffers[0];
    if(!strncmp(gi[i].name,"state_",6)) {
      for(size_t j=0;j<gno;j++)if(!strncmp(go[j].name,"next_",5)&&!strcmp(go[j].name+5,gi[i].name)) {
        if(gi[i].type!=go[j].type||gi[i].bytes!=go[j].bytes||gi[i].rank!=go[j].rank||memcmp(gi[i].dims,go[j].dims,gi[i].rank*sizeof(int64_t)))fail("GPU state","metadata mismatch");
        gi[i].pair=(int)j;go[j].pair=(int)i;break;
      }
      if(gi[i].pair<0)fail("GPU state","missing state output");
      gpu_check(jl_litert_buffer_create(gpu_model,0,i,&gi[i].buffers[1]),"GPU alternate state buffer");gpu_owned[gpu_owned_count++]=gi[i].buffers[1];
    } else gi[i].buffers[1]=gi[i].buffers[0];
  }
  for(size_t j=0;j<gno;j++) {
    if(go[j].pair>=0){go[j].buffers[0]=gi[go[j].pair].buffers[1];go[j].buffers[1]=gi[go[j].pair].buffers[0];}
    else {gpu_check(jl_litert_buffer_create(gpu_model,1,j,&go[j].buffers[0]),"GPU output buffer");gpu_owned[gpu_owned_count++]=go[j].buffers[0];go[j].buffers[1]=go[j].buffers[0];}
  }
  fprintf(stderr,"vision state_feat_q/state_desire_q remain zero: feature/image do not depend on the prefix's unused policy branch\n");
  csv=fopen(argv[7],"w");if(!csv)fail(argv[7],strerror(errno));
  fprintf(csv,"frame,elapsed_s,vision_ms,bridge_ms,policy_ms,gpu_run_ms,gpu_read_ms,execute_ms,battery_deci_c,gpu_milli_c,gpu_clock_hz,nonfinite\n");fflush(csv);
  double times[1200],total=0,vs=0,bs=0,ps=0,beginning=now();
  for(int f=0;f<frames;f++) {
    int phase=f&1;input_files(argv[4],f,phase);
    jl_litert_buffer *inputs[GPU_IO],*outputs[GPU_IO];
    for(size_t i=0;i<gni;i++)inputs[i]=gi[i].buffers[phase];
    for(size_t j=0;j<gno;j++)outputs[j]=go[j].buffers[phase];
    double a=now();check(jl_binding_run(vision.binding[phase]),"vision run");check(jl_binding_sync_outputs(vision.binding[phase]),"vision synchronize");double b=now();
    for(size_t i=0;i<gni;i++)if(strncmp(gi[i].name,"state_",6)) {
      jl_tensor destination={gi[i].name,gi[i].type,gi[i].dims,gi[i].rank,gpu_hosts[0][i],gi[i].bytes};
      jl_tensor *source=(int)i==feature?tensor(&vision,1,phase,"view_70"):tensor(&vision,0,phase,gi[i].name);
      convert(&destination,source);gpu_check(jl_litert_buffer_write(inputs[i],destination.data,destination.nbytes),"GPU input write");
    }
    double c=now();gpu_check(jl_litert_run(gpu_model,inputs,gni,outputs,gno),"GPU policy run");double d=now();
    gpu_check(jl_litert_buffer_read(outputs[main_output],gpu_hosts[1][main_output],go[main_output].bytes),"GPU output read");double e=now();
    double vm=(b-a)*1000,bm=(c-b)*1000,pm=(e-c)*1000,ms=(e-a)*1000;
    times[f]=ms;total+=ms;vs+=vm;bs+=bm;ps+=pm;
    jl_tensor main={go[main_output].name,go[main_output].type,go[main_output].dims,go[main_output].rank,gpu_hosts[1][main_output],go[main_output].bytes};
    size_t bad=nonfinite(&main)+nonfinite(tensor(&vision,1,phase,"view_70"));
    fprintf(csv,"%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%ld,%ld,%ld,%zu\n",f,now()-beginning,vm,bm,pm,(d-c)*1000,(e-d)*1000,ms,sensor("/sys/class/power_supply/battery/temp"),sensor("/sys/class/kgsl/kgsl-3d0/temp"),sensor("/sys/class/kgsl/kgsl-3d0/gpuclk"),bad);fflush(csv);
    char path[1024];int n=snprintf(path,sizeof(path),"%s/out_%d.bin",argv[5],f);if(n<0||(size_t)n>=sizeof(path))fail("output","path too long");
    FILE *fp=fopen(path,"wb");if(!fp)fail(path,strerror(errno));size_t written=fwrite(main.data,1,main.nbytes,fp);int closed=fclose(fp);
    if(written!=main.nbytes||closed)fail(path,"output write failed");
    fprintf(stderr,"frame=%d total_ms=%.3f vision_ms=%.3f bridge_ms=%.3f policy_ms=%.3f run_ms=%.3f read_ms=%.3f nonfinite=%zu\n",f,ms,vm,bm,pm,(d-c)*1000,(e-d)*1000,bad);
    if(bad)fail("mixed frame","nonfinite output/feature; stopped before feedback");
  }
  qsort(times,(size_t)frames,sizeof(double),compare_double);
  printf("mixed_litert n=%ld mean_ms=%.6f p50_ms=%.6f p99_ms=%.6f vision_mean_ms=%.6f bridge_mean_ms=%.6f policy_mean_ms=%.6f fully_accelerated=1\n",frames,total/frames,times[(frames-1)/2],times[(frames-1)*99/100],vs/frames,bs/frames,ps/frames);
  fflush(stdout);
  cleanup();
  return 0;
}
