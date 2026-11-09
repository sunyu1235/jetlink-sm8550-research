// Exact cached vision (HTP + CPU GELU) -> FP32 QNN GPU policy.
// State timing: both sessions read the same old frame. Only GPU owns desire/feat
// feedback; only vision owns image feedback. Vision's unused feat state stays 0.
#include "jl_ort.h"
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
  if(csv) { fclose(csv); csv=NULL; }
  clean_stage(&policy); clean_stage(&vision);
  jl_probe_shared_close();
  jl_session_release(policy.session); jl_session_release(vision.session);
  jl_env_release(environment);
}
static void fail(const char *where,const char *reason) { fprintf(stderr,"%s: %s\n",where,reason);exit(1); }
static void check(char *e,const char *where) { if(e){fprintf(stderr,"%s: %s\n",where,e);jl_free(e);exit(1);} }
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
int main(int argc,char **argv) {
  if(argc!=7){fprintf(stderr,"usage: mixed_probe vision.onnx policy.onnx inputdir outdir frames csv\n");return 2;}
  char *end=NULL;long frames=strtol(argv[5],&end,10);
  if(!end||*end||frames<1||frames>1200)fail("frames","expected 1..1200");
  atexit(cleanup);
  check(jl_env_create(2,&environment),"environment");fprintf(stderr,"ORT %s\n",jl_version());
  jl_option htp[]={{"backend_type","htp"},{"htp_performance_mode","burst"},{"enable_htp_fp16_precision","1"},{"qnn_context_priority","high"},{"enable_htp_shared_memory_allocator","1"},{"rpc_control_latency","0"},{"htp_graph_finalization_optimization_mode","3"},{"soc_model","43"}};
  jl_option vcfg[]={{"session.disable_cpu_ep_fallback","0"}};
  fprintf(stderr,"VISION_SESSION_CREATE_BEGIN\n");double start=now();alarm(120);
  check(jl_session_create(environment,argv[1],"QNN",htp,sizeof(htp)/sizeof(htp[0]),vcfg,1,4,&vision.session),"vision session");
  alarm(0);fprintf(stderr,"VISION_SESSION_CREATE_END seconds=%.3f\n",now()-start);
  jl_option gpu[]={{"backend_type","gpu"}};
  // QNN GPU does not accept the fused Gemm forms produced from ten linears.
  // Keep the supported MatMul+Add and explicit tanh formula intact.
  jl_option pcfg[]={{"session.disable_cpu_ep_fallback","1"},
    {"optimization.disable_specified_optimizers","MatMulAddFusion,GeluFusion,FastGeluFusion"}};
  fprintf(stderr,"GPU_POLICY_SESSION_CREATE_BEGIN\n");start=now();alarm(120);
  check(jl_session_create(environment,argv[2],"QNN",gpu,sizeof(gpu)/sizeof(gpu[0]),pcfg,sizeof(pcfg)/sizeof(pcfg[0]),4,&policy.session),"GPU policy session");
  alarm(0);fprintf(stderr,"GPU_POLICY_SESSION_CREATE_END seconds=%.3f\n",now()-start);
  vision.shared=1;allocate(&vision,"vision");allocate(&policy,"policy");
  feedback(&vision,"state_img_q");feedback(&policy,"state_desire_q");feedback(&policy,"state_feat_q");
  jl_tensor *vf=tensor(&vision,1,0,"view_70"),*pf=tensor(&policy,0,0,"view_39");
  if(!vf||vf->elem_type!=JL_FLOAT16||vf->nbytes!=32768||!pf||pf->elem_type!=JL_FLOAT||pf->nbytes!=65536)fail("feature seam","unexpected view_70/view_39");
  bind(&vision,"vision binding");bind(&policy,"GPU policy binding");
  fprintf(stderr,"vision state_feat_q intentionally remains zero: prefix feature/image independent of policy history; GPU owns real feat history\n");
  csv=fopen(argv[6],"w");if(!csv)fail(argv[6],strerror(errno));
  fprintf(csv,"frame,elapsed_s,vision_ms,bridge_ms,policy_ms,execute_ms,battery_deci_c,gpu_milli_c,gpu_clock_hz,nonfinite,state_nonfinite\n");fflush(csv);
  double times[1200],total=0,vs=0,bs=0,ps=0;double beginning=now();
  for(int f=0;f<frames;f++) {
    int p=f&1; input_files(argv[3],f,p);
    double a=now();
    // Small old desire history is mirrored into the unused prefix policy branch.
    convert(tensor(&vision,0,p,"state_desire_q"),tensor(&policy,0,p,"state_desire_q"));
    const char *external[]={"desire","traffic_convention","action_t"};
    for(size_t i=0;i<3;i++)convert(tensor(&policy,0,p,external[i]),tensor(&vision,0,p,external[i]));
    double pre=now();
    check(jl_binding_run(vision.binding[p]),"vision run");
    check(jl_binding_sync_outputs(vision.binding[p]),"vision synchronize");
    double b=now();convert(tensor(&policy,0,p,"view_39"),tensor(&vision,1,p,"view_70"));double c=now();
    check(jl_binding_run(policy.binding[p]),"GPU policy run");
    check(jl_binding_sync_outputs(policy.binding[p]),"GPU policy synchronize");
    double d=now();
    double vm=(b-pre)*1000,bm=((pre-a)+(c-b))*1000,pm=(d-c)*1000,ms=(d-a)*1000;
    times[f]=ms;total+=ms;vs+=vm;bs+=bm;ps+=pm;
    jl_tensor *out=tensor(&policy,1,p,"outputs");if(!out)fail("GPU policy","missing outputs");
    size_t bad=nonfinite(out),state_bad=nonfinite(tensor(&policy,1,p,"next_state_feat_q"))+nonfinite(tensor(&policy,1,p,"next_state_desire_q"))+nonfinite(tensor(&vision,1,p,"view_70"));
    fprintf(csv,"%d,%.6f,%.6f,%.6f,%.6f,%.6f,%ld,%ld,%ld,%zu,%zu\n",f,now()-beginning,vm,bm,pm,ms,sensor("/sys/class/power_supply/battery/temp"),sensor("/sys/class/kgsl/kgsl-3d0/temp"),sensor("/sys/class/kgsl/kgsl-3d0/gpuclk"),bad,state_bad);fflush(csv);
    char path[1024];int n=snprintf(path,sizeof(path),"%s/out_%d.bin",argv[4],f);
    if(n<0||(size_t)n>=sizeof(path))fail("output","path too long");
    FILE *fp=fopen(path,"wb");if(!fp)fail(path,strerror(errno));size_t written=fwrite(out->data,1,out->nbytes,fp);int closed=fclose(fp);
    if(written!=out->nbytes||closed)fail(path,"output write failed");
    if(bad||state_bad)fail("mixed frame","nonfinite output/state; stopped before feedback");
    if(f==0||f%16==15)fprintf(stderr,"frame=%d total_ms=%.3f vision_ms=%.3f bridge_ms=%.3f policy_ms=%.3f\n",f,ms,vm,bm,pm);
  }
  qsort(times,(size_t)frames,sizeof(double),compare_double);
  printf("mixed n=%ld mean_ms=%.6f p50_ms=%.6f p99_ms=%.6f vision_mean_ms=%.6f bridge_mean_ms=%.6f policy_mean_ms=%.6f\n",frames,total/frames,times[(frames-1)/2],times[(frames-1)*99/100],vs/frames,bs/frames,ps/frames);
  return 0;
}
