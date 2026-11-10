// Actual QNN host-functional graph tests. No hardware timing or device access.
#include <dlfcn.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "QnnInterface.h"
#include "HTP/QnnHtpDevice.h"
static QNN_INTERFACE_VER_TYPE api;
static Qnn_DeviceHandle_t device=NULL;
static uint16_t reference[2][65536];
#define STEP(label,call) do { Qnn_ErrorHandle_t rc=(call); if(rc!=QNN_SUCCESS){printf("FAIL %s=%llu\n",label,(unsigned long long)rc);goto cleanup;} } while(0)
static int run_case(Qnn_BackendHandle_t backend,unsigned mode,uint32_t dims[4]) {
 int passed=0;void *cached=NULL;Qnn_ContextHandle_t context=NULL;Qnn_GraphHandle_t graph=NULL;
 size_t n=(size_t)dims[0]*dims[1]*dims[2]*dims[3];
 uint16_t *input=malloc((n+2)*2),*output=malloc((n+2)*2);
 if(!input||!output){free(input);free(output);return 0;}
 input[0]=output[0]=0x1234;input[n+1]=output[n+1]=0xabcd;
 STEP("contextCreate",api.contextCreate(backend,device,NULL,&context));
 STEP("graphCreate",api.graphCreate(context,"gelu_test",NULL,&graph));
 Qnn_Tensor_t x=QNN_TENSOR_INIT,y=QNN_TENSOR_INIT;
 x.v1.name="input";x.v1.type=QNN_TENSOR_TYPE_APP_WRITE;x.v1.dataType=QNN_DATATYPE_FLOAT_16;x.v1.rank=4;x.v1.dimensions=dims;x.v1.memType=QNN_TENSORMEMTYPE_RAW;
 y.v1.name="output";y.v1.type=QNN_TENSOR_TYPE_APP_READ;y.v1.dataType=QNN_DATATYPE_FLOAT_16;y.v1.rank=4;y.v1.dimensions=dims;y.v1.memType=QNN_TENSORMEMTYPE_RAW;
 STEP("tensorInput",api.tensorCreateGraphTensor(graph,&x));STEP("tensorOutput",api.tensorCreateGraphTensor(graph,&y));
 Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;op.v1.name="gelu";op.v1.packageName="jetlink_gelu";op.v1.typeName=mode?"GeluTanhFp16":"GeluErfFp16";op.v1.numOfInputs=1;op.v1.inputTensors=&x;op.v1.numOfOutputs=1;op.v1.outputTensors=&y;
 STEP("graphAddNode",api.graphAddNode(graph,op));STEP("graphFinalize",api.graphFinalize(graph,NULL,NULL));
 x.v1.clientBuf.data=input+1;x.v1.clientBuf.dataSize=(uint32_t)n*2;y.v1.clientBuf.data=output+1;y.v1.clientBuf.dataSize=(uint32_t)n*2;
 size_t mismatches=0,finite_mismatches=0;
 for(unsigned round=0;round<2;++round){
  if(round==1){
   Qnn_ContextBinarySize_t size=0,written=0;
   STEP("contextGetBinarySize",api.contextGetBinarySize(context,&size));
   cached=malloc((size_t)size);if(!cached)goto cleanup;
   STEP("contextGetBinary",api.contextGetBinary(context,cached,size,&written));
   STEP("contextFreeBeforeReload",api.contextFree(context,NULL));context=NULL;graph=NULL;
   STEP("contextCreateFromBinary",api.contextCreateFromBinary(backend,device,NULL,cached,written,&context,NULL));
   STEP("graphRetrieve",api.graphRetrieve(context,"gelu_test",&graph));
  }
  for(size_t i=0;i<n;++i){input[i+1]=(uint16_t)(n==65536?(round?65535-i:i):(i*997+round*30011));output[i+1]=0xbeef;}
  STEP("graphExecute",api.graphExecute(graph,&x,1,&y,1,NULL,NULL));
  for(size_t i=0;i<n;++i)if(output[i+1]!=reference[mode][input[i+1]]){++mismatches;if((input[i+1]&0x7c00)!=0x7c00)++finite_mismatches;}
  if(input[0]!=0x1234||output[0]!=0x1234||input[n+1]!=0xabcd||output[n+1]!=0xabcd){puts("FAIL canary");goto cleanup;}
 }
 printf("CASE mode=%s shape=%u,%u,%u,%u executions=2 serialized_reload=1 comparisons=%zu mismatches=%zu finite_mismatches=%zu\n",mode?"tanh":"erf",dims[0],dims[1],dims[2],dims[3],n*2,mismatches,finite_mismatches);
 passed=mismatches==0;
 cleanup:
 if(context && api.contextFree(context,NULL)!=QNN_SUCCESS)passed=0;
 free(cached);free(input);free(output);return passed;
}
static int invalid_cases(Qnn_BackendHandle_t backend) {
 uint32_t dims[4]={1,1,1,64},bad_dims[4]={1,1,1,65};
 Qnn_Tensor_t x=QNN_TENSOR_INIT,y=QNN_TENSOR_INIT;
 x.v1.name="x";x.v1.dataType=QNN_DATATYPE_FLOAT_16;x.v1.rank=4;x.v1.dimensions=dims;x.v1.type=QNN_TENSOR_TYPE_APP_WRITE;
 y.v1.name="y";y.v1.dataType=QNN_DATATYPE_FLOAT_16;y.v1.rank=4;y.v1.dimensions=dims;y.v1.type=QNN_TENSOR_TYPE_APP_READ;
 Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;op.v1.name="validation";op.v1.packageName="jetlink_gelu";op.v1.typeName="GeluErfFp16";op.v1.numOfInputs=1;op.v1.inputTensors=&x;op.v1.numOfOutputs=1;op.v1.outputTensors=&y;
 if(api.backendValidateOpConfig(backend,op)!=QNN_SUCCESS){puts("FAIL valid_config");return 0;}
 unsigned rejected=0;
 for(unsigned test=0;test<6;test++){
  const char *saved=op.v1.typeName;
  if(test==0)op.v1.typeName="UnknownGelu";
  if(test==1)x.v1.dataType=QNN_DATATYPE_FLOAT_32;
  if(test==2)x.v1.rank=3;
  if(test==3)y.v1.dimensions=bad_dims;
  if(test==4)op.v1.numOfInputs=0;
  if(test==5)dims[3]=0;
  Qnn_ErrorHandle_t rc=api.backendValidateOpConfig(backend,op);
  if(rc==QNN_SUCCESS){printf("FAIL invalid_config_accepted=%u\n",test);return 0;}
  rejected++;
  op.v1.typeName=saved;x.v1.dataType=QNN_DATATYPE_FLOAT_16;x.v1.rank=4;y.v1.dimensions=dims;op.v1.numOfInputs=1;dims[3]=64;
 }
 printf("CONFIG rejected=%u expected=6\n",rejected);return 1;
}
int main(int argc,char **argv){
 if(argc!=4){fprintf(stderr,"usage: test_qnn_gelu BACKEND PACKAGE REFERENCE\n");return 1;}
 FILE *f=fopen(argv[3],"rb");if(!f)return 1;
 for(unsigned mode=0;mode<2;++mode)for(unsigned i=0;i<65536;++i){unsigned char b[2];if(fread(b,1,2,f)!=2)return 1;reference[mode][i]=(uint16_t)(b[0]|((unsigned)b[1]<<8));}
 if(fgetc(f)!=EOF)return 1;
 fclose(f);
 void *lib=dlopen(argv[1],RTLD_NOW|RTLD_GLOBAL);if(!lib){puts(dlerror());return 1;}
 Qnn_ErrorHandle_t(*get)(const QnnInterface_t***,uint32_t*)=dlsym(lib,"QnnInterface_getProviders");if(!get)return 1;
 const QnnInterface_t **providers=NULL;uint32_t count=0;if(get(&providers,&count)||!count)return 1;
 api=providers[0]->QNN_INTERFACE_VER_NAME;
 Qnn_BackendHandle_t backend=NULL;int result=2;unsigned tests=0;
 STEP("backendCreate",api.backendCreate(NULL,NULL,&backend));
 STEP("registerOpPackage",api.backendRegisterOpPackage(backend,argv[2],"exampleInterfaceProvider","CPU"));
 QnnHtpDevice_CustomConfig_t arch={0},soc={0};
 arch.option=QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;arch.arch.deviceId=0;arch.arch.arch=QNN_HTP_DEVICE_ARCH_V73;
 soc.option=QNN_HTP_DEVICE_CONFIG_OPTION_SOC;soc.socModel=43;
 QnnDevice_Config_t ac=QNN_DEVICE_CONFIG_INIT,sc=QNN_DEVICE_CONFIG_INIT;
 ac.option=QNN_DEVICE_CONFIG_OPTION_CUSTOM;ac.customConfig=&arch;
 sc.option=QNN_DEVICE_CONFIG_OPTION_CUSTOM;sc.customConfig=&soc;
 const QnnDevice_Config_t *dc[]={&ac,&sc,NULL};
 STEP("deviceCreateV73SoC43",api.deviceCreate(NULL,dc,&device));
 puts("CONFIG offline_min_arch=73 soc=43");
 if(!invalid_cases(backend))goto cleanup;
 uint32_t shapes[][4]={{1,1,1,1},{1,1,1,63},{1,1,1,64},{1,1,1,65},{1,1,1,127},{1,1,1,128},{1,1,1,129},{1,1,1,65536},{2,4,16,512},{1,3,5,7},{1,4,16,3072},{1,32,64,1024}};
 for(unsigned mode=0;mode<2;++mode)for(unsigned i=0;i<sizeof(shapes)/sizeof(shapes[0]);++i){if(!run_case(backend,mode,shapes[i]))goto cleanup;++tests;}
 result=0;
 {void *package=dlopen(argv[2],RTLD_NOW);if(!package){result=2;goto cleanup;}
  unsigned long(*count_vectors)(void)=dlsym(package,"jl_polynomial_vectors");
  unsigned long vectors=count_vectors?count_vectors():0;
  printf("POLYNOMIAL vectors=%lu\n",vectors);if(!vectors)result=2;dlclose(package);}
 cleanup:
 if(device && api.deviceFree(device)!=QNN_SUCCESS)result=2;
 if(backend && api.backendFree(backend)!=QNN_SUCCESS)result=2;
 printf("RESULT passed_graphs=%u expected_graphs=24 status=%s\n",tests,result?"FAIL":"PASS");fflush(stdout);dlclose(lib);return result;
}
