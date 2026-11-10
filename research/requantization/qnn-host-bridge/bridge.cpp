// Linux host-functional ORT bridge. No device execution or speed claims.
#include <onnxruntime/onnxruntime_c_api.h>
#include "QnnInterface.h"
#include "QnnWrapperUtils.hpp"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"
#include <dlfcn.h>
#include <cmath>
#include <cfenv>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <vector>
#include <memory>
#include <stdexcept>
using namespace qnn_wrapper_api;
static std::mutex qnn_mutex, domain_mutex;
using Compose = ModelError_t(*)(Qnn_BackendHandle_t,QNN_INTERFACE_VER_TYPE,Qnn_ContextHandle_t,const GraphConfigInfo_t**,uint32_t,GraphInfoPtr_t**,uint32_t*,bool,QnnLog_Callback_t,QnnLog_Level_t);
using FreeInfo = ModelError_t(*)(GraphInfoPtr_t**,uint32_t);
static void check(Qnn_ErrorHandle_t r,const char* what){if(r)throw std::runtime_error(std::string(what)+": "+std::to_string(r));}
static void* openlib(const char* env,const char* fallback){const char*p=getenv(env);void*h=dlopen(p?p:fallback,RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());return h;}
struct Runtime {
 void* lib=nullptr;void* model=nullptr;QNN_INTERFACE_VER_TYPE api={};
 Qnn_BackendHandle_t backend=nullptr;Qnn_DeviceHandle_t device=nullptr;Qnn_ContextHandle_t context=nullptr;
 GraphInfoPtr_t* graphs=nullptr;uint32_t ng=0;FreeInfo freeinfo=nullptr;
 Qnn_Tensor_t x=QNN_TENSOR_INIT,y=QNN_TENSOR_INIT;Qnn_ScaleOffset_t xq{},yq{};
 std::vector<uint16_t> xb,yb;
 ~Runtime(){if(graphs&&freeinfo)freeinfo(&graphs,ng);if(context)api.contextFree(context,nullptr);if(device)api.deviceFree(device);if(backend)api.backendFree(backend);if(model)dlclose(model);if(lib)dlclose(lib);}
 void init(){
 lib=openlib("JETLINK_QNN_BACKEND","libQnnHtp.so");model=openlib("JETLINK_QNN_MODEL","/workspace/shared/jetlink-exact-integration/model-libs-native/x86_64-linux-clang/libqnn-first-projection-native.so");
 auto get=(Qnn_ErrorHandle_t(*)(const QnnInterface_t***,uint32_t*))dlsym(lib,"QnnInterface_getProviders");
 auto compose=(Compose)dlsym(model,"QnnModel_composeGraphs");freeinfo=(FreeInfo)dlsym(model,"QnnModel_freeGraphsInfo");
 if(!get||!compose||!freeinfo)throw std::runtime_error("QNN exports missing");
 const QnnInterface_t** p=nullptr;uint32_t n=0;check(get(&p,&n),"getProviders");bool found=false;
 for(uint32_t i=0;i<n;i++)if(p[i]->apiVersion.coreApiVersion.major==QNN_API_VERSION_MAJOR&&p[i]->apiVersion.coreApiVersion.minor>=QNN_API_VERSION_MINOR){api=p[i]->QNN_INTERFACE_VER_NAME;found=true;break;}
 if(!found)throw std::runtime_error("QNN API version mismatch");
 check(api.backendCreate(nullptr,nullptr,&backend),"backendCreate");
 QnnHtpDevice_CustomConfig_t arch={},soc={};arch.option=QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;arch.arch.deviceId=0;arch.arch.arch=QNN_HTP_DEVICE_ARCH_V73;soc.option=QNN_HTP_DEVICE_CONFIG_OPTION_SOC;soc.socModel=43;
 QnnDevice_Config_t ac=QNN_DEVICE_CONFIG_INIT,sc=QNN_DEVICE_CONFIG_INIT;ac.option=QNN_DEVICE_CONFIG_OPTION_CUSTOM;ac.customConfig=&arch;sc.option=QNN_DEVICE_CONFIG_OPTION_CUSTOM;sc.customConfig=&soc;const QnnDevice_Config_t* dc[]={&ac,&sc,nullptr};
 check(api.deviceCreate(nullptr,dc,&device),"deviceCreate");check(api.contextCreate(backend,device,nullptr,&context),"contextCreate");
 QnnHtpGraph_CustomConfig_t opt={};opt.option=QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;opt.optimizationOption.type=QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;opt.optimizationOption.floatValue=3;
 QnnGraph_Config_t gc=QNN_GRAPH_CONFIG_INIT;gc.option=QNN_GRAPH_CONFIG_OPTION_CUSTOM;gc.customConfig=&opt;const QnnGraph_Config_t* gcs[]={&gc,nullptr};
 char graph_name[]="qnn_first_projection_native";GraphConfigInfo_t gi={graph_name,gcs};const GraphConfigInfo_t* gis[]={&gi};
 check(compose(backend,api,context,gis,1,&graphs,&ng,false,nullptr,QNN_LOG_LEVEL_ERROR),"composeGraphs");
 if(ng!=1||graphs[0]->numInputTensors!=1||graphs[0]->numOutputTensors!=1)throw std::runtime_error("Expected one graph/input/output");
 x=graphs[0]->inputTensors[0];y=graphs[0]->outputTensors[0];validate(x,512);validate(y,1024);
 xq=getQnnTensorQuantParams(x).scaleOffsetEncoding;yq=getQnnTensorQuantParams(y).scaleOffsetEncoding;
 xb.resize(288*512);yb.resize(288*1024);setQnnTensorMemType(x,QNN_TENSORMEMTYPE_RAW);setQnnTensorClientBuf(x,{xb.data(),uint32_t(xb.size()*2)});setQnnTensorMemType(y,QNN_TENSORMEMTYPE_RAW);setQnnTensorClientBuf(y,{yb.data(),uint32_t(yb.size()*2)});
 check(api.graphFinalize(graphs[0]->graph,nullptr,nullptr),"graphFinalize");
 fprintf(stderr,"QnnFirstProjection: input scale=%.17g offset=%d output scale=%.17g offset=%d; x86 HTP SoC43 v73 O3\n",xq.scale,xq.offset,yq.scale,yq.offset);
 }
 static void validate(const Qnn_Tensor_t&t,uint32_t width){auto q=getQnnTensorQuantParams(t);auto d=getQnnTensorDimensions(t);if((t.version!=QNN_TENSOR_VERSION_1&&t.version!=QNN_TENSOR_VERSION_2)||getQnnTensorRank(t)!=3||!d||d[0]!=1||d[1]!=288||d[2]!=width||getQnnTensorDataType(t)!=QNN_DATATYPE_UFIXED_POINT_16||q.quantizationEncoding!=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET||!std::isfinite(q.scaleOffsetEncoding.scale)||q.scaleOffsetEncoding.scale<=0)throw std::runtime_error("Unexpected graph tensor metadata");}
 void execute(const uint16_t*in,uint16_t*out){
 struct NearestEven { int prior; NearestEven():prior(std::fegetround()){if(prior<0||std::fesetround(FE_TONEAREST))throw std::runtime_error("Cannot set nearest-even rounding");} ~NearestEven(){std::fesetround(prior);} } rounding;
 for(size_t i=0;i<xb.size();++i){_Float16 h;memcpy(&h,in+i,2);float v=(float)h;if(!std::isfinite(v))throw std::runtime_error("Nonfinite input rejected");float z=std::nearbyint(v/xq.scale)-float(xq.offset);xb[i]=(uint16_t)std::fmin(65535.f,std::fmax(0.f,z));}
 if(const char*path=getenv("JETLINK_DUMP_INPUT")){FILE*f=fopen(path,"wb");if(!f)throw std::runtime_error("dump open failed");fwrite(xb.data(),2,xb.size(),f);fclose(f);}
 check(api.graphExecute(graphs[0]->graph,&x,1,&y,1,nullptr,nullptr),"graphExecute");
 if(const char*path=getenv("JETLINK_DUMP_OUTPUT")){FILE*f=fopen(path,"wb");if(!f)throw std::runtime_error("dump open failed");fwrite(yb.data(),2,yb.size(),f);fclose(f);}
 for(size_t i=0;i<yb.size();++i){float f=float(int32_t(yb[i])+yq.offset)*yq.scale;_Float16 h=(_Float16)f;memcpy(out+i,&h,2);}
 }
};
struct Kernel{const OrtApi*a;std::unique_ptr<Runtime> r;};
static OrtStatus* ORT_API_CALL create(const OrtCustomOp*,const OrtApi*a,const OrtKernelInfo*,void**out){try{std::lock_guard<std::mutex> lock(qnn_mutex);auto k=std::make_unique<Kernel>();k->a=a;k->r=std::make_unique<Runtime>();k->r->init();*out=k.release();return nullptr;}catch(const std::exception&e){return a->CreateStatus(ORT_FAIL,e.what());}}
static void ORT_API_CALL destroy(void*p){std::lock_guard<std::mutex> lock(qnn_mutex);delete static_cast<Kernel*>(p);}
static OrtStatus* ORT_API_CALL compute(void*p,OrtKernelContext*c){auto k=static_cast<Kernel*>(p);auto a=k->a;const OrtValue*in=nullptr;OrtValue*out=nullptr;OrtTensorTypeAndShapeInfo*s=nullptr;OrtStatus*e=a->KernelContext_GetInput(c,0,&in);if(e)return e;e=a->GetTensorTypeAndShape(in,&s);if(e)return e;size_t rank=0;int64_t dims[3]={};ONNXTensorElementDataType type;e=a->GetDimensionsCount(s,&rank);if(!e&&rank!=3)e=a->CreateStatus(ORT_INVALID_ARGUMENT,"Expected rank 3");if(!e)e=a->GetDimensions(s,dims,3);if(!e)e=a->GetTensorElementType(s,&type);a->ReleaseTensorTypeAndShapeInfo(s);if(e)return e;if(dims[0]!=1||dims[1]!=288||dims[2]!=512||type!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)return a->CreateStatus(ORT_INVALID_ARGUMENT,"Expected FLOAT16 [1,288,512]");dims[2]=1024;e=a->KernelContext_GetOutput(c,0,dims,3,&out);if(e)return e;void*ip=nullptr,*op=nullptr;e=a->GetTensorMutableData((OrtValue*)in,&ip);if(!e)e=a->GetTensorMutableData(out,&op);if(e)return e;try{std::lock_guard<std::mutex> lock(qnn_mutex);k->r->execute((uint16_t*)ip,(uint16_t*)op);return nullptr;}catch(const std::exception&ex){return a->CreateStatus(ORT_FAIL,ex.what());}}
static const char* ORT_API_CALL name(const OrtCustomOp*){return "QnnFirstProjection";}
static const char* ORT_API_CALL provider(const OrtCustomOp*){return "CPUExecutionProvider";}
static size_t ORT_API_CALL count(const OrtCustomOp*){return 1;}
static ONNXTensorElementDataType ORT_API_CALL type(const OrtCustomOp*,size_t){return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;}
static OrtCustomOpInputOutputCharacteristic ORT_API_CALL required(const OrtCustomOp*,size_t){return INPUT_OUTPUT_REQUIRED;}
static OrtMemType ORT_API_CALL memory(const OrtCustomOp*,size_t){return OrtMemTypeDefault;}
static int ORT_API_CALL version(const OrtCustomOp*){return 1;}
static OrtCustomOp op=[](){OrtCustomOp o={};o.version=ORT_API_VERSION;o.GetName=name;o.GetExecutionProviderType=provider;o.GetInputType=type;o.GetOutputType=type;o.GetInputTypeCount=count;o.GetOutputTypeCount=count;o.KernelDestroy=destroy;o.GetInputCharacteristic=required;o.GetOutputCharacteristic=required;o.GetInputMemoryType=memory;o.CreateKernelV2=create;o.KernelComputeV2=compute;o.GetStartVersion=version;o.GetEndVersion=version;return o;}();
extern "C" __attribute__((visibility("default"))) OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions*options,const OrtApiBase*base){auto a=base->GetApi(ORT_API_VERSION);if(!a)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API mismatch");static OrtCustomOpDomain*d=nullptr;std::lock_guard<std::mutex>lock(domain_mutex);OrtStatus*e=nullptr;if(!d){e=a->CreateCustomOpDomain("jetlink.qnn",&d);if(!e)e=a->CustomOpDomain_Add(d,&op);if(e){if(d)a->ReleaseCustomOpDomain(d);d=nullptr;return e;}}return a->AddCustomOpDomain(options,d);}
