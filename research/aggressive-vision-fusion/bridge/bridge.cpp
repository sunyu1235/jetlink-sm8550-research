// Single saved-context vision pipeline. Device qualification is pending.
#include <algorithm>
#include <cstdint>
#include <onnxruntime/onnxruntime_c_api.h>
#include "QnnInterface.h"
#include "QnnWrapperUtils.hpp"
#include "System/QnnSystemInterface.h"
#include "HTP/QnnHtpDevice.h"
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <mutex>
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
using namespace qnn_wrapper_api;
static std::mutex qnn_mutex, domain_mutex;
static void check(Qnn_ErrorHandle_t r,const char*w){if(r)throw std::runtime_error(std::string(w)+": "+std::to_string(r));}
static const char* env(const char*n,const char*d){
 const char*p=getenv(n);
#if defined(__ANDROID__)
 (void)d;
 if(!p||!*p)throw std::runtime_error(std::string("Android requires explicit ")+n);
 return p;
#else
 return p?p:d;
#endif
}
#if defined(__ANDROID__)
static const char* package_target="HTP";
static const char* execution_scope="Android bench candidate; device qualification not established";
#else
static const char* package_target="CPU";
static const char* execution_scope="Linux host only";
#endif
static void* lib(const char*n,const char*d){void*h=dlopen(env(n,d),RTLD_NOW|RTLD_GLOBAL);if(!h)throw std::runtime_error(dlerror());return h;}
// Tensor descriptor pointers are owned by sys/binary; retain both until contextFree.
struct Graph {Qnn_ContextHandle_t ctx=nullptr;Qnn_GraphHandle_t graph=nullptr;QnnSystemContext_Handle_t sys=nullptr;std::vector<char> binary;std::vector<Qnn_Tensor_t> in,out;};
struct Runtime {
 void *backend_lib=nullptr,*system_lib=nullptr;QNN_INTERFACE_VER_TYPE api={};QNN_SYSTEM_INTERFACE_VER_TYPE sapi={};Qnn_BackendHandle_t backend=nullptr;Qnn_DeviceHandle_t device=nullptr;Graph vision;
 ~Runtime(){for(Graph*g:{&vision}){if(g->ctx)api.contextFree(g->ctx,nullptr);if(g->sys)sapi.systemContextFree(g->sys);}if(device)api.deviceFree(device);if(backend)api.backendFree(backend);if(system_lib)dlclose(system_lib);if(backend_lib)dlclose(backend_lib);}
 template<class I>void info(Graph&g,const I&i){if(!i.graphName||std::string(i.graphName)!="vision_merged"||i.numGraphInputs!=1||i.numGraphOutputs!=1||!i.graphInputs||!i.graphOutputs)throw std::runtime_error("bad graph input count");g.in.assign(i.graphInputs,i.graphInputs+i.numGraphInputs);g.out.assign(i.graphOutputs,i.graphOutputs+i.numGraphOutputs);check(api.graphRetrieve(g.ctx,i.graphName,&g.graph),"graphRetrieve");}
 void load(Graph&g,const char*path){std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error(std::string("Cannot read context ")+path);g.binary.assign(std::istreambuf_iterator<char>(f),{});if(g.binary.empty())throw std::runtime_error("empty context");check(sapi.systemContextCreate(&g.sys),"systemContextCreate");const QnnSystemContext_BinaryInfo_t*b=nullptr;Qnn_ContextBinarySize_t size=0;check(sapi.systemContextGetBinaryInfo(g.sys,g.binary.data(),g.binary.size(),&b,&size),"binaryInfo");if(!b)throw std::runtime_error("null binary metadata");uint32_t n=0;QnnSystemContext_GraphInfo_t*gs=nullptr;
 switch(b->version){case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:n=b->contextBinaryInfoV1.numGraphs;gs=b->contextBinaryInfoV1.graphs;break;case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:n=b->contextBinaryInfoV2.numGraphs;gs=b->contextBinaryInfoV2.graphs;break;case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:n=b->contextBinaryInfoV3.numGraphs;gs=b->contextBinaryInfoV3.graphs;break;default:throw std::runtime_error("unsupported binary version");}if(n!=1||!gs)throw std::runtime_error("expected one graph");check(api.contextCreateFromBinary(backend,device,nullptr,g.binary.data(),g.binary.size(),&g.ctx,nullptr),"contextCreateFromBinary");switch(gs->version){case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:info(g,gs->graphInfoV1);break;case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:info(g,gs->graphInfoV2);break;case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:info(g,gs->graphInfoV3);break;default:throw std::runtime_error("unsupported graph info version");}}
 static size_t find(std::vector<Qnn_Tensor_t>&v,const std::string&name,const std::vector<uint32_t>&dims){size_t pos=v.size();for(size_t i=0;i<v.size();i++){if(v[i].version!=QNN_TENSOR_VERSION_1&&v[i].version!=QNN_TENSOR_VERSION_2)throw std::runtime_error("unsupported tensor version");const char*n=getQnnTensorName(v[i]);if(n&&name==n){if(pos!=v.size())throw std::runtime_error("duplicate tensor name");pos=i;}}if(pos==v.size())throw std::runtime_error("missing tensor "+name);auto&t=v[pos];if(getQnnTensorDataFormat(t)!=QNN_TENSOR_DATA_FORMAT_DENSE||getQnnTensorDataType(t)!=QNN_DATATYPE_FLOAT_16||getQnnTensorRank(t)!=dims.size()||!getQnnTensorDimensions(t)||!std::equal(dims.begin(),dims.end(),getQnnTensorDimensions(t)))throw std::runtime_error("tensor contract mismatch "+name);auto*dynamic=getQnnTensorIsDynamicDimensions(t);if(dynamic&&std::any_of(dynamic,dynamic+dims.size(),[](uint8_t v){return v!=0;}))throw std::runtime_error("dynamic tensor unsupported "+name);return pos;}
 static void bind(Qnn_Tensor_t&t,void*p,size_t bytes){setQnnTensorMemType(t,QNN_TENSORMEMTYPE_RAW);setQnnTensorClientBuf(t,{p,uint32_t(bytes)});}
 void init(){backend_lib=lib("JETLINK_QNN_BACKEND","libQnnHtp.so");system_lib=lib("JETLINK_QNN_SYSTEM","libQnnSystem.so");auto get=(Qnn_ErrorHandle_t(*)(const QnnInterface_t***,uint32_t*))dlsym(backend_lib,"QnnInterface_getProviders");auto sget=(Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t***,uint32_t*))dlsym(system_lib,"QnnSystemInterface_getProviders");if(!get||!sget)throw std::runtime_error("provider exports missing");const QnnInterface_t**p=nullptr;uint32_t n=0;check(get(&p,&n),"getProviders");bool found=false;for(uint32_t i=0;i<n;i++)if(p[i]->apiVersion.coreApiVersion.major==QNN_API_VERSION_MAJOR&&p[i]->apiVersion.coreApiVersion.minor>=QNN_API_VERSION_MINOR){api=p[i]->QNN_INTERFACE_VER_NAME;found=true;break;}if(!found)throw std::runtime_error("backend version mismatch");const QnnSystemInterface_t**sp=nullptr;check(sget(&sp,&n),"systemProviders");found=false;for(uint32_t i=0;i<n;i++)if(sp[i]->systemApiVersion.major==QNN_SYSTEM_API_VERSION_MAJOR&&sp[i]->systemApiVersion.minor>=QNN_SYSTEM_API_VERSION_MINOR){sapi=sp[i]->QNN_SYSTEM_INTERFACE_VER_NAME;found=true;break;}if(!found)throw std::runtime_error("system version mismatch");check(api.backendCreate(nullptr,nullptr,&backend),"backendCreate");check(api.backendRegisterOpPackage(backend,env("JETLINK_GELU_PACKAGE","/workspace/shared/htp-gelu-native21/build/libJetlinkGeluHostHidden.so"),"exampleInterfaceProvider",package_target),"registerPackage");QnnHtpDevice_CustomConfig_t arch={},soc={};arch.option=QNN_HTP_DEVICE_CONFIG_OPTION_ARCH;arch.arch.deviceId=0;arch.arch.arch=QNN_HTP_DEVICE_ARCH_V73;soc.option=QNN_HTP_DEVICE_CONFIG_OPTION_SOC;soc.socModel=43;QnnDevice_Config_t ac=QNN_DEVICE_CONFIG_INIT,sc=QNN_DEVICE_CONFIG_INIT;ac.option=QNN_DEVICE_CONFIG_OPTION_CUSTOM;ac.customConfig=&arch;sc.option=QNN_DEVICE_CONFIG_OPTION_CUSTOM;sc.customConfig=&soc;const QnnDevice_Config_t*dc[]={&ac,&sc,nullptr};check(api.deviceCreate(nullptr,dc,&device),"deviceCreate");load(vision,env("JETLINK_VISION_CONTEXT","/workspace/shared/jetlink-aggressive-fusion-20261110/vision/context/vision.bin"));find(vision.in,"cat_2",{1,24,128,256});find(vision.out,"view_39",{1,32,512});fprintf(stderr,"VisionNative: one saved context; package target=%s; %s\n",package_target,execution_scope);}
 static void finite(const void*data,size_t count){auto*p=static_cast<const uint16_t*>(data);for(size_t j=0;j<count;j++)if((p[j]&0x7c00)==0x7c00)throw std::runtime_error("nonfinite vision tensor");}
 void execute(void*input,void*output){finite(input,24*128*256);bind(vision.in[0],input,24*128*256*2);bind(vision.out[0],output,32*512*2);check(api.graphExecute(vision.graph,vision.in.data(),1,vision.out.data(),1,nullptr,nullptr),"vision graphExecute");finite(output,32*512);}

};
struct Kernel{const OrtApi*a;std::unique_ptr<Runtime>r;};
static OrtStatus* ORT_API_CALL create(const OrtCustomOp*,const OrtApi*a,const OrtKernelInfo*,void**out){try{std::lock_guard<std::mutex>lock(qnn_mutex);auto k=std::make_unique<Kernel>();k->a=a;*out=k.release();return nullptr;}catch(const std::exception&e){return a->CreateStatus(ORT_FAIL,e.what());}}
static void ORT_API_CALL destroy(void*p){std::lock_guard<std::mutex>lock(qnn_mutex);delete static_cast<Kernel*>(p);}
static OrtStatus* ORT_API_CALL compute(void*p,OrtKernelContext*c){auto k=static_cast<Kernel*>(p);auto a=k->a;const OrtValue*in=nullptr;OrtTensorTypeAndShapeInfo*s=nullptr;auto e=a->KernelContext_GetInput(c,0,&in);if(e)return e;if(!in)return a->CreateStatus(ORT_INVALID_ARGUMENT,"Missing vision input");e=a->GetTensorTypeAndShape(in,&s);if(e)return e;size_t rank=0;int64_t d[4]={};ONNXTensorElementDataType type;e=a->GetDimensionsCount(s,&rank);if(!e&&rank!=4)e=a->CreateStatus(ORT_INVALID_ARGUMENT,"Expected rank4");if(!e)e=a->GetDimensions(s,d,4);if(!e)e=a->GetTensorElementType(s,&type);a->ReleaseTensorTypeAndShapeInfo(s);if(e)return e;if(d[0]!=1||d[1]!=24||d[2]!=128||d[3]!=256||type!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)return a->CreateStatus(ORT_INVALID_ARGUMENT,"Vision input requires FP16 [1,24,128,256]");void*ip=nullptr;void*optr=nullptr;e=a->GetTensorMutableData((OrtValue*)in,&ip);if(e)return e;
 try{Runtime::finite(ip,24*128*256);OrtValue*out=nullptr;int64_t od[]={1,32,512};e=a->KernelContext_GetOutput(c,0,od,3,&out);if(!e)e=a->GetTensorMutableData(out,&optr);if(e)return e;std::lock_guard<std::mutex>lock(qnn_mutex);if(!k->r){auto r=std::make_unique<Runtime>();r->init();k->r=std::move(r);}k->r->execute(ip,optr);return nullptr;}catch(const std::exception&e){return a->CreateStatus(ORT_FAIL,e.what());}catch(...){return a->CreateStatus(ORT_FAIL,"Unknown vision runtime failure");}}
static const char* ORT_API_CALL name(const OrtCustomOp*){return "QnnVisionPipeline";}
static const char* ORT_API_CALL provider(const OrtCustomOp*){return "CPUExecutionProvider";}
static size_t ORT_API_CALL inputs(const OrtCustomOp*){return 1;}
static size_t ORT_API_CALL outputs(const OrtCustomOp*){return 1;}
static ONNXTensorElementDataType ORT_API_CALL type(const OrtCustomOp*,size_t){return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;}
static OrtCustomOpInputOutputCharacteristic ORT_API_CALL required(const OrtCustomOp*,size_t){return INPUT_OUTPUT_REQUIRED;}
static OrtMemType ORT_API_CALL memory(const OrtCustomOp*,size_t){return OrtMemTypeDefault;}
static int ORT_API_CALL version(const OrtCustomOp*){return 1;}
static OrtCustomOp op=[](){OrtCustomOp o={};o.version=ORT_API_VERSION;o.GetName=name;o.GetExecutionProviderType=provider;o.GetInputType=type;o.GetOutputType=type;o.GetInputTypeCount=inputs;o.GetOutputTypeCount=outputs;o.KernelDestroy=destroy;o.GetInputCharacteristic=required;o.GetOutputCharacteristic=required;o.GetInputMemoryType=memory;o.CreateKernelV2=create;o.KernelComputeV2=compute;o.GetStartVersion=version;o.GetEndVersion=version;return o;}();
extern "C" __attribute__((visibility("default"))) OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions*options,const OrtApiBase*base){auto a=base->GetApi(ORT_API_VERSION);if(!a)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API mismatch");static OrtCustomOpDomain*d=nullptr;std::lock_guard<std::mutex>lock(domain_mutex);OrtStatus*e=nullptr;if(!d){e=a->CreateCustomOpDomain("jetlink.qnn",&d);if(!e)e=a->CustomOpDomain_Add(d,&op);if(e){if(d)a->ReleaseCustomOpDomain(d);d=nullptr;return e;}}return a->AddCustomOpDomain(options,d);}
