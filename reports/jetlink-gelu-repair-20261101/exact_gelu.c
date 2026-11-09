// Experimental CPU GELU replacement for causal precision diagnosis.
// Every finite FP16 bit pattern is evaluated from the exact-erf definition.
// The table covers the complete FP16 domain; no calibration data are used.
#include <onnxruntime/onnxruntime_c_api.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint16_t table[2][65536];
static pthread_once_t table_once = PTHREAD_ONCE_INIT;
// The domain must outlive all registered sessions; it is process lifetime here.
static OrtCustomOpDomain *domain;
static pthread_mutex_t domain_lock = PTHREAD_MUTEX_INITIALIZER;

static void make_table(void) {
  for (uint32_t i = 0; i < 65536; ++i) {
    uint16_t bits = (uint16_t)i;
    _Float16 half;
    memcpy(&half, &bits, sizeof(half));
    double x = (double)half;
    _Float16 result = (_Float16)(0.5 * x * (1.0 + erf(x * 0.70710678118654752440)));
    memcpy(&table[0][i], &result, sizeof(result));
    result = (_Float16)(0.5 * x * (1.0 + tanh(0.79788456080286535588 * (x + 0.044715 * x * x * x))));
    memcpy(&table[1][i], &result, sizeof(result));
  }
}

typedef struct { const OrtApi *api; int fp32; int tanh_mode; } Kernel;
typedef struct { const void *input; void *output; size_t n; int fp32; int tanh_mode; } Work;

static void evaluate_chunk(void *data, size_t chunk) {
  Work *w = data;
  size_t start = chunk * 4096, end = start + 4096;
  if (end > w->n) end = w->n;
  if (w->fp32) {
    const float *x = w->input;
    float *y = w->output;
    for (size_t i = start; i < end; ++i) {
      double z = x[i];
      double f = w->tanh_mode ? tanh(0.79788456080286535588 * (z + 0.044715 * z * z * z))
                             : erf(z * 0.70710678118654752440);
      y[i] = (float)(0.5 * z * (1.0 + f));
    }
  } else {
    const uint16_t *x = w->input;
    uint16_t *y = w->output;
    const uint16_t *lookup = table[w->tanh_mode];
    for (size_t i = start; i < end; ++i) y[i] = lookup[x[i]];
  }
}

static const char *ORT_API_CALL op_name(const OrtCustomOp *op);
static OrtStatus *ORT_API_CALL create(const OrtCustomOp *op, const OrtApi *api,
                                    const OrtKernelInfo *info, void **out) {
  (void)info;
  Kernel *k = calloc(1, sizeof(*k));
  if (!k) return api->CreateStatus(ORT_FAIL, "ExactGelu allocation failed");
  k->api = api;
  k->fp32 = strstr(op_name(op), "32") != NULL;
  k->tanh_mode = strstr(op_name(op), "Tanh") != NULL;
  pthread_once(&table_once, make_table);
  *out = k;
  return NULL;
}

static OrtStatus *ORT_API_CALL compute(void *kernel, OrtKernelContext *ctx) {
  Kernel *k = kernel;
  const OrtApi *a = k->api;
  const OrtValue *input = NULL;
  OrtValue *output = NULL;
  OrtTensorTypeAndShapeInfo *shape = NULL;
  OrtStatus *status = a->KernelContext_GetInput(ctx, 0, &input);
  if (status) return status;
  status = a->GetTensorTypeAndShape(input, &shape);
  if (status) return status;
  size_t rank = 0, count = 0;
  int64_t dims[16];
  status = a->GetDimensionsCount(shape, &rank);
  if (!status && rank > 16) status = a->CreateStatus(ORT_INVALID_ARGUMENT, "ExactGelu rank > 16");
  if (!status) status = a->GetDimensions(shape, dims, rank);
  if (!status) status = a->GetTensorShapeElementCount(shape, &count);
  a->ReleaseTensorTypeAndShapeInfo(shape);
  if (status) return status;
  status = a->KernelContext_GetOutput(ctx, 0, dims, rank, &output);
  if (status) return status;
  void *in = NULL, *out = NULL;
  status = a->GetTensorMutableData((OrtValue *)input, &in);
  if (!status) status = a->GetTensorMutableData(output, &out);
  if (status) return status;
  Work work = {in, out, count, k->fp32, k->tanh_mode};
  return a->KernelContext_ParallelFor(ctx, evaluate_chunk, (count + 4095) / 4096, 0, &work);
}

static void ORT_API_CALL destroy(void *kernel) { free(kernel); }
static const char *ORT_API_CALL provider(const OrtCustomOp *op) { (void)op; return "CPUExecutionProvider"; }
static size_t ORT_API_CALL count(const OrtCustomOp *op) { (void)op; return 1; }
static OrtCustomOpInputOutputCharacteristic ORT_API_CALL required(const OrtCustomOp *op, size_t i) {
  (void)op; (void)i; return INPUT_OUTPUT_REQUIRED;
}
static OrtMemType ORT_API_CALL memory(const OrtCustomOp *op, size_t i) {
  (void)op; (void)i; return OrtMemTypeDefault;
}
static int ORT_API_CALL start_version(const OrtCustomOp *op) { (void)op; return 1; }
static int ORT_API_CALL end_version(const OrtCustomOp *op) { (void)op; return 1; }
static ONNXTensorElementDataType ORT_API_CALL type(const OrtCustomOp *op, size_t i);
#define DEFINITION { .version = ORT_API_VERSION, .GetName = op_name, \
  .GetExecutionProviderType = provider, .GetInputType = type, .GetOutputType = type, \
  .GetInputTypeCount = count, .GetOutputTypeCount = count, .KernelDestroy = destroy, \
  .GetInputCharacteristic = required, .GetOutputCharacteristic = required, \
  .GetInputMemoryType = memory, .CreateKernelV2 = create, .KernelComputeV2 = compute, \
  .GetStartVersion = start_version, .GetEndVersion = end_version }
static OrtCustomOp op16 = DEFINITION;
static OrtCustomOp op32 = DEFINITION;
static OrtCustomOp tanh16 = DEFINITION;
static OrtCustomOp tanh32 = DEFINITION;
static const char *ORT_API_CALL op_name(const OrtCustomOp *op) {
  if (op == &op32) return "ExactGelu32";
  if (op == &tanh16) return "ExactGeluTanh";
  if (op == &tanh32) return "ExactGeluTanh32";
  return "ExactGelu";
}
static ONNXTensorElementDataType ORT_API_CALL type(const OrtCustomOp *op, size_t i) {
  (void)i; return (op == &op32 || op == &tanh32) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

__attribute__((visibility("default")))
OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *base) {
  const OrtApi *api = base->GetApi(ORT_API_VERSION);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "ExactGelu ORT API version mismatch");
  pthread_mutex_lock(&domain_lock);
  OrtStatus *status = NULL;
  if (!domain) {
    status = api->CreateCustomOpDomain("jetlink.precision", &domain);
    if (!status) status = api->CustomOpDomain_Add(domain, &op16);
    if (!status) status = api->CustomOpDomain_Add(domain, &op32);
    if (!status) status = api->CustomOpDomain_Add(domain, &tanh16);
    if (!status) status = api->CustomOpDomain_Add(domain, &tanh32);
    if (status) {
      if (domain) api->ReleaseCustomOpDomain(domain);
      domain = NULL;
    }
  }
  if (!status) status = api->AddCustomOpDomain(options, domain);
  pthread_mutex_unlock(&domain_lock);
  return status;
}
