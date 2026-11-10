"""Locally adapt the licensed SDK interface; do not redistribute its output."""
import pathlib,re,sys
src=pathlib.Path(sys.argv[1]);dst=pathlib.Path(sys.argv[2]);s=src.read_text()
s=re.sub(r'DECLARE_PKG_OPS_OPTS_LIST\(PKG_(Softmax|Relu|ReluFp16|MaxPool)\)\n','',s)
s=s.replace('END_PKG_OPS_OPTS_LIST()','DECLARE_PKG_OPS_OPTS_LIST(PKG_GeluFp16)\nEND_PKG_OPS_OPTS_LIST()')
a=s.index('static constexpr auto sg_opNameRelu');b=s.index('static Qnn_ApiVersion_t',a)
s=s[:a]+'static constexpr auto sg_opNameErf = "GeluErfFp16";\nstatic constexpr auto sg_opNameTanh = "GeluTanhFp16";\nstatic std::array<const char *, 2> sg_opNames{{sg_opNameErf,sg_opNameTanh}};\n\n'+s[b:]
a=s.index('Qnn_ErrorHandle_t exampleValidateOpConfig(');b=s.index('\n/* The following functions',a)
s=s[:a]+'''Qnn_ErrorHandle_t exampleValidateOpConfig(Qnn_OpConfig_t cfg) {
  if (cfg.version != QNN_OPCONFIG_VERSION_1 || !cfg.v1.packageName || !cfg.v1.typeName ||
      std::string(cfg.v1.packageName) != sg_examplePackageName ||
      (std::string(cfg.v1.typeName) != sg_opNameErf && std::string(cfg.v1.typeName) != sg_opNameTanh) ||
      cfg.v1.numOfParams != 0 || cfg.v1.numOfInputs != 1 || cfg.v1.numOfOutputs != 1)
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  if (!cfg.v1.inputTensors || !cfg.v1.outputTensors)
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  const Qnn_Tensor_t &x = cfg.v1.inputTensors[0];
  const Qnn_Tensor_t &y = cfg.v1.outputTensors[0];
  if ((x.version != QNN_TENSOR_VERSION_1 && x.version != QNN_TENSOR_VERSION_2) ||
      (y.version != QNN_TENSOR_VERSION_1 && y.version != QNN_TENSOR_VERSION_2))
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  auto xd = x.version == QNN_TENSOR_VERSION_1 ? x.v1.dataType : x.v2.dataType;
  auto yd = y.version == QNN_TENSOR_VERSION_1 ? y.v1.dataType : y.v2.dataType;
  auto xr = x.version == QNN_TENSOR_VERSION_1 ? x.v1.rank : x.v2.rank;
  auto yr = y.version == QNN_TENSOR_VERSION_1 ? y.v1.rank : y.v2.rank;
  const uint32_t *xs = x.version == QNN_TENSOR_VERSION_1 ? x.v1.dimensions : x.v2.dimensions;
  const uint32_t *ys = y.version == QNN_TENSOR_VERSION_1 ? y.v1.dimensions : y.v2.dimensions;
  if (xd != QNN_DATATYPE_FLOAT_16 || yd != QNN_DATATYPE_FLOAT_16 || xr != 4 || yr != 4 || !xs || !ys)
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  for (unsigned i=0; i<4; ++i)
    if (xs[i] == 0 || xs[i] != ys[i]) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  return QNN_SUCCESS;
}
''' +s[b:]
dst.write_text(s)
