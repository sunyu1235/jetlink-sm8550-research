#!/usr/bin/env bash
# Experimental staged conversion. Final C++ graph-node validation is mandatory.
# Requires an already spliced, fully statically annotated ONNX and licensed local SDK.
set -euo pipefail
[[ $# == 6 ]] || { echo 'Usage: build_validated_context.sh MODEL NEW_OUT PRESERVE_CONVERTER QNN_LAUNCHER GELU_XML GELU_HOST_SO'; exit 2; }
model=$(realpath "$1"); out=$2; converter=$(realpath "$3"); launcher=$(realpath "$4"); xml=$(realpath "$5"); package=$(realpath "$6")
: "${QNN_SDK_ROOT:?Set SDK root}"
[[ ! -e "$out" ]] || { echo 'Output must be new'; exit 2; }; mkdir -p "$out/qnn"; out=$(realpath "$out")
# The converter's pre-fallback graph temporarily mixes FP32 and FP16; this phase
# is for serialization, NOT validation. Audit the final graph and validate below.
bash "$converter" --input_network "$model" --output_path "$out/qnn/mixed_first.cpp" --op_package_config "$xml" --float_fallback --float_bitwidth 16 --float_bias_bitwidth 16 --act_bitwidth 16 --weights_bitwidth 8 --no_simplification --preserve_io layout --disable_node_validation --disable_qnn_op_config_validation 2>&1 | tee "$out/converter.log"
python - "$out" "$QNN_SDK_ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);sdk=Path(sys.argv[2]);source=(p/'qnn/mixed_first.cpp').read_text();assert source.count('#define DO_GRAPH_NODE_VALIDATIONS 0')==1
(p/'qnn/mixed_first_validated.cpp').write_text(source.replace('#define DO_GRAPH_NODE_VALIDATIONS 0','#define DO_GRAPH_NODE_VALIDATIONS 1'))
g=json.load(open(p/'qnn/mixed_first_net.json'))['graph'];ts=g['tensors'];gelus=[n for n in g['nodes'].values() if 'Gelu' in n['type']];assert len(gelus)==63;assert all(ts[v]['data_type']==534 for n in gelus for v in n['input_names']+n['output_names'])
assert sum(t['data_type']==1032 for t in ts.values())==1
(p/'htp.json').write_text(json.dumps({'graphs':[{'graph_names':['mixed_first'],'O':3}],'devices':[{'soc_model':43,'dsp_arch':'v73'}]}))
(p/'extensions.json').write_text(json.dumps({'backend_extensions':{'shared_library_path':str(sdk/'lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so'),'config_file_path':str(p/'htp.json')}}))
PY
bash "$launcher" qnn-model-lib-generator -c "$out/qnn/mixed_first_validated.cpp" -b "$out/qnn/mixed_first.bin" -o "$out/model-libs-validated" -t x86_64-linux-clang 2>&1 | tee "$out/library.log"
bash "$launcher" qnn-context-binary-generator --model "$out/model-libs-validated/x86_64-linux-clang/libmixed_first.so" --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" --op_packages "$package:exampleInterfaceProvider:CPU" --config_file "$out/extensions.json" --binary_file mixed-first-sm8550 --output_dir "$out/context" --log_level verbose 2>&1 | tee "$out/context.log"
echo 'Offline validated preparation complete. Full context numerical execution and device speed still unproven.'
