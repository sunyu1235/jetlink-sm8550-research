#!/bin/bash
# Prints the exact command unless explicitly passed --run. Full conversion needs a granted slot.
set -euo pipefail
run=false
if [[ ${1:-} == --run ]]; then run=true; shift; fi
if [[ $# != 2 ]]; then echo "usage: bash $0 [--run] INPUT_ONNX OUTPUT_CPP" >&2; exit 2; fi
launcher=${PRESERVE_CONVERTER_LAUNCHER:-/workspace/shared/htp-gelu/integration/run_preserve_converter.sh}
xml=${GELU_OP_CONFIG:-/workspace/shared/htp-gelu/integration/Gelu.xml}
command=(bash "$launcher" --input_network "$1" --output_path "$2"
  --op_package_config "$xml"
  --float_fallback --float_bitwidth 16 --float_bias_bitwidth 16
  --act_bitwidth 16 --weights_bitwidth 8 --no_simplification --preserve_io layout)
printf '%q ' "${command[@]}"; printf '\n'
if "$run"; then
  [[ -f "$1" ]] || { echo "Missing spliced model: $1" >&2; exit 2; }
  [[ ! -e "$2" ]] || { echo "Refusing to overwrite converter output: $2" >&2; exit 2; }
  mkdir -p "$(dirname "$2")"
  exec "${command[@]}"
fi
