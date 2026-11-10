# Recreate the converter input, without redistributing models or SDK files

These authored preparation tools are carried forward from the previous QNN
integration. They are included because `preserve_fixed.cpp/bin` are generated
local artifacts, not files available in the public source archive.

Use the exact original embedded-weight `light-speed.onnx`, SHA-256
`6274eb4a3f3eb7ad1ab341d222bbba2ba709b06645dbbd15ac4971d6cb22ab21`,
and QAIRT/QNN 2.42.0.251225. Work in a new directory and retain the original.
These scripts intentionally reject different node counts or lossy bias repair.
Do not weaken those checks to accommodate another model.

1. `map_model.py SOURCE.onnx MAPPED.onnx --report map.json` maps all 36 erf and
   27 tanh activations to the custom package, adding rank-only wrappers where
   needed. Source/output must be in the same directory, so external tensor
   locations remain valid. It does not change existing initializer bytes.
2. `promote_images.py MAPPED.onnx IMAGES16.onnx --report images.json` applies the
   previously tested image-boundary promotion. Its input contract is FP16 pixel
   values exactly representing the original uint8 values 0..255. This is not an
   arbitrary input-type substitution. The final vision bridge retains the
   original cast and original external full-model signatures.
3. Configure the official SDK Python environment, including its Python module
   path and supported dependencies. With `QNN_SDK_ROOT` set, run the wrapper:

   ```sh
   python convert_preserve_arithmetic.py \
     --input_network /absolute/path/IMAGES16.onnx \
     --output_path /absolute/new/preparation/preserve.cpp \
     --op_package_config /absolute/path/Gelu.xml \
     --float_bitwidth 16 --float_bias_bitwidth 16
   ```

   The wrapper disables one version-specific arithmetic-folding helper in that
   process. It does not edit SDK files. Removing this hook or using a different
   converter version can fold across FP16 rounding and is a different candidate.
4. Run `repair_preserved_bias.py --directory /absolute/new/preparation`.
   It accepts exactly 19 mixed-type FC biases only when float32→float16→float32
   preserves every bit of their represented values. It then writes
   `preserve_fixed.cpp/bin`. It refuses to overwrite an existing result.
5. Pass this directory to `../extract_region.py` or
   `../run_native_region.sh`. Use the validated four-layout vector package for
   the final candidate, not the earlier Plain-only package or an uncorrected
   polynomial.

The existing frozen converter artifacts were used for the numerical results in
the parent README. The portable bias-repair CLI was rerun on copies/links of
those converter inputs and reproduced both fixed files byte for byte:

- `preserve_fixed.cpp`: b8d60b95d25e32cc1b703f44d4c8906e33db2e18c3a1f411fff4120c86a07838
- `preserve_fixed.bin`: ccc603b554aff8caaacfe7d1a13a5da0c972eb755ce54da6ea74669172990040

The full mapping/conversion sequence was not redundantly rerun for this source
packaging pass. Earlier tested generated artifacts and hashes remain the
executed candidate. A new executor must check its own generated ABI, numerical
results and package compatibility; successful source extraction is not target
qualification.

This folder contains no SDK implementation, generated converter C++, original
model, learned weights, context binary or runtime library.
