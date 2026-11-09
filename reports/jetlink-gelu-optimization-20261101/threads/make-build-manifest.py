import hashlib,json
from pathlib import Path
root=Path(r'C:\Users\sunbo\Documents\openpilot')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
entries={
 'thread_runner_source':'reports/jetlink-gelu-optimization-20261101/threads/history_probe_threads.c',
 'ort_header_copy':'reports/jetlink-gelu-optimization-20261101/threads/jl_ort.h',
 'qnn_wrapper_source':'reports/jetlink-gelu-repair-20261101/qnn_jl_custom.c',
 'build_script':'reports/jetlink-gelu-optimization-20261101/threads/build_threads.ps1',
 'built_runner':'outputs/jetlink-gelu-optimization-20261101/threads/history_probe_threads',
 'thread_summary':'reports/jetlink-gelu-optimization-20261101/threads/summary.json',
 'thread_checks':'reports/jetlink-gelu-optimization-20261101/threads/checks.json',
 'thread_report':'reports/jetlink-gelu-optimization-20261101/threads/README.md',
 'accepted_original_library':'outputs/jetlink-gelu-repair-20261101/libspecifiedgelu.so',
 'context_model':'outputs/jetlink-gelu-repair-20261101/light-speed-gelu-both-ctx.onnx',
 'history_input_manifest':'reports/jetlink-gelu-history-20261101/input-manifest.json',
 'unit_range_preflight':'reports/jetlink-gelu-optimization-20261101/threads/preflight-thread-range.ps1',
}
artifacts={}
for key,rel in entries.items():
 p=root/rel
 artifacts[key]={'path':rel,'exists':p.is_file(),'size_bytes':p.stat().st_size if p.is_file() else None,'sha256':sha(p) if p.is_file() else None}
remote={'path':'/data/local/tmp/jetlink-gelu-optimization-20261101/threads/history_probe_threads','sha256':'57091fba0fb396483fce99881e307ede319c44e36ed2df93b94cd7579195a1ac','matches_local':artifacts['built_runner']['sha256']=='57091fba0fb396483fce99881e307ede319c44e36ed2df93b94cd7579195a1ac'}
manifest={'compiler':'C:/mp-sdk/ndk/27.3.13750724/toolchains/llvm/prebuilt/windows-x86_64/bin/aarch64-linux-android30-clang.cmd','target':'aarch64-linux-android30','flags':['-O2','-Wall','-Wextra','-Werror','-DJL_ORT_DLOPEN','-ldl'],'compile_success':True,'remote_deployment':remote,'artifacts':artifacts}
(root/'reports/jetlink-gelu-optimization-20261101/threads/build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print('built hash',artifacts['built_runner']['sha256'],'remote matches',remote['matches_local'])
