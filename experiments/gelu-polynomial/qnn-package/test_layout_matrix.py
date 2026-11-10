from pathlib import Path
import subprocess
p=Path(__file__).resolve().parent
s=(p/'GeluFp16.cpp').read_text()
test=(p/'test_qnn_gelu.c').read_text().replace('{1,32,64,1024}}','{1,32,64,1024},{2,9,5,33},{1,9,7,63},{2,3,3,65}}').replace('expected_graphs=24','expected_graphs=30')
for co in (0,1):
 for ci in (0,1):
  f=p/f'force-{co}{ci}';f.mkdir(exist_ok=True);(f/'build').mkdir(exist_ok=True)
  for n in ['LocalLicensedInterface.cpp','compact_gelu.h','gelu_tables.h','vector_core.h','core.cpp','core.h','coefficients.h','prepare_interface.py']:
   if not (f/n).exists():(f/n).symlink_to('../'+n)
  ot='F16CroutonTensor' if co else 'PlainFloat16Tensor';it='F16CroutonTensor' if ci else 'PlainFloat16Tensor'
  src='\n'.join(x for x in s.split('\n') if not ('DEF_PACKAGE_OP' in x and f', {ot}, {it}>' not in x))
  prop=f'{"Crouton" if co else "Flat"}("*"), {"Crouton" if ci else "Flat"}("in0"), '
  src=src.replace('MainMemory("*", "in0"))',prop+'MainMemory("*", "in0"))')
  (f/'GeluFp16.cpp').write_text(src);(f/'test_qnn_gelu.c').write_text(test)
  for n in ['build_host.sh','run_tests.sh']:(f/n).write_text((p/n).read_text().replace(str(p),str(f)).replace('passed_graphs=24 expected_graphs=24','passed_graphs=30 expected_graphs=30'))
  with (f/'build.log').open('w') as log:subprocess.run(['sh',str(f/'build_host.sh')],stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['sh',str(f/'run_tests.sh')],check=True)
  print(f.name,'PASS',flush=True)
