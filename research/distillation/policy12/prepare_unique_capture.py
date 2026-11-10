"""Instrument8 unique policy MLPs after exact CSE, preserving original teacher math."""
import copy,json
from pathlib import Path
import onnx
root=Path(__file__).resolve().parent;base=root.parent/'real-block';target=base/'teacher-policy8-cse-capture.onnx'
if target.exists():raise FileExistsError(target)
blocks=json.loads((root/'blocks.json').read_text());aliases=json.loads((root/'exact-cse.json').read_text())['aliases']
def canonical(name):
 while name in aliases:name=aliases[name]
 return name
selected=[b for b in blocks if b['name'] not in ['gelu_52','gelu_53','gelu_54','gelu_55']]
m=onnx.load(base/'teacher-exact-cse.onnx',load_external_data=False);vals={v.name:v for v in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)};seen={v.name for v in m.graph.output}
for b in selected:
 for key in ['x','y']:
  b['capture_'+key]=canonical(b[key]);name=b['capture_'+key]
  if name not in seen:m.graph.output.append(copy.deepcopy(vals[name]));seen.add(name)
onnx.save(m,target);onnx.checker.check_model(str(target));(root/'capture-blocks.json').write_text(json.dumps(selected,indent=2));print('unique blocks',len(selected),'outputs',len(m.graph.output))
