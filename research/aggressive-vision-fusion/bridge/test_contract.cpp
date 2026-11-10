// Source-only ABI tests: no QNN libraries or native inference are loaded.
#include "bridge.cpp"
#include <cassert>
#include <functional>
int main(){
 uint32_t dims[]={1,32,512};uint8_t dynamic[]={0,0,1};
 Qnn_Tensor_t t=QNN_TENSOR_INIT;t.version=QNN_TENSOR_VERSION_2;t.v2.name="view_39";t.v2.dataFormat=QNN_TENSOR_DATA_FORMAT_DENSE;t.v2.dataType=QNN_DATATYPE_FLOAT_16;t.v2.rank=3;t.v2.dimensions=dims;
 std::vector<Qnn_Tensor_t> tensors={t};
 assert(Runtime::find(tensors,"view_39",{1,32,512})==0);
 int rejected=0;
 auto reject=[&](std::function<void()> mutate){tensors={t};mutate();try{Runtime::find(tensors,"view_39",{1,32,512});}catch(const std::runtime_error&){rejected++;return;}throw std::runtime_error("malformed descriptor accepted");};
 reject([&]{tensors[0].v2.name="wrong";});
 reject([&]{tensors.push_back(t);});
 reject([&]{tensors[0].v2.dimensions=nullptr;});
 reject([&]{tensors[0].v2.rank=2;});
 reject([&]{tensors[0].v2.dataType=QNN_DATATYPE_FLOAT_32;});
 reject([&]{tensors[0].v2.dataFormat=static_cast<Qnn_TensorDataFormat_t>(999);});
 reject([&]{tensors[0].v2.isDynamicDimensions=dynamic;});
 reject([&]{tensors[0].version=static_cast<Qnn_TensorVersion_t>(999);});
 assert(rejected==8);
 puts("PASS: 1 valid and 8 malformed descriptor tests; no native execution");
}
