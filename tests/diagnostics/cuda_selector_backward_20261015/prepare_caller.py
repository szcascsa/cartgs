"""只在诊断 executable 内指定 backward strategy，不添加产品配置。"""
from pathlib import Path
D=Path(__file__).resolve().parent
root=D.parents[2]
s=(root/'src/selector_cuda.cpp').read_text()
s='#include <cstdlib>\n#include <iostream>\n'+s
s=s.replace('  std::vector<torch::Tensor> tensors=inputs;', '''  if(const char* value=std::getenv("SELECTOR_DIAGNOSTIC_BACKWARD")) {
    strategy=static_cast<BackwardStrategy>(std::atoi(value));
    static bool printed=false;
    if(!printed){std::cerr<<"diagnostic backward strategy="<<value<<'\\n';printed=true;}
  }
  std::vector<torch::Tensor> tensors=inputs;''')
(D/'diagnostic_network.cpp').write_text(s)
