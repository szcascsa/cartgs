"""只读CUDA libdevice，判断原精确expf是否有可安全省略的通用分支。"""
import ctypes as C
from pathlib import Path

HERE=Path(__file__).resolve().parent
lib=C.CDLL('/usr/lib/x86_64-linux-gnu/libLLVM-12.so')
def bind(name,result,args):
    f=getattr(lib,name);f.restype=result;f.argtypes=args;return f
ptr=C.c_void_p
create=bind('LLVMCreateMemoryBufferWithContentsOfFile',C.c_int,[C.c_char_p,C.POINTER(ptr),C.POINTER(C.c_char_p)])
context=bind('LLVMContextCreate',ptr,[])()
parse=bind('LLVMParseBitcodeInContext2',C.c_int,[ptr,ptr,C.POINTER(ptr)])
get=bind('LLVMGetNamedFunction',ptr,[ptr,C.c_char_p])
show=bind('LLVMPrintValueToString',ptr,[ptr])
memory,error,module=ptr(),C.c_char_p(),ptr()
assert create(b'/usr/local/cuda-11.8/nvvm/libdevice/libdevice.10.bc',C.byref(memory),C.byref(error))==0,error.value
assert parse(context,memory,C.byref(module))==0
function=get(module,b'__nv_expf')
assert function
ir=C.string_at(show(function)).decode()
(HERE/'reports/CUDA_EXPF_IR.txt').write_text(ir)
print(ir[:12000])
