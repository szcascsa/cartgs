"""复用33项边界检查；最终Small冻结后端。"""
from common import *
code=(ORIGINAL/'boundaries.py').read_text().replace('from cpu_study import *','from common import *')
code=code.replace("engine=CpuRenderer(source['model'],source['pc'],ref.noise,6,selector_prefix=True,factored=True)","engine=Mid(source['model'],source['pc'],ref.noise,6,selector_prefix=True,factored=True,selection_mode=1,comp_batch=6)")
namespace=dict(__name__='small_boundary_protocol',__file__=str(HERE/'boundaries.py'))
exec(compile(code,str(HERE/'boundaries.py'),'exec'),namespace)
if __name__=='__main__':namespace['main']()
