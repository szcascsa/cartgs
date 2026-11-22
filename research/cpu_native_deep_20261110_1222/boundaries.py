"""复用33项成熟边界测试，后端改用本轮Deep，输出在本轮。"""
from common import *
code=(OLD/'boundaries.py').read_text().replace('from cpu_study import *','from common import *')
code=code.replace("engine=CpuRenderer(source['model'],source['pc'],ref.noise,6,selector_prefix=True,factored=True)","engine=Deep(source['model'],source['pc'],ref.noise,6,selector_prefix=True,factored=True)")
namespace=dict(__name__='deep_boundary_protocol',__file__=str(HERE/'boundaries.py'))
exec(compile(code,str(HERE/'boundaries.py'),'exec'),namespace)
if __name__=='__main__':namespace['main']()
