"""沿用上一轮完整质量协议，GPU oracle与原随机键，输出仅在本轮。"""
from common import *
code=(OLD/'quality.py').read_text().replace('from cpu_study import *','from common import *')
code=code.replace('engine=CpuRenderer(model,pc,ref.noise,6,student,True,True,True)','engine=Deep(model,pc,ref.noise,6,student,True,True,True)')
namespace=dict(__name__='deep_quality_protocol',__file__=str(HERE/'quality.py'))
exec(compile(code,str(HERE/'quality.py'),'exec'),namespace)
legacy_call=namespace['call']
def dispatch(engine,cam,condition,**kwargs):
    return request(engine,cam,condition,**kwargs) if isinstance(engine,Deep) else legacy_call(engine,cam,condition,**kwargs)
namespace['call']=dispatch
if __name__=='__main__':namespace['main']()
