"""独立定位短ON包驻留长尾，保留每段GPU时间。"""
from experiment import *

folder=ROOT/'runs/diagnostic03'
folder.mkdir()
monitor=Monitor(folder)
with torch.inference_mode(False),torch.no_grad():
    renderer=Renderer('tum')
try:
    with torch.inference_mode():
        warm(renderer)
        frames=[index%792 for index in range(1600)]
        _,result=measure(renderer,folder/'cartgs80','cartgs_full',3,frames,500000,0,monitor)
        print(json.dumps(result),flush=True)
finally:
    renderer.close()
    monitor.close()
