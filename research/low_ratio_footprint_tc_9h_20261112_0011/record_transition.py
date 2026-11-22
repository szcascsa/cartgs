"""记录已完成验收的阶段切换，不重置墙钟，不改预注册门限。"""
from pathlib import Path
from datetime import datetime
import json
import time

HERE=Path(__file__).resolve().parent


def main():
    protocol=json.loads((HERE/'reports/PROTOCOL.json').read_text())
    summary=json.loads((HERE/'reports/RENDER_AGGREGATE/SUMMARY.json').read_text())
    accepted=json.loads((HERE/'reports/ACCEPTED_RENDER.json').read_text())
    controller=json.loads((HERE/'reports/RENDER_CONTROLLER.json').read_text())
    assert controller['state']=='complete'
    quality=next(v for v in summary['quality_gate'] if v['variant']==accepted['variant'])
    gain=next(v for v in summary['primary_gain'] if v['variant']==accepted['variant'])
    reduction=1-1/gain['speedup']
    assert quality['passed'] and reduction>=protocol['meaningful_render_latency_reduction']
    assert gain['slower_scene_rounds']==0
    for tool in ['memcheck','initcheck','synccheck']:
        job=next(n for n in controller['accepted'] if n.startswith(tool))
        assert 'ERROR SUMMARY: 0 errors' in (HERE/'logs'/(job+'.log')).read_text()
    path=HERE/'reports/PHASE_TRANSITION.json'
    assert not path.exists()
    record=dict(from_phase='rendering',to_phase='joint_inference',time=time.time(),
        local_time=datetime.now().astimezone().isoformat(),original_start=protocol['start'],
        original_deadline=protocol['deadline'],reason='提前达到渲染目标并完成双轮跨场景和画质验收',
        render_candidate=accepted,latency_reduction=reduction,quality=quality,gain=gain,
        sanitizer=['memcheck','initcheck','synccheck'],full_costs_retained=True)
    path.write_text(json.dumps(record,indent=2,ensure_ascii=False))
    (HERE/'reports/STATUS.json').write_text(json.dumps(dict(state='running',phase='joint_inference',
        time=record['time'],start=protocol['start'],deadline=protocol['deadline']),indent=2))
    print('EARLY_PHASE_SWITCH',record['local_time'],reduction)


if __name__=='__main__':main()
