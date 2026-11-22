"""等待正式渲染控制器结束后补齐质量，再按预注册规则决定阶段切换。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    path=HERE/'reports/POST_RENDER.json';assert not path.exists()
    state=dict(state='waiting',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_at=time.time())
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
    save()
    try:
        while True:
            controller=json.loads((HERE/'reports/RENDER_CONTROLLER.json').read_text())
            if controller['state'] in ('complete','failed'):break
            time.sleep(2)
        assert controller['state']=='complete',controller
        state['state']='quality_supplement';save()
        subprocess.run([sys.executable,'-u','-B','quality_supplement.py'],cwd=HERE,check=True)
        subprocess.run([sys.executable,'-u','-B','aggregate_render.py'],cwd=HERE,check=True)
        summary=json.loads((HERE/'reports/RENDER_AGGREGATE/SUMMARY.json').read_text())
        quality={r['variant']:r['passed'] for r in summary['quality_gate']}
        gain={r['variant']:r['speedup'] for r in summary['primary_gain']}
        choice=next((v for v in ['tile_safe_bitexp_stop','tile_safe_bitexp','tile_safe_fast'] if quality[v]),None)
        assert choice is not None,'全部候选质量门限失败，保留F2'
        mode={'tile_safe_bitexp_stop':38,'tile_safe_bitexp':37,'tile_safe_fast':40}[choice]
        accepted=dict(state='quality_accepted',mode=mode,variant=choice,render_speedup=gain[choice],time=time.time(),
            quality=summary['quality_gate'],requires_joint_quality_validation=True,production_changed=False)
        (HERE/'reports/ACCEPTED_RENDER.json').write_text(json.dumps(accepted,indent=2,ensure_ascii=False))
        state.update(state='complete',ended_at=time.time(),accepted_mode=mode);save()
    except BaseException as error:
        state.update(state='failed',error=repr(error),ended_at=time.time());save();raise


if __name__=='__main__':main()
