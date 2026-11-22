"""仅依据开发集锁定候选集合；保留集只能验收，不能调参。"""
from nine_study import *


def main():
    path=HERE/'reports/FROZEN_RENDER.json';assert not path.exists()
    verify_protected()
    assert json.loads((HERE/'reports/DEV_CONTROLLER06.json').read_text())['state']=='complete'
    boundary=json.loads((HERE/'reports/edges01/SUMMARY.json').read_text());assert boundary['passed']
    selected=[37,38,39,40]
    sources={str(p.relative_to(HERE)):sha(p) for p in (HERE/'source').rglob('*') if p.is_file()}
    sources['render_runtime.py']=sha(HERE/'render_runtime.py')
    save(path,dict(state='frozen_pending_validation',modes=selected,primary=38,fallback=37,near_exact=40,
        frontier_only=39,binary=extension().__file__,binary_sha256=sha(extension().__file__),sources=sources,
        tune=TUNE,heldout=[s['id'] for s in SCENES if s['slug'] not in TUNE],time=time.time(),
        note='四候选独立质量验收；不得根据保留集改变参数；近似不是bitwise'))
    print('RENDER_SET_FROZEN',selected,flush=True)


if __name__=='__main__':main()
