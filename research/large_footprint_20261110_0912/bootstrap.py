"""独立低点数/大footprint研究，沿用只读最终版和原checkpoint。"""
from pathlib import Path
import hashlib,json,shutil,time

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'sparse_rasterizer_sixhour_20261110_0053'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    for name in ['reports/jobs','logs','evaluation','build','source']: (HERE/name).mkdir(parents=True,exist_ok=True)
    assert not (HERE/'reports/PROTOCOL.json').exists()
    for name in ['graph_forward.cu','graph_inputs.h','graph_kernels.cuh','graph_main.cu','ordered_tile_coverage.cuh',
        'reference_emit.cuh','fused_preprocess.h','preprocess_sort.h','chunked_pixels.cuh']:
        shutil.copy2(PREVIOUS/'source'/name,HERE/'source'/name)
    shutil.copytree(PREVIOUS/'source/reference',HERE/'source/reference',dirs_exist_ok=True)
    for name in ['launch.py','job_runner.py']:shutil.copy2(PREVIOUS/name,HERE/name)
    scenes=json.loads((PREVIOUS/'reports/SCENES.json').read_text());assert len(scenes)==24
    (HERE/'reports/SCENES.json').write_text(json.dumps(scenes,indent=2))
    if str(HERE).startswith('/home/'):
        protected=json.loads((PREVIOUS/'reports/PROTECTED_BEFORE.json').read_text())
        for name,value in protected.items():assert digest(name)==value,name
        for path in PREVIOUS.rglob('*'):
            if path.is_file():protected[str(path)]=digest(path)
        (HERE/'reports/PROTECTED_BEFORE.json').write_text(json.dumps(protected,indent=2))
    protocol=dict(created_at=time.time(),baseline_previous=str(PREVIOUS),reference='原始A为数值oracle；上一轮最终Graph G/准备R为性能对照',
        tuning=[s['id'] for s in scenes if s['split']=='tune'],heldout=[s['id'] for s in scenes if s['split']=='heldout'],
        main_ratios=[.01,.05,.1],other_conditions=['local0.03','local0.3','local0.7','global0.01','fixed0','fixed1000','fixed5000','full'],
        correctness='FP32 RGB/radii逐位一致，Tile身份及Alpha次序不变',full_gate='本轮允许Full变慢，代价逐场景报告，禁止Ratio/K绕开新路径',
        performance='全24组、连续/跳转视角、同输入、轮换顺序、至少两轮；纯渲染/完整延迟分开',
        workspace='沿用1GiB总上限、256MiB位表上限、一个device/Stream一个实例；不缓存帧内容',
        policy='只写独立研究目录，不改生产/训练/原模型/共享conda，不commit/push/部署，不操作外部进程')
    (HERE/'reports/PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    print('BOOTSTRAP_COMPLETE',len(scenes),flush=True)


if __name__=='__main__':main()
