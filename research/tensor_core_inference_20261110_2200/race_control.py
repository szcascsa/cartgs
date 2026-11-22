"""只加载原v31，定位整程序racecheck中的ATen告警，不加载Tensor扩展。"""
from study import *


@torch.no_grad()
def main():
    out = HERE / 'reports/race_control01'
    out.mkdir(exist_ok=False)
    _, source = load_scene(TUNE[0])
    student, _, _ = student_for(source, TUNE[0])
    run = engine(source, student, original=True, graph=False)
    camera = source.camera(len(source.scene.views) // 2)
    for budget in [0, 1, 17, 65, 1000]:
        packet, ids, stats = run.prepare(camera, budget=budget)
        assert len(ids) == min(budget, stats['eligible_candidates'])
    torch.cuda.synchronize()
    loaded = Path(run._ext.__file__).resolve()
    frozen = json.loads((HERE / 'reports/FROZEN_CANDIDATE.json').read_text())
    assert str(loaded) == frozen['baseline']
    assert sha(loaded) == frozen['baseline_sha256']
    save(out / 'SUMMARY.json', dict(passed=True, baseline_only=True,
         loaded_binary=str(loaded), loaded_sha256=sha(loaded),
         tensor_extension_loaded=False))
    print('ORIGINAL_V31_CONTROL_PASS', loaded, flush=True)


if __name__ == '__main__':
    main()
