"""新增保留视角固定属性画质；同Gaussian输入与F2配对，不需要Student前向。"""
from release_joint import *
from nine_study import nested_packet
import lpips
from torchmetrics.functional.image import structural_similarity_index_measure as ssim


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--group', required=True, type=int)
    args = parser.parse_args()
    out = HERE / 'reports' / args.output
    out.mkdir(exist_ok=False)
    mode = json.loads((HERE / 'reports/SELECTED_RASTER_20261115.json').read_text())['mode']
    perceptual = lpips.LPIPS(net='alex').cuda().eval()
    rows = []
    def metrics(image, gt):
        image = image.clamp(0, 1)
        return dict(psnr=float(-10 * torch.log10((image - gt).square().mean().clamp_min(1e-30))),
            ssim=float(ssim(image[None], gt[None], data_range=1.)),
            lpips=float(perceptual(image[None] * 2 - 1, gt[None] * 2 - 1)))
    for spec in SCENES[args.group * 2:args.group * 2 + 2]:
        _, source = load_source(spec['slug'])
        n = len(source.scene.views)
        views = [int(.58 * n), int(.58 * n) + 1, int(.23 * n), int(.81 * n)]
        assert not set(views).intersection(frame for _, frame in motions(source))
        full, full_ids, _ = p.request(source.runtime, None, 'full')
        reference = BaselineRenderer()
        candidate = TailStrongRenderer(mode) if mode in [301, 302] else TailCutoffRenderer(203) if mode == 203 else EncodedSignRenderer(182)
        for condition in CONDITIONS:
            count = min(len(full_ids), int(condition[5:])) if condition.startswith('fixed') else len(full_ids) if condition == 'full' else int(float(condition[5:]) * len(full_ids))
            packet, ids = nested_packet(full, (count + .01) / max(1, len(full_ids)))
            for frame in views:
                camera, bg = source.camera(frame), source.scene.background
                expected, radii = reference.render(camera, packet, bg)
                actual, other_radii = candidate.render(camera, packet, bg)
                assert bits_equal(radii, other_radii) and bool(torch.isfinite(actual).all())
                gt = source.scene.ground_truth(frame).clamp(0, 1)
                base, result = metrics(expected, gt), metrics(actual, gt)
                rows.append(dict(scene=spec['id'], workload='nested', condition=condition, frame=frame,
                    K=len(ids), N=source.runtime.total, psnr_loss=base['psnr'] - result['psnr'],
                    baseline_psnr=base['psnr'], baseline_ssim=base['ssim'], baseline_lpips=base['lpips'],
                    radii_bitwise=True, **result))
        print('UNSEEN_NESTED_QUALITY', spec['id'], flush=True)
        del source, full, full_ids, reference, candidate
        gc.collect()
        torch.cuda.empty_cache()
    append(out / 'quality.csv', rows)
    save(out / 'DONE.json', dict(passed=True, quality_rows=len(rows), no_images=True, no_student_required=True))


if __name__ == '__main__':
    main()
