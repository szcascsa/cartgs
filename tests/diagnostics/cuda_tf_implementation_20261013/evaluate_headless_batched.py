"""逐帧语义的批量指标评估；只批量算指标，不抽帧、不改变renderer/TF。"""
import os
import sys
from pathlib import Path

import matplotlib
from evo.tools import settings
import torch

matplotlib.use("Agg")
settings.SETTINGS.plot_backend = "Agg"

class MetricValue:
    def __init__(self, owner):
        self.owner = owner
        self.value = None

    def item(self):
        return self

    def __float__(self):
        if self.value is None:
            self.owner.flush()
        return self.value

class BatchedMetric:
    def __init__(self, metric):
        self.metric = metric
        self.pending = []

    def __call__(self, prediction, target):
        value = MetricValue(self)
        self.pending.append((prediction.detach(), target.detach(), value))
        if len(self.pending) == 8:
            self.flush()
        return value

    @torch.no_grad()
    def flush(self):
        if not self.pending:
            return
        prediction = torch.cat([row[0] for row in self.pending])
        target = torch.cat([row[1] for row in self.pending])
        values = self.metric(prediction, target).reshape(-1).cpu().tolist()
        assert len(values) == len(self.pending)
        for row, value in zip(self.pending, values):
            row[2].value = float(value)
        self.pending.clear()

def main():
    source = Path(__file__).resolve().parents[3] / "python" / "run.py"
    sys.path.insert(0, str(source.parent))
    sys.argv[0] = str(source)
    code = source.read_text()
    replacements = {
        'calc_psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()':
            'calc_psnr = BatchedMetric(PeakSignalNoiseRatio(data_range=1.0, dim=(1,2,3), reduction="none").cuda())',
        'calc_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()':
            'calc_ssim = BatchedMetric(StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").cuda())',
        'calc_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()':
            'calc_lpips = BatchedMetric(LearnedPerceptualImagePatchSimilarity(normalize=True, reduction="none").cuda())',
    }
    for metric in ("psnr", "ssim", "lpips"):
        replacements[f'{metric}_list = np.array({metric}_list)'] = f'{metric}_list = np.array([float(value) for value in {metric}_list])'
    for old, new in replacements.items():
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    # 图像不交付；跳过写盘不影响render或逐帧指标。
    code = code.replace("import cv2\n", "import cv2\ncv2.imwrite = lambda *args, **kwargs: True\n")
    limit = int(os.environ.get("CARTGS_EVAL_PARITY_FRAMES", "0"))
    if limit:
        needle = "associations = associate_frames(tstamp, gt_tstamp)"
        assert code.count(needle) == 1
        code = code.replace(needle, needle + f"\n    associations = associations[:{limit}]")
    print(f"[Evaluation BATCH] batch=8 per_image_reduction=none parity_frame_limit={limit}", flush=True)
    exec(compile(code, str(source), "exec"), {"__name__": "__main__", "__file__": str(source),
         "BatchedMetric": BatchedMetric})

if __name__ == "__main__":
    main()
