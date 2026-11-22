"""冻结 Anchor v2：复用共同 8 帧，保存 Full/四档 LoD 图片，不训练。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

repo, dataset = map(Path, sys.argv[1:3])
root = repo.parent
result = root/"online-fr1_xyz"
checkpoint = result/"6828_shutdown/ply"
frames = root/"validation/evaluation_frames.json"
output_root = root/"anchor_render_images"
output_root.mkdir(exist_ok=False)
model_files = [checkpoint/"selector.pt", checkpoint/"selector_config.yaml", checkpoint/"transform_field.pt",
               checkpoint/"point_cloud/iteration_6828/point_cloud.ply",
               checkpoint/"point_cloud/iteration_6828/selector_metadata.ply",
               result/"CameraTrajectory_TUM.txt", frames]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {str(path): sha(path) for path in model_files}
summary = []
for arm, ratio in (("Full", None), ("LOD_1", .01), ("LOD_5", .05), ("LOD_10", .1), ("LOD_15", .15)):
    output = output_root/arm
    output.mkdir()
    launcher = ("from evo.tools import settings; settings.SETTINGS.plot_backend='Agg'; "
                "import sys,runpy; sys.path.insert(0,"+repr(str(repo/"python"))+"); "
                "sys.argv=["+repr(str(repo/"python/run.py"))+"]+sys.argv[1:]; "
                "runpy.run_path(sys.argv[0],run_name='__main__')")
    command = [sys.executable, "-c", launcher, result, dataset, "--output_path", output,
               "--evaluation-frame-set", frames]
    if ratio is not None:
        command += ["--selector", checkpoint/"selector.pt", "--selector-ratio", str(ratio),
                    "--selector-config", checkpoint/"selector_config.yaml"]
    with (output/"evaluation.log").open("w") as log:
        subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, OMP_NUM_THREADS="2", MPLBACKEND="Agg"), check=True)
    evaluated = json.loads((output/"evaluation_frames.json").read_text())
    assert evaluated["gt_indices"] == json.loads(frames.read_text())["gt_indices"]
    assert len(list((output/"image").glob("*.png"))) == len(evaluated["gt_indices"]) == 8
    stats = json.loads((output/"selector_stats.json").read_text())
    old = root/"quality_smoke"/arm
    assert stats["active_gaussians"] == json.loads((old/"selector_stats.json").read_text())["active_gaussians"]
    for metric in ("psnr", "ssim", "lpips"):
        assert (output/(metric+".txt")).read_bytes() == (old/(metric+".txt")).read_bytes(), (arm, metric)
    summary.append(dict(arm=arm, ratio=ratio, images=8, gaussians=stats["active_gaussians"], metrics_exact=True))
    print(arm, stats["active_gaussians"], "8 images; metrics exact", flush=True)
gt = output_root/"GT"
gt.mkdir()
for image in evaluated["rgb_paths"]:
    shutil.copyfile(image, gt/Path(image).name)
after = {str(path): sha(path) for path in model_files}
assert before == after
(output_root/"render_manifest.json").write_text(json.dumps(dict(
    scene="fr1_xyz", checkpoint=str(checkpoint), frames=evaluated,
    arms=summary, unchanged_model_sha256=before, renders=40, ground_truths=8), indent=2))
print("Complete: 40 renders + 8 GT; source checkpoint unchanged", flush=True)
