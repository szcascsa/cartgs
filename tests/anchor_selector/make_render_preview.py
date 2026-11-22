"""将真实渲染图排成对照预览；原图保持不变。"""
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont

root = Path(sys.argv[1])
manifest = json.loads((root/"render_manifest.json").read_text())
frames = manifest["frames"]["gt_indices"]
filenames = [Path(p).name for p in manifest["frames"]["rgb_paths"]]
counts = {arm["arm"]: arm["gaussians"] for arm in manifest["arms"]}
columns = [("GT", "GT"), ("Full", "Full"), ("LOD_15", "15%"),
           ("LOD_10", "10%"), ("LOD_5", "5%"), ("LOD_1", "1%")]
font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)
small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
previews = root/"previews"
previews.mkdir(exist_ok=True)
for folder, _ in columns:
    paths = list((root/folder if folder == "GT" else root/folder/"image").glob("*.png"))
    assert len(paths) == 8
    for path in paths:
        with Image.open(path) as image:
            assert image.size == (640, 480)
            image.verify()

def sheet(indices, width, filename):
    height, gap, left, top = width*3//4, 8, 90, 78
    canvas = Image.new("RGB", (left+6*(width+gap)+gap, top+len(indices)*(height+gap)+gap), "#151a22")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 5), "Anchor / fr1_xyz：固定 8 帧真实渲染，列为 GT、Full、15%、10%、5%、1%", font=small, fill="white")
    for column, (folder, label) in enumerate(columns):
        x = left+column*(width+gap)
        text = label if folder == "GT" else f"{label} · {counts[folder]:,} 点"
        draw.text((x, 40), text, font=font, fill="white")
        for row, index in enumerate(indices):
            path = root/folder/filenames[index] if folder == "GT" else root/folder/"image"/filenames[index]
            with Image.open(path) as image:
                canvas.paste(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), (x, top+row*(height+gap)))
            if column == 0:
                draw.text((8, top+row*(height+gap)+8), f"帧 {frames[index]}", font=small, fill="white")
    canvas.save(previews/filename)

sheet(range(4), 320, "anchor_comparison_1.png")
sheet(range(4, 8), 320, "anchor_comparison_2.png")
for index, frame in enumerate(frames):
    sheet([index], 640, f"frame_{frame}_comparison.png")
(root/"README.md").write_text("""# Anchor 渲染图片

场景 fr1_xyz；冻结 checkpoint：6828_shutdown；保留原有 Cached Bernoulli 选择及 TF，无训练或参数修改。

- Full、1%、5%、10%、15% 各 8 张，共 40 张 640×480 原始渲染 PNG。
- GT 文件夹含相同 8 帧真值图。
- previews/anchor_comparison_1.png 与 anchor_comparison_2.png 为全部帧的缩略对照；frame_*_comparison.png 保留每幅图的原始尺寸。
- 预览列顺序：GT、Full、15%、10%、5%、1%。预览仅缩放拼接，不改变原始渲染图片。
- 各档实际点数分别为 Full 24561、1% 284、5% 1217、10% 2493、15% 3728；Cached Bernoulli 并非 exact-K。
- 本次逐视图 PSNR/SSIM/LPIPS 与既有评估完全一致，源模型、TF、Selector、metadata、相机轨迹和帧集合的前后 SHA 一致。

服务器原图目录：`/home/prmi/zwj/scaffold-selector-refactor-20261020/anchor_render_images/`。
""", encoding="utf-8")
print("48 original images verified; 2 overview sheets and 8 full-resolution strips saved")
