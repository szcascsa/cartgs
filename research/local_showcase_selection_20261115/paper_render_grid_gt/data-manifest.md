# 论文渲染图来源

两行五列：第一行Replica room0 Mono第950帧，第二行TUM freiburg2_xyz RGB-D第551帧（索引从0计数）。横向1%、5%、10%、30%局部渲染，最右列为该视角的GT原图。

图片来源及SHA256、原始尺寸、实际K、PSNR／SSIM／LPIPS见FIGURE_VERIFIED.json。真实实验数据，同一行使用同一视角。没有裁剪、放大细节或调色；原始宽高比保留，SVG内嵌图像像素与源PNG完全相同。

PNG：600 DPI排版；PDF／SVG：矢量文字与原生栅格图像。两场景原始宽高比不同，因此两行高度不同，五列宽度一致。

脚本：../make_paper_render_grid.py。未改动论文正文、实验模型或已有结果。
