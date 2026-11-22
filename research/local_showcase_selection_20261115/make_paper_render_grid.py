"""论文局部渲染图：四Ratio加最右侧GT，两行五列，保留完整画面。"""
from pathlib import Path
import base64
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'paper_render_grid_gt'
RATIOS=[1,5,10,30]
COLUMNS=[1,5,10,30,'GT']
GROUPS=[('Replica/room0/mono','Replica room0\n(Mono)',950),
        ('TUM_RGBD/freiburg2_xyz/rgbd','TUM fr2_xyz\n(RGB-D)',551)]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,
                         'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
                         'image.composite_image':False})
    evidence=json.loads((ROOT/'furnished_revision/FURNISHED_SELECTION.json').read_text(encoding='utf-8'))
    sources=[];arrays=[]
    for group,_,frame in GROUPS:
        selected=next(g for g in evidence['groups'] if g['id']==group)
        assert selected['frame']['rgb_index']==frame
        for column in COLUMNS:
            filename='GT.png' if column=='GT' else f'ratio_{column:02d}.png'
            path=ROOT/'furnished_revision/selected'/group/filename
            record=None if column=='GT' else next(r for r in selected['records'] if r['ratio']==column/100)
            if record is not None:
                assert record['selection_identical'] and max(record['abs_errors'].values())==0
            with Image.open(path) as image:
                data=np.array(image.convert('RGB'))
            arrays.append(data)
            source=dict(group=group,rgb_index=frame,column=str(column),
                                ratio=None if column=='GT' else column/100,
                                source=str(path.relative_to(ROOT)).replace('\\','/'),
                                sha256=sha(path),width=data.shape[1],height=data.shape[0])
            if record is not None:source.update(K=record['K'],**record['scores'])
            sources.append(source)
    width=8.9
    left,right,gap,top,bottom,row_gap=.350,.015,.025,.200,.015,.025
    cell_width=(width-left-right-(len(COLUMNS)-1)*gap)/len(COLUMNS)
    row_heights=[cell_width*arrays[index].shape[0]/arrays[index].shape[1] for index in [0,len(COLUMNS)]]
    height=top+sum(row_heights)+row_gap+bottom
    figure=plt.figure(figsize=(width,height),facecolor='white')
    for line,(_,label,_) in enumerate(GROUPS):
        y=height-top-sum(row_heights[:line+1])-line*row_gap
        figure.text(.165/width,(y+row_heights[line]/2)/height,label,
                    rotation=90,ha='center',va='center',fontsize=9,linespacing=1.1)
        for column,value in enumerate(COLUMNS):
            x=left+column*(cell_width+gap)
            ax=figure.add_axes([x/width,y/height,cell_width/width,row_heights[line]/height])
            ax.imshow(arrays[line*len(COLUMNS)+column],interpolation='none',aspect='equal')
            ax.set_axis_off()
            if line==0:
                figure.text((x+cell_width/2)/width,(height-.082)/height,'GT' if value=='GT' else f'{value}%',
                            ha='center',va='center',fontsize=10)
    figure.canvas.draw()
    for text in figure.texts:
        bounds=text.get_window_extent(figure.canvas.get_renderer())
        assert bounds.x0>=0 and bounds.y0>=0 and bounds.x1<=figure.bbox.width and bounds.y1<=figure.bbox.height
    stem=OUTPUT/'LOCAL_RENDER_2X5_GT'
    figure.savefig(stem.with_suffix('.png'),dpi=600,facecolor='white')
    figure.savefig(stem.with_suffix('.svg'),facecolor='white')
    figure.savefig(stem.with_suffix('.pdf'),facecolor='white')
    figure.savefig(OUTPUT/'LOCAL_RENDER_2X5_GT_PREVIEW.png',dpi=200,facecolor='white')
    plt.close(figure)
    # SVG逐图解码，验证嵌入的10张图与源PNG像素完全相同，未裁剪或调色。
    tree=ET.parse(stem.with_suffix('.svg'))
    embedded=tree.findall('.//{http://www.w3.org/2000/svg}image')
    assert len(embedded)==10
    for element,original in zip(embedded,arrays):
        href=element.attrib['{http://www.w3.org/1999/xlink}href']
        data=base64.b64decode(href.split(',',1)[1])
        assert np.array_equal(np.array(Image.open(io.BytesIO(data)).convert('RGB')),original)
    assert all(sha(ROOT/s['source'])==s['sha256'] for s in sources)
    manifest=dict(complete=True,layout='2 rows x 5 columns',ratios=RATIOS,columns=COLUMNS,sources=sources,
                  native_aspect_preserved=True,full_images=True,svg_embedded_pixels_identical=True,
                  crop_or_color_adjustment=False,detail_insets=False,png_dpi=600,
                  size_inches=[width,height],files={p.name:sha(p) for p in OUTPUT.iterdir() if p.suffix in ['.png','.svg','.pdf']})
    (OUTPUT/'FIGURE_VERIFIED.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUTPUT/'data-manifest.md').write_text('# 论文渲染图来源\n\n'
        '两行五列：第一行Replica room0 Mono第950帧，第二行TUM freiburg2_xyz RGB-D第551帧（索引从0计数）。'
        '横向1%、5%、10%、30%局部渲染，最右列为该视角的GT原图。\n\n'
        '图片来源及SHA256、原始尺寸、实际K、PSNR／SSIM／LPIPS见FIGURE_VERIFIED.json。'
        '真实实验数据，同一行使用同一视角。没有裁剪、放大细节或调色；原始宽高比保留，SVG内嵌图像像素与源PNG完全相同。\n\n'
        'PNG：600 DPI排版；PDF／SVG：矢量文字与原生栅格图像。'
        '两场景原始宽高比不同，因此两行高度不同，五列宽度一致。\n\n'
        '脚本：../make_paper_render_grid.py。未改动论文正文、实验模型或已有结果。\n',encoding='utf-8')
    archive=OUTPUT/'PAPER_RENDER_2X5_GT.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in OUTPUT.iterdir():
            if path.is_file() and path!=archive:z.write(path,path.name)
        z.write(ROOT/'make_paper_render_grid.py','make_paper_render_grid.py')
    print('PAPER_GRID_VERIFIED',json.dumps(dict(png=str(stem.with_suffix('.png')),size_inches=[width,height],embedded_images=10)))


if __name__=='__main__':main()
