"""原分辨率像素拼接；标注原FP32渲染PSNR和每帧输入K，不改写渲染图。"""
from pathlib import Path
import base64
import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('H:/AIProjects/cartgs优化更细致版/outputs/fr2_xyz_rgbd_global_local_full_20261116')
FOLDERS = ['Global_0.5', 'Local_0.5', 'Full', 'GT']
LABELS = ['全局Ratio 0.5', '局部Ratio 0.5', 'Full', '原始RGB参考图']
COLORS = ['#BC4851', '#319B80', '#41566A', '#41566A']
FONT = 'C:/Windows/Fonts/msyh.ttc'
SVG_NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG_NS)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(parent, x, y, message, size, color='#26343F', bold=False):
    child = ET.SubElement(parent, f'{{{SVG_NS}}}text',
        x=str(x),y=str(y),fill=color,style=f'font-family:Microsoft YaHei;font-size:{size}px;font-weight:{700 if bold else 400}')
    child.text = message


def draw(rows, path):
    margin, gap, width, height, top, row_height = 24, 16, 640, 480, 105, 568
    canvas_width = 2*margin + 4*width + 3*gap
    canvas_height = top + len(rows)*row_height + 86
    canvas = Image.new('RGB', (canvas_width,canvas_height),'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT,24)
    heading = ImageFont.truetype(FONT,32)
    note = ImageFont.truetype(FONT,22)
    draw.text((margin,14),'TUM freiburg2_xyz RGB-D：全局0.5／局部0.5／Full',font=heading,fill='#26343F')
    draw.text((margin,60),'同一checkpoint、相同相机位姿；每格640×480原图，无裁剪与调色。',font=font,fill='#52636F')
    svg = ET.Element(f'{{{SVG_NS}}}svg',width=str(canvas_width),height=str(canvas_height),viewBox=f'0 0 {canvas_width} {canvas_height}')
    ET.SubElement(svg,f'{{{SVG_NS}}}rect',width=str(canvas_width),height=str(canvas_height),fill='white')
    text(svg,margin,42,'TUM freiburg2_xyz RGB-D：全局0.5／局部0.5／Full',32,bold=True)
    text(svg,margin,87,'同一checkpoint、相同相机位姿；每格640×480原图，无裁剪与调色。',24,'#52636F')
    for row_index,row in enumerate(rows):
        y = top + row_index*row_height
        row_label = f"视角{row['view']} · RGB帧{row['rgb_index']}"
        draw.text((margin,y),row_label,font=font,fill='#26343F')
        text(svg,margin,y+25,row_label,24,bold=True)
        for index,(folder,label,color) in enumerate(zip(FOLDERS,LABELS,COLORS)):
            x = margin + index*(width+gap)
            if folder == 'GT':
                caption = label
            else:
                prefix = {'Global_0.5':'global','Local_0.5':'local','Full':'full'}[folder]
                count = int(row['full_N'] if prefix=='full' else row[prefix+'_K'])
                caption = f"{label}   PSNR {float(row[prefix+'_psnr']):.2f} dB   {'N' if prefix=='full' else 'K'}={count:,}"
            draw.text((x,y+34),caption,font=font,fill=color)
            text(svg,x,y+59,caption,24,color)
            source = ROOT/'images'/folder/row['image']
            image = Image.open(source).convert('RGB')
            assert image.size==(width,height)
            canvas.paste(image,(x,y+70))
            assert np.array_equal(np.asarray(canvas.crop((x,y+70,x+width,y+70+height))),np.asarray(image))
            ET.SubElement(svg,f'{{{SVG_NS}}}image',x=str(x),y=str(y+70),width=str(width),height=str(height),
                href='data:image/png;base64,'+base64.b64encode(source.read_bytes()).decode())
    footer = top+len(rows)*row_height
    for index,message in enumerate(['K为当前视角提交Renderer的点数；全局和局部Ratio预算分母不同。',
                                   'PSNR取原FP32渲染；Full为我们同一最终Master；5帧样本不代表全轨迹均值。']):
        draw.text((margin,footer+5+index*30),message,font=note,fill='#52636F')
        text(svg,margin,footer+30+index*30,message,22,'#52636F')
    canvas.save(path.with_suffix('.png'),dpi=(450,450))
    ET.ElementTree(svg).write(path.with_suffix('.svg'),encoding='utf-8',xml_declaration=True)


def main():
    verified = json.loads((ROOT/'VERIFIED.json').read_text(encoding='utf-8'))
    assert verified['complete'] and verified['views']==5 and verified['images']==20
    rows = list(csv.DictReader((ROOT/'FRAME_METRICS.csv').open(encoding='utf-8-sig')))
    assert len(rows)==5 and len(list((ROOT/'images').glob('*/*.png')))==20
    (ROOT/'figures').mkdir(exist_ok=True)
    (ROOT/'figures/data-manifest.md').write_text('# 数据清单\n\n'
        '|图|数据|性质|来源|输出|\n|---|---|---|---|---|\n'
        '|五视角总览及每视角对比|FRAME_METRICS.csv、images/|真实|冻结历史checkpoint选帧复验|PNG、SVG|\n\n'
        '五帧在有效轨迹10%、30%、50%、70%、90%分位预选，不按PSNR筛选；拼图保留原分辨率图像像素。\n',encoding='utf-8')
    draw(rows,ROOT/'figures/FR2_XYZ_RGBD_FIVE_VIEWS')
    for row in rows:
        draw([row],ROOT/f"figures/VIEW_{int(row['view']):02d}_FRAME_{int(row['rgb_index']):06d}")
    (ROOT/'README_渲染说明.md').write_text('# TUM freiburg2_xyz RGB-D五视角渲染\n\n'
        '使用与此前指标表相同的双网络真实视角＋LI最终模型，Full地图15634点。'
        '从3615个有效评价视角按轨迹10%、30%、50%、70%、90%分位选择RGB帧361、1099、1825、2562、3307，未按质量筛选。\n\n'
        '全局0.5复用原全图选点和零视角条件，实际保留7818点；局部0.5使用名义Anchor视锥候选池、严格预算和双网络真实视角；'
        'Full是同一最终Master未经选择与补偿的完整渲染。三者相机位姿完全相同，GT为原始RGB，640×480，无裁剪、调色或位姿优化。\n\n'
        '15次渲染的PSNR、SSIM、LPIPS与历史逐帧指标完全一致；全局选点ID、局部逐帧ID哈希匹配；84项输入哈希不变。'
        '指标在FP32图像上计算，PNG采用原评价的8位写出方式。当前图标注的是逐帧输入K，整段轨迹去重总数U仍见此前场景表，不能从这五帧推算全轨迹U。\n\n'
        'figures/含五视角总览和每视角单独对比，PNG及SVG；images/含20张原分辨率PNG；FRAME_METRICS.csv含逐帧三项质量与点数。'
        'full_0、global_0.5、local_0.5的JSON保存完整c2w/w2c位姿；EXPERIMENT.json保存源checkpoint、评价路径与抽帧依据。\n',encoding='utf-8')
    receipt = dict(complete=True,source_metrics_unchanged=True,views=5,raw_images=20,
        figures=6,png_and_svg=True,rendered_cells_pixel_equal_to_raw=True,render_parity=verified,
        files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix!='.gz'
               and p.name not in ['DELIVERY_VERIFIED.json','FR2_XYZ_RGBD_RENDERS.zip']})
    (ROOT/'DELIVERY_VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(ROOT/'FR2_XYZ_RGBD_RENDERS.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file() and p.suffix not in ['.gz','.zip']:
                archive.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(ROOT/'FR2_XYZ_RGBD_RENDERS.zip') as archive:
        assert archive.testzip() is None
    print('COMPARISON_VERIFIED',len(rows),20,6,flush=True)


if __name__=='__main__':
    main()
