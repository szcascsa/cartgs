"""保留原分辨率渲染图，另生成带GT、预算和PSNR的浏览拼图。"""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
CASES = {
    'replica_room0_mono': 'Replica room0 · Mono · 输入帧950（第951帧）',
    'tum_fr2_xyz_rgbd': 'TUM fr2_xyz · RGB-D · 输入帧551（第552帧）',
}


def main():
    output = ROOT/'figures'
    output.mkdir(exist_ok=True)
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    fonts = {size: ImageFont.truetype(font_path,size) for size in [21,24,28,34]}
    all_rows = []
    for case,title in CASES.items():
        directory = ROOT/'runs'/case
        receipt = json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
        assert receipt['complete'] and receipt['source_unchanged'] and receipt['metrics_match_history']
        assert len(receipt['records']) == 4
        canvas = Image.new('RGB',(2604,680),'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((22,18),title,font=fonts[34],fill='#24333D')
        draw.text((22,70),f"CaRtGS按Opacity降序保留Top-K · Full地图N={receipt['records'][0]['N']:,} · 原图评价口径",
                  font=fonts[24],fill='#52636F')
        panels = [('GT',directory/'GT.png',None)]+[
            (f"保留{int(float(row['ratio'])*100)}%", directory/f"TOPK_{int(float(row['ratio'])*100):02d}pct.png",row)
            for row in receipt['records']]
        for index,(label,path,row) in enumerate(panels):
            x=20+index*516
            draw.text((x,124),label,font=fonts[28],fill='#24333D')
            if row:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == row['image_sha256']
                draw.text((x,164),f"K={row['K']:,}  |  PSNR={row['psnr']:.2f} dB",font=fonts[21],fill='#52636F')
                all_rows.append(row)
            with Image.open(path) as image:
                assert image.size == (receipt['width'],receipt['height'])
                preview=ImageOps.contain(image.convert('RGB'),(500,380),Image.Resampling.LANCZOS)
                canvas.paste(preview,(x+(500-preview.width)//2,210+(380-preview.height)//2))
        draw.text((22,612),'原始PNG保留完整分辨率；本拼图仅缩放预览，无裁切、亮度或颜色增强。',font=fonts[24],fill='#52636F')
        canvas.save(output/f'{case}_TOPK_COMPARISON.png')
    with (ROOT/'REQUESTED_FRAME_METRICS.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    assert len(all_rows) == 8 and max(r['history_max_abs_error'] for r in all_rows) < 2e-5
    (ROOT/'README_渲染说明.md').write_text('# CaRtGS Top-K指定视角渲染\n\n'
        'Replica room0 Mono：输入帧950，对应frame000950.jpg，原评价序号947。'
        'TUM fr2_xyz RGB-D：输入帧551，对应1311867188.831350.png，原评价序号551。输入与评价编号从0开始。\n\n'
        '分别导出保留1%、5%、10%、30%的结果。Top-K按激活后不透明度稳定降序排序，'
        'K=floor(ratio×CaRtGS自身Full地图点数N)，预算使用Decimal计算。直接复用已有selected_ids.npy，'
        '同时重新核对排序、ID哈希和对应checkpoint；不训练、不补偿、不修改高斯属性。\n\n'
        '复用CaRtGS原评价运行时、原估计相机轨迹和原始RGB图。Replica沿用单目尺度对齐，TUM RGB-D不做尺度校正。'
        '每张图的PSNR、SSIM、LPIPS均与此前保存的同帧同Ratio结果核验，最大误差小于2e-5。'
        '原源码、checkpoint、参考图与历史评价文件的哈希前后不变。\n\n'
        'runs目录包含8张原分辨率渲染PNG和2张GT。图像按原评价方式clamp[0,1]，乘255后截断为uint8，'
        '不改变色彩、亮度或相机；以无损PNG保存。figures目录为带GT、K和PSNR的缩略拼图。'
        'CSV保存完整精度指标及来源路径；没有重新计算全轨迹指标。\n',encoding='utf-8')
    verified=dict(complete=True,cases=2,original_render_pngs=8,gt_pngs=2,contact_sheets=2,
        max_historical_metric_error=max(r['history_max_abs_error'] for r in all_rows),
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ['runs','figures'] for p in (ROOT/folder).rglob('*') if p.is_file()})
    (ROOT/'VERIFIED.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=ROOT/'CARTGS_TOPK_REQUESTED_RENDERS.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipout:
        for folder in ['runs','figures']:
            for path in (ROOT/folder).rglob('*'):
                if path.is_file():zipout.write(path,path.relative_to(ROOT))
        for name in ['REQUESTED_FRAME_METRICS.csv','README_渲染说明.md','VERIFIED.json','render_requested.py','compose_results.py']:
            zipout.write(ROOT/name,name)
    with zipfile.ZipFile(archive) as zipin:
        assert zipin.testzip() is None and len(zipin.namelist())==len(set(zipin.namelist()))
    print('REQUESTED_RENDERS_PACKAGED',len(all_rows),'maximum_metric_error',verified['max_historical_metric_error'],flush=True)


if __name__ == '__main__':
    main()
