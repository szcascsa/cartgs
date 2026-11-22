"""从原分辨率PNG生成10张四栏对比图和两张浏览总览。"""
from pathlib import Path
import csv
import zipfile
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'artifacts'
FONT='C:/Windows/Fonts/msyh.ttc'
METHODS=[('GT','GT'),('Full','Full'),('Global_0.3','全局 ratio=0.3'),('Local_0.3','局部 ratio=0.3')]


def text(draw,xy,value,size=28,fill='#1e293b'):
    draw.text(xy,value,font=ImageFont.truetype(FONT,size),fill=fill)


def caption(row,index):
    if index==0:return '对应原始RGB'
    prefix=['','full','global','local'][index]
    return 'PSNR %.2f dB  |  K=%s'%(float(row[prefix+'_psnr']),row[prefix+'_K'])


def main():
    rows=list(csv.DictReader((DATA/'CAMERAS_AND_METRICS.csv').open()))
    assert len(rows)==10
    output=ROOT/'comparisons';output.mkdir(exist_ok=True)
    for row in rows:
        canvas=Image.new('RGB',(4870,871),'#ffffff');draw=ImageDraw.Draw(canvas)
        text(draw,(16,8),'Replica office1 · Mono · 随机视角%s · frame%06d'%(row['view'],int(row['gt_index'])),36)
        text(draw,(16,59),'相机位置 (%.5f, %.5f, %.5f)  |  原图 1200×680'%(float(row['camera_x']),float(row['camera_y']),float(row['camera_z'])),25)
        for index,(folder,label) in enumerate(METHODS):
            x=14+index*1214
            text(draw,(x,101),label,28);text(draw,(x,137),caption(row,index),23)
            im=Image.open(DATA/'images'/folder/(row['image_stem']+'.png')).convert('RGB')
            assert im.size==(1200,680)
            canvas.paste(im,(x,177))
        canvas.save(output/(row['image_stem']+'_comparison.png'))
    for page in range(2):
        canvas=Image.new('RGB',(1970,2016),'#f1f5f9');draw=ImageDraw.Draw(canvas)
        text(draw,(14,8),'Replica office1 单目：随机10个位姿（%d/2）'%(page+1),32)
        text(draw,(14,52),'同一baseline、同一相机位姿；GT / Full / 全局0.3 / 局部0.3',23)
        for offset,row in enumerate(rows[page*5:page*5+5]):
            y=96+offset*380
            draw.rectangle((8,y,1962,y+368),fill='white')
            text(draw,(14,y+3),'视角%02d · frame%06d · 相机位置(%.3f, %.3f, %.3f)'%(int(row['view']),int(row['gt_index']),float(row['camera_x']),float(row['camera_y']),float(row['camera_z'])),22)
            for index,(folder,label) in enumerate(METHODS):
                x=13+index*489
                text(draw,(x,y+38),label,22);text(draw,(x,y+67),caption(row,index),20)
                im=Image.open(DATA/'images'/folder/(row['image_stem']+'.png')).convert('RGB')
                im=im.resize((480,272),Image.Resampling.LANCZOS)
                canvas.paste(im,(x,y+96))
        canvas.save(ROOT/('overview_%d.png'%(page+1)))
    with zipfile.ZipFile(ROOT/'Replica_office1_mono_10views.zip','w',zipfile.ZIP_DEFLATED) as z:
        for base in [DATA,output]:
            for f in base.rglob('*'):
                if f.is_file():z.write(f,str(f.relative_to(ROOT)))
        for name in ['overview_1.png','overview_2.png']:z.write(ROOT/name,name)
    print('COMPARISONS_VERIFIED',len(list(output.glob('*.png'))),'ORIGINALS',len(list((DATA/'images').rglob('*.png'))))
    print('ZIP_BYTES',(ROOT/'Replica_office1_mono_10views.zip').stat().st_size)


if __name__=='__main__':main()
