"""筛查Replica明亮、有纹理的候选原图，最终由原图可视核查家具和物体覆盖。"""
from pathlib import Path
import csv
import json

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent


def main():
    with (ROOT/'reports/ALL_FRAME_RANKING.csv').open(encoding='utf-8-sig') as stream:
        rows=[r for r in csv.DictReader(stream) if r['dataset']=='Replica']
    scenes=sorted({r['scene'] for r in rows})
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    board=Image.new('RGB',(2160,8*255),'white');draw=ImageDraw.Draw(board);selected=[]
    for line,scene in enumerate(scenes):
        values=[r for r in rows if r['scene']==scene and r['mode']=='rgbd' and int(r['rgb_index'])%25==0]
        candidates=[]
        for row in values:
            bgr=cv2.imread(row['rgb_path']);rgb=cv2.resize(bgr[:,:,::-1],(360,204),interpolation=cv2.INTER_AREA)/255.
            gray=(rgb@np.array([.2126,.7152,.0722])).astype(np.float32)
            gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
            edge=np.sqrt(gx*gx+gy*gy)>.15
            texture=float(edge.mean());coverage=float((edge.reshape(6,34,6,60).mean(axis=(1,3))>.06).mean())
            item=dict(row,brightness=float(gray.mean()),dark_fraction=float((gray<.15).mean()),
                      edge_density=texture,edge_coverage=coverage)
            if item['brightness']>=.32 and item['dark_fraction']<.3:
                candidates.append(item)
        rich=sorted(candidates,key=lambda r:r['edge_density']*r['edge_coverage'],reverse=True)
        quality=sorted([r for r in candidates if r['edge_density']>.12 and r['edge_coverage']>.6],
                       key=lambda r:float(r['mean_psnr']),reverse=True)
        chosen=[]
        for ranking,limit in [(rich,3),(quality,6)]:
            for row in ranking:
                if all(abs(int(row['rgb_index'])-int(x['rgb_index']))>=90 for x in chosen):chosen.append(row)
                if len(chosen)>=limit:break
        chosen=chosen[:6]
        for col,row in enumerate(chosen):
            image=Image.open(row['rgb_path']).convert('RGB');image.thumbnail((350,198))
            x=col*360;y=line*255;board.paste(image,(x,y+49))
            draw.text((x+4,y+2),f"{scene}  RGB {row['rgb_index']}",font=font,fill='black')
            draw.text((x+4,y+24),f"L={row['brightness']:.2f} P={float(row['mean_psnr']):.1f} E={row['edge_density']:.2f}",font=font,fill='black')
            selected.append(row)
    output=ROOT/'furnished_preview';output.mkdir(exist_ok=True)
    board.save(output/'CANDIDATE_GT_BOARD.jpg',quality=95)
    (output/'CANDIDATES.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    print('CANDIDATES',len(selected),flush=True)


if __name__=='__main__':main()
