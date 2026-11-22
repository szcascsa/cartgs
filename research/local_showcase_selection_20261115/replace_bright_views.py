"""将两组暗光Replica展示图换为office4明亮视角，保留原图和历史结果。"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'bright_revision'
RATIOS=[1,5,10,30]


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_row(figure,grid,export):
    folder=OUT/'selected'/export['id'];record=export['frame']
    names=['GT.png']+[f'ratio_{n:02d}.png' for n in RATIOS]
    for i,name in enumerate(names):
        ax=figure.add_subplot(grid[i]);ax.axis('off')
        ax.imshow(Image.open(folder/name),interpolation='nearest')
        label='GT（原图）' if i==0 else f"{RATIOS[i-1]}%   PSNR {record['psnr_'+str(RATIOS[i-1])]:.2f} dB\nK={record['K_'+str(RATIOS[i-1])]:,}"
        ax.set_title(label,fontsize=10,pad=6,color='#243441')


def title(export):
    return export['id'].replace('TUM_RGBD','TUM').replace('/rgbd',' · RGB-D').replace('/mono',' · Mono').replace('/',' · ')


def main():
    original=load(ROOT/'reports/EXPORT_VERIFIED.json')
    mono=load(ROOT/'images/alternatives/Replica/office4/mono/RENDER_VERIFIED.json')
    assert mono['checkpoint_unchanged']
    frame=mono['candidates'][0]
    replacement=dict(id=mono['group'],label='alternatives',frame=frame,
                     records=[r for r in mono['records'] if r['rgb_index']==frame['rgb_index']])
    choices=[next(e for e in original['exports'] if e['id']=='Replica/office4/rgbd'),replacement,
             next(e for e in original['exports'] if e['id']=='TUM_RGBD/freiburg2_xyz/rgbd'),
             next(e for e in original['exports'] if e['id']=='TUM_RGBD/freiburg2_xyz/mono')]
    OUT.mkdir(exist_ok=True);figures=OUT/'figures';figures.mkdir(exist_ok=True)
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'svg.fonttype':'none'})
    rows=[];source_hashes={};brightness=[]
    for export in choices:
        assert len(export['records'])==4
        assert all(r['selection_identical'] and max(r['abs_errors'].values())==0 for r in export['records'])
        source=ROOT/'selected'/export['label']/export['id']
        if export is replacement:
            source=ROOT/'images/alternatives'/export['id']/f"rgb_{frame['rgb_index']:06d}"
        destination=OUT/'selected'/export['id'];destination.mkdir(parents=True,exist_ok=True)
        for name in ['GT.png']+[f'ratio_{n:02d}.png' for n in RATIOS]:
            path=source/name;digest=sha(path);source_hashes[str(path.relative_to(ROOT)).replace('\\','/')]=digest
            shutil.copyfile(path,destination/name);assert sha(destination/name)==digest
        rgb=np.asarray(Image.open(destination/'GT.png').convert('RGB'),dtype=np.float32)/255.
        y=rgb@np.array([.2126,.7152,.0722],dtype=np.float32)
        brightness.append(dict(id=export['id'],mean_luma=float(y.mean()),fraction_below_015=float((y<.15).mean())))
        for record in export['records']:
            rows.append(dict(id=export['id'],rgb_index=export['frame']['rgb_index'],ratio=record['ratio'],
                             N=record['N'],candidate_N=record['candidates'],K=record['K'],**record['scores']))
        figure=plt.figure(figsize=(16,3.6))
        grid=figure.add_gridspec(1,5,left=.012,right=.988,bottom=.05,top=.76,wspace=.035)
        draw_row(figure,[grid[0,i] for i in range(5)],export)
        figure.suptitle(f"{title(export)}   |   RGB索引 {export['frame']['rgb_index']}（从0计数）",y=.98,fontsize=15,fontweight='bold')
        figure.text(.5,.86,f"同一视角四档对比   ·   Master N={export['frame']['N']:,}   ·   局部候选池={export['frame']['candidates']:,}",ha='center',fontsize=9,color='#56636D')
        for ext in ['png','svg']:figure.savefig(figures/(export['id'].replace('/','__')+'.'+ext),dpi=450)
        plt.close(figure)
    with (OUT/'SELECTED_VIEW_METRICS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    figure=plt.figure(figsize=(16,11.8))
    grid=figure.add_gridspec(4,5,left=.012,right=.988,bottom=.052,top=.86,hspace=.55,wspace=.035)
    for line,export in enumerate(choices):
        draw_row(figure,[grid[line,i] for i in range(5)],export)
        position=grid[line,0].get_position(figure)
        figure.text(.015,position.y1+.047,f"{title(export)}   |   RGB索引 {export['frame']['rgb_index']}（从0计数）",fontsize=11,fontweight='bold')
    figure.suptitle('局部渲染：Replica／TUM明亮视角展示',y=.983,fontsize=20,fontweight='bold')
    figure.text(.5,.943,'Replica两组改用office4明亮视角；TUM两组保留。每行1%／5%／10%／30%共用同一相机。',ha='center',fontsize=11,color='#56636D')
    figure.text(.5,.016,'原始渲染图未调亮、未修复。Ratio按名义Anchor视锥候选池计算；K为实际选点数。此图为精选案例，不代表数据集均值。',ha='center',fontsize=9,color='#56636D')
    for ext in ['png','svg']:figure.savefig(figures/('BRIGHT_LOCAL_FOUR_GROUPS.'+ext),dpi=450)
    figure.savefig(figures/'BRIGHT_LOCAL_FOUR_GROUPS_PREVIEW.png',dpi=140);plt.close(figure)
    receipt=dict(complete=True,groups=choices,brightness=brightness,source_image_sha256=source_hashes,
                 images_unmodified=True,selection_and_metrics_match_history=True,
                 replaced=['Replica/office1/rgbd','Replica/office1/mono'])
    (OUT/'BRIGHT_SELECTION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 明亮视角替换版','',
           'Replica RGB-D改为office4第825帧，Replica Mono改为office4第774帧；TUM两组保持原视角。索引从0计数。'
           '这是按明亮画面重新选择的展示案例，不再称四类绝对PSNR最优。图片直接复用或补渲染冻结模型，未调亮或改变像素。','',
           '| 场景／模式 | RGB帧 | 1% PSNR | 5% PSNR | 10% PSNR | 30% PSNR |','|---|---:|---:|---:|---:|---:|']
    for export in choices:
        f=export['frame'];values=' | '.join(f"{f['psnr_'+str(n)]:.2f}" for n in RATIOS)
        lines.append(f"| {export['id']} | {f['rgb_index']} | {values} |")
    lines+=['','每组selected/目录包含GT.png及ratio_01／05／10／30.png原尺寸图片；PSNR来自保存PNG前的FP32渲染。'
             '指标、实际K和选点hash与原评价完全一致，详见BRIGHT_SELECTION.json和SELECTED_VIEW_METRICS.csv。', '',
             'Replica两组GT的平均显示亮度分别为'+ '、'.join(f"{b['mean_luma']:.3f}" for b in brightness[:2])+'（0—1，BT.709加权RGB）；这是原图统计，不是调色。']
    (OUT/'README_明亮视角.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (figures/'data-manifest.md').write_text('# 数据来源\n\n指标：../SELECTED_VIEW_METRICS.csv；原图：../selected/；来源与选点核验：../BRIGHT_SELECTION.json。\n\n真实实测，非模拟数据；脚本：../../replace_bright_views.py。\n',encoding='utf-8')
    archive=ROOT/'LOCAL_BRIGHT_VIEWS_RESULTS.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in OUT.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(ROOT))
        z.write(ROOT/'replace_bright_views.py','replace_bright_views.py')
    current=dict(version='bright_revision',main_figure='bright_revision/figures/BRIGHT_LOCAL_FOUR_GROUPS.png',
                 selected='bright_revision/selected',zip_file=archive.name,zip_sha256=sha(archive))
    (ROOT/'CURRENT_SHOWCASE.json').write_text(json.dumps(current,indent=2)+'\n')
    for name,digest in source_hashes.items():assert sha(ROOT/name)==digest
    print('BRIGHT_REPLACEMENT_COMPLETE',json.dumps(current))


if __name__=='__main__':main()
