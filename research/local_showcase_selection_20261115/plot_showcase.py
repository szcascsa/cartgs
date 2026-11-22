"""真实原尺寸渲染图片的同视角对比排版，原始PNG另行保留。"""
from pathlib import Path
import hashlib
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT=Path(__file__).resolve().parent
RATIOS=[1,5,10,30]


def group_images(export):
    folder=ROOT/'selected'/export['label']/export['id']
    return [folder/'GT.png']+[folder/f'ratio_{n:02d}.png' for n in RATIOS]


def draw_row(fig,slots,export):
    record=export['frame']
    for index,path in enumerate(group_images(export)):
        ax=fig.add_subplot(slots[index]);ax.axis('off')
        ax.imshow(Image.open(path),interpolation='nearest')
        if index==0:title='GT（原图）'
        else:
            n=RATIOS[index-1]
            title=f"{n}%   PSNR {record['psnr_'+str(n)]:.2f} dB\nK={record['K_'+str(n)]:,}"
        ax.set_title(title,fontsize=10,pad=6,color='#243441')


def main():
    data=json.loads((ROOT/'reports/EXPORT_VERIFIED.json').read_text(encoding='utf-8'))
    assert data['complete'] and data['historical_metric_max_abs_error']==0
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'svg.fonttype':'none'})
    output=ROOT/'figures';output.mkdir(exist_ok=True)
    best=sorted([e for e in data['exports'] if e['label']=='best'],
                key=lambda e:(0 if e['id'].startswith('Replica') else 1,0 if e['id'].endswith('rgbd') else 1))
    for export in data['exports']:
        row=export['frame'];figure=plt.figure(figsize=(16,3.6))
        grid=figure.add_gridspec(1,5,left=.012,right=.988,bottom=.05,top=.76,wspace=.035)
        draw_row(figure,[grid[0,i] for i in range(5)],export)
        title=export['id'].replace('TUM_RGBD','TUM').replace('/rgbd',' · RGB-D').replace('/mono',' · Mono').replace('/',' · ')
        figure.suptitle(f"{title}   |   RGB索引 {row['rgb_index']}（从0计数）",y=.98,fontsize=15,fontweight='bold')
        figure.text(.5,.86,f"同一视角四档平均PSNR {row['mean_psnr']:.2f} dB   ·   Master N={row['N']:,}   ·   局部候选池={row['candidates']:,}",
                    ha='center',fontsize=9,color='#56636D')
        name=export['label']+'__'+export['id'].replace('/','__')
        for extension in ['png','svg']:figure.savefig(output/(name+'.'+extension),dpi=450)
        plt.close(figure)
    figure=plt.figure(figsize=(16,11.8))
    grid=figure.add_gridspec(4,5,left=.012,right=.988,bottom=.052,top=.86,hspace=.55,wspace=.035)
    for line,export in enumerate(best):
        draw_row(figure,[grid[line,i] for i in range(5)],export)
        row=export['frame'];cell=grid[line,0].get_position(figure)
        title=export['id'].replace('TUM_RGBD','TUM').replace('/rgbd',' · RGB-D').replace('/mono',' · Mono').replace('/',' · ')
        figure.text(.015,cell.y1+.047,f"{title}   |   RGB索引 {row['rgb_index']}（从0计数）",fontsize=11,fontweight='bold')
    figure.suptitle('局部渲染：Replica／TUM各模式的最佳共用视角',y=.983,fontsize=20,fontweight='bold')
    figure.text(.5,.943,'26组、49,740个视角筛选；按同一视角1%／5%／10%／30%平均PSNR选取',ha='center',fontsize=11,color='#56636D')
    figure.text(.5,.016,'精选最佳案例，非数据集平均表现。Ratio以名义Anchor视锥候选池为预算分母；K为实际选点数。原图未作亮度增强或修复。',
                ha='center',fontsize=9,color='#56636D')
    for extension in ['png','svg']:figure.savefig(output/('BEST_LOCAL_FOUR_GROUPS.'+extension),dpi=450)
    figure.savefig(output/'BEST_LOCAL_FOUR_GROUPS_PREVIEW.png',dpi=140)
    plt.close(figure)
    manifest=['# 数据来源','',
              '| 图 | 指标与筛选 | 图片来源 | 数据类别 | 脚本 |','|---|---|---|---|---|',
              '| 四组最佳视角及两组明亮备选 | reports/SELECTED_VIEW_METRICS.csv、RANKING.json | selected/best、selected/alternatives原尺寸PNG | 真实实测，无模拟数据 | plot_showcase.py |','',
              '每组四档相同相机，所有指标和选点身份与历史评价匹配。PNG／SVG为排版文件，源图片保留在selected/；指标来自保存8位PNG前的FP32渲染。']
    (output/'data-manifest.md').write_text('\n'.join(manifest)+'\n',encoding='utf-8')
    archive_path=ROOT/'LOCAL_BEST_VIEWS_RESULTS.zip';temporary=ROOT/'LOCAL_BEST_VIEWS_RESULTS.tmp.zip'
    extras={str(path.relative_to(ROOT)).replace('\\','/'):path for path in output.iterdir()}
    extras['plot_showcase.py']=ROOT/'plot_showcase.py'
    with zipfile.ZipFile(archive_path) as old,zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as new:
        for item in old.infolist():
            if item.filename not in extras:new.writestr(item,old.read(item.filename))
        for name,path in extras.items():new.write(path,name)
    temporary.replace(archive_path)
    hashes={str(path.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ['selected','figures','reports'] for path in (ROOT/folder).rglob('*') if path.is_file()}
    (ROOT/'DELIVERY_VERIFIED.json').write_text(json.dumps(dict(complete=True,files=hashes,
        zip_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest()),indent=2)+'\n')
    print('FIGURES_COMPLETE',output)


if __name__=='__main__':
    main()
