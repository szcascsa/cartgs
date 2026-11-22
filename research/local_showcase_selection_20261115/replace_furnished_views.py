"""以room0明亮客厅替换空墙展示，固定原始相机、四Ratio和历史质量口径。"""
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

import replace_bright_views as layout

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'furnished_revision'
RATIOS=[1,5,10,30]


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    choices=[];inputs={};receipts={}
    for mode in ['rgbd','mono']:
        p=ROOT/f'images/furnished/Replica/room0/{mode}/RENDER_VERIFIED.json'
        data=load(p);inputs[str(p.relative_to(ROOT))]=sha(p);receipts[data['group']]=data
        assert data['checkpoint_unchanged']
        frame=next(x for x in data['candidates'] if x['rgb_index']==950)
        for key in frame:
            if key.startswith(('psnr_','ssim_','lpips_','mean_','min_')):frame[key]=float(frame[key])
            elif key.startswith('K_'):frame[key]=int(frame[key])
        records=[x for x in data['records'] if x['rgb_index']==950]
        choices.append(dict(id=data['group'],label='furnished',frame=frame,records=records,
                            source=str(p.parent/'rgb_000950')))
    original=load(ROOT/'reports/EXPORT_VERIFIED.json')
    for mode in ['rgbd','mono']:
        export=next(e for e in original['exports'] if e['id']==f'TUM_RGBD/freiburg2_xyz/{mode}')
        choices.append(dict(export,source=str(ROOT/'selected'/export['label']/export['id'])))
    OUT.mkdir(exist_ok=True);figures=OUT/'figures';figures.mkdir(exist_ok=True)
    # 复用本轮已经检查过的四Ratio排版函数，只改变其本轮输出目录。
    layout.OUT=OUT
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'svg.fonttype':'none'})
    rows=[];hashes={}
    for export in choices:
        assert len(export['records'])==4
        assert all(x['selection_identical'] and max(x['abs_errors'].values())==0 for x in export['records'])
        source=Path(export['source']);dest=OUT/'selected'/export['id'];dest.mkdir(parents=True,exist_ok=True)
        for name in ['GT.png','ratio_01.png','ratio_05.png','ratio_10.png','ratio_30.png']:
            path=source/name;digest=sha(path);hashes[str(path.relative_to(ROOT)).replace('\\','/')]=digest
            shutil.copyfile(path,dest/name);assert sha(dest/name)==digest
        for record in export['records']:
            rows.append(dict(id=export['id'],rgb_index=export['frame']['rgb_index'],ratio=record['ratio'],
                             N=record['N'],candidate_N=record['candidates'],K=record['K'],**record['scores']))
        fig=plt.figure(figsize=(16,3.6))
        grid=fig.add_gridspec(1,5,left=.012,right=.988,bottom=.05,top=.76,wspace=.035)
        layout.draw_row(fig,[grid[0,i] for i in range(5)],export)
        fig.suptitle(f"{layout.title(export)}   |   RGB索引 {export['frame']['rgb_index']}（从0计数）",y=.98,fontsize=15,fontweight='bold')
        fig.text(.5,.86,f"同一视角四档对比   ·   Master N={export['frame']['N']:,}   ·   局部候选池={export['frame']['candidates']:,}",ha='center',fontsize=9,color='#56636D')
        for ext in ['png','svg']:fig.savefig(figures/(export['id'].replace('/','__')+'.'+ext),dpi=450)
        plt.close(fig)
    fig=plt.figure(figsize=(16,11.8))
    grid=fig.add_gridspec(4,5,left=.012,right=.988,bottom=.052,top=.86,hspace=.55,wspace=.035)
    for line,export in enumerate(choices):
        layout.draw_row(fig,[grid[line,i] for i in range(5)],export)
        pos=grid[line,0].get_position(fig)
        fig.text(.015,pos.y1+.047,f"{layout.title(export)}   |   RGB索引 {export['frame']['rgb_index']}（从0计数）",fontsize=11,fontweight='bold')
    fig.suptitle('局部渲染：明亮且内容丰富的视角',y=.983,fontsize=20,fontweight='bold')
    fig.text(.5,.943,'Replica改用room0客厅：沙发、茶几、花瓶、靠垫及落地灯；TUM两组保留。每行四档共用同一相机。',ha='center',fontsize=10.5,color='#56636D')
    fig.text(.5,.016,'精选展示案例，非数据集平均值。原始图像未调亮或修复；Ratio按名义Anchor视锥候选池计算，K为实际选点数。',ha='center',fontsize=9,color='#56636D')
    for ext in ['png','svg']:fig.savefig(figures/('FURNISHED_LOCAL_FOUR_GROUPS.'+ext),dpi=450)
    fig.savefig(figures/'FURNISHED_LOCAL_FOUR_GROUPS_PREVIEW.png',dpi=140);plt.close(fig)
    with (OUT/'SELECTED_VIEW_METRICS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    payload=dict(complete=True,groups=choices,source_image_sha256=hashes,
                 source_receipt_sha256=inputs,checkpoints_unchanged=True,source_receipts=receipts,
                 selection_and_metrics_match_history=True,images_unmodified=True,
                 rationale='人工检查原图中的家具／物体覆盖，选择明亮且有沙发、茶几、花瓶、靠垫和落地灯的room0客厅；不声明全局质量最优。')
    (OUT/'FURNISHED_SELECTION.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 明亮且内容丰富的视角替换版','',payload['rationale'],'',
           'Replica RGB-D和Mono均选room0的RGB索引950（从0计数），分别使用各自模型及原估计相机；TUM两组保持。',
           '重新选择展示帧，不修改图像亮度、模型或训练；四档选点身份、K和三项质量指标与历史评价相同。','',
           '| 场景／模式 | RGB帧 | 1% PSNR | 5% PSNR | 10% PSNR | 30% PSNR |','|---|---:|---:|---:|---:|---:|']
    for export in choices:
        row=export['frame'];values=' | '.join(f"{row['psnr_'+str(n)]:.2f}" for n in RATIOS)
        lines.append(f"| {export['id']} | {row['rgb_index']} | {values} |")
    lines+=['','selected/内为16张原尺寸渲染PNG和4张GT。figures/内为单组和四组排版PNG／SVG。'
             '指标来自保存8位PNG前的FP32渲染；各自完整选点核验和来源见FURNISHED_SELECTION.json。']
    (OUT/'README_客厅替换版.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (figures/'data-manifest.md').write_text('# 数据来源\n\n真实实测：../SELECTED_VIEW_METRICS.csv。原图：../selected/。核验与来源：../FURNISHED_SELECTION.json。脚本：../../replace_furnished_views.py。\n',encoding='utf-8')
    archive=ROOT/'LOCAL_FURNISHED_VIEWS_RESULTS.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in OUT.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(ROOT))
        for name in ['replace_furnished_views.py','replace_bright_views.py']:z.write(ROOT/name,name)
    current=dict(version='furnished_revision',main_figure='furnished_revision/figures/FURNISHED_LOCAL_FOUR_GROUPS.png',
                 selected='furnished_revision/selected',zip_file=archive.name,zip_sha256=sha(archive))
    (ROOT/'CURRENT_SHOWCASE.json').write_text(json.dumps(current,indent=2)+'\n')
    assert all(sha(ROOT/name)==digest for name,digest in hashes.items())
    print('FURNISHED_REPLACEMENT_COMPLETE',json.dumps(current))


if __name__=='__main__':main()
