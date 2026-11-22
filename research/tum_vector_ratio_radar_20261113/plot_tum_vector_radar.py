"""TUM、VECtor全部已有场景宏平均，复用Replica的等距质量／虚线K雷达样式。"""
from pathlib import Path
import csv
import json
import math
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
STYLE = ROOT.parent/'replica_ratio_radar_20261113'
if not STYLE.is_dir():STYLE=ROOT
sys.path.insert(0,str(STYLE))
import plot_replica_radar_with_counts as style

base = style.base
SOURCE = ROOT.parent/'cartgs_pruning_baselines_20261112/reports/METHODS_ALL_METRICS.csv'
if not SOURCE.exists():SOURCE=ROOT/'data/METHODS_ALL_METRICS.csv'
SOURCE_SHA = 'e56cc01dfe0dcfb1613f06d4211ce5d7e137b6c6ce341c2667db200cc027c891'
DATASETS = {
    'TUM':dict(source='TUM_RGBD',title='TUM',modes=['mono','rgbd'],
               scenes=['freiburg1_desk','freiburg1_room','freiburg1_xyz','freiburg2_xyz','freiburg3_long_office_household']),
    'VECTOR':dict(source='VECtor',title='VECtor',modes=['mono','stereo'],
                  scenes=['corner-slow','corridors-dolly','robot-normal'])}
MODE_LABELS = {'mono':'单目','rgbd':'RGB-D','stereo':'双目'}


def aggregate(config,destination):
    assert base.digest(SOURCE)==SOURCE_SHA
    source=list(csv.DictReader(SOURCE.open(encoding='utf-8-sig')))
    selected=[r for r in source if r['group'].startswith(config['source']+'/') and r['method'] in base.METHODS]
    lookup={};per_scene=[]
    for r in selected:
        _,scene,mode=r['group'].split('/')
        entry=dict(group=r['group'],scene=scene,mode=mode,method=r['method'],ratio=float(r['ratio']),
                   N=int(r['N']),K=float(r['K_mean']),frames=int(r['frames']),source=r['source'])
        entry.update({metric:float(r[metric]) for metric in style.METRICS})
        assert all(math.isfinite(entry[k]) for k in ['K',*style.METRICS])
        assert 0<=entry['K']<=entry['N']
        key=scene,mode,entry['method'],entry['ratio'];assert key not in lookup
        lookup[key]=entry;per_scene.append(entry)
    expected={(s,m,method,r) for s in config['scenes'] for m in config['modes']
              for method in base.METHODS for r in base.RATIOS}
    assert set(lookup)==expected
    rows=[]
    for mode in ['all',*config['modes']]:
        cameras=config['modes'] if mode=='all' else [mode]
        for method in base.METHODS:
            for ratio in base.RATIOS:
                points=[lookup[s,m,method,ratio] for s in config['scenes'] for m in cameras]
                entry=dict(mode=mode,method=method,ratio=ratio,scenes=len(config['scenes']),
                           scene_mode_groups=len(points),weighting='equal_scene_then_equal_camera_mode',
                           evaluated_frames=sum(p['frames'] for p in points))
                for input_key,output_key in [('N','N_mean'),('K','K_mean'),*[(m,m) for m in style.METRICS]]:
                    entry[output_key]=base.mean([base.mean([lookup[s,m,method,ratio][input_key] for m in cameras])
                                                for s in config['scenes']])
                assert 0<=entry['psnr']<=40 and 0<=entry['ssim']<=1 and 0<=entry['lpips']<=1
                rows.append(entry)
    data=destination/'data';data.mkdir(parents=True,exist_ok=True)
    base.write_csv(data/'PER_SCENE.csv',per_scene);base.write_csv(data/'RADAR_MEANS.csv',rows)
    return rows,per_scene


def draw(ax,rows,metric,mode,k_limit,config,small=False):
    style.draw_radar(ax,rows,metric,mode,k_limit,small=small)
    if mode!='all':
        ax.set_title(f"{MODE_LABELS[mode]} · {len(config['scenes'])}场景均值",fontsize=14,fontweight='bold',pad=14,color='#26343F')


def plot_dataset(tag,config):
    destination=ROOT/tag;rows,per_scene=aggregate(config,destination)
    figures=destination/'figures';figures.mkdir(exist_ok=True)
    n=len(config['scenes']);camera_text='与'.join(MODE_LABELS[m] for m in config['modes'])
    title=config['title'];subtitle=f'每场景先平均{camera_text}，再对{n}个场景等权平均'
    k_limit=math.ceil(max(r['K_mean'] for r in rows)/10000)*10000
    note=f'右刻度为K：中心0、外圈{k_limit/1000:g}千点；CaRtGS两虚线重合，以交错线段显示'
    (figures/'data-manifest.md').write_text('# 数据清单\n\n'
        '|图|数据|性质|来源|输出|\n|---|---|---|---|---|\n'
        '|三张主图、三张分模式图及一张总览|data/RADAR_MEANS.csv|真实|METHODS_ALL_METRICS.csv|PNG/SVG/PDF|\n\n'
        '实线读取质量指标，虚线读取K_mean；质量刻度沿用Replica原版。\n',encoding='utf-8')
    for metric in style.METRICS:
        fig=base.plt.figure(figsize=(7.2,9.4))
        ax=fig.add_axes([.035,.325,.93,.565]);draw(ax,rows,metric,'all',k_limit,config)
        fig.suptitle(title+' · '+base.METRICS[metric][0],fontsize=20,fontweight='bold',y=.983,color='#22313B')
        fig.text(.5,.935,f'{n}场景等权平均 · 每场景{camera_text}等权',ha='center',fontsize=11,color='#52636F')
        style.legends(fig,.273,.244,fontsize=9.8)
        fig.text(.5,.220,'平均保留Gaussian数K（千点）',ha='center',fontsize=10,color='#33434F')
        style.count_table(fig,rows,'all',[.045,.080,.91,.123])
        fig.text(.5,.048,note,ha='center',fontsize=8.8,color='#52636F')
        fig.text(.5,.023,'实线质量越外越好；各方法按自身N保留，点数越多不代表质量越高',ha='center',fontsize=8.8,color='#52636F')
        base.save(fig,figures/f'{tag}_{metric.upper()}_RADAR')
        fig=base.plt.figure(figsize=(13.6,9.4))
        for index,mode in enumerate(config['modes']):
            ax=fig.add_axes([.025+index*.5,.325,.45,.555]);draw(ax,rows,metric,mode,k_limit,config,small=True)
            style.count_table(fig,rows,mode,[.025+index*.5,.080,.45,.123],fontsize=7.6)
            fig.text(.25+index*.5,.220,'平均保留Gaussian数K（千点）',ha='center',fontsize=10,color='#33434F')
        fig.suptitle(title+' · '+base.METRICS[metric][0]+' · 分模式核对',fontsize=20,fontweight='bold',y=.985,color='#22313B')
        style.legends(fig,.273,.244,fontsize=10)
        fig.text(.5,.035,f'{n}个场景等权；'+note,ha='center',fontsize=10,color='#52636F')
        base.save(fig,figures/f'{tag}_{metric.upper()}_BY_MODE')
    fig=base.plt.figure(figsize=(17.4,8.4))
    for index,metric in enumerate(style.METRICS):
        ax=fig.add_axes([.005+index/3,.345,.323,.46]);draw(ax,rows,metric,'all',k_limit,config,small=True)
        ax.set_title(base.METRICS[metric][0],fontsize=18,fontweight='bold',pad=12)
        style.count_table(fig,rows,'all',[.012+index/3,.095,.309,.11],fontsize=6.9)
        fig.text(1/6+index/3,.222,'平均保留Gaussian数K（千点）',ha='center',fontsize=9,color='#33434F')
    fig.suptitle(f'{title} · {n}场景均值 · 质量与保留点数',fontsize=20,fontweight='bold',y=.985,color='#22313B')
    fig.text(.5,.930,subtitle+'；蓝：随机删除，金：Opacity删除，红：我们全局LoD',ha='center',fontsize=11,color='#52636F')
    style.legends(fig,.280,.249,fontsize=11)
    fig.text(.5,.043,'质量沿用原等距刻度；'+note+'；虚线向外仅表示保留点更多。',ha='center',fontsize=10,color='#52636F')
    base.save(fig,figures/f'{tag}_RADAR_OVERVIEW')
    provenance=dict(dataset=config['source'],scenes=config['scenes'],modes=config['modes'],
        source=str(SOURCE),source_sha256=SOURCE_SHA,source_rows=len(per_scene),mean_rows=len(rows),
        real_data=True,average=subtitle,methods=base.METHODS,ratios=base.RATIOS,
        quality_scale='PSNR 0–40；SSIM 0–1；LPIPS反向1–0；全部为等距径向刻度',
        K_axis_max=k_limit,K_definition='全局保留模型Gaussian数／Renderer输入K，不是逐视角可见点数',
        count_table='单位千点，显示两位小数；完整精度见CSV',
        comparison='相同Ratio，各方法按自身N保留，不同K；各自已有轨迹均值，不是固定相机位姿对照',
        new_training=False,new_rendering=False)
    (destination/'data/SOURCE_MANIFEST.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (destination/'README_结果说明.md').write_text(f'# {title}质量与保留点数\n\n'+subtitle+'，不按帧数加权。\n\n'
        '场景：'+ '、'.join(config['scenes'])+'。\n\n'
        '蓝色为CaRtGS随机删除，金色为CaRtGS Opacity删除，红色为我们全局LoD。实线读取左侧质量刻度，'
        f'虚线读取右侧K刻度（0–{k_limit:g}），图下表以千点表示平均K。'
        '两种CaRtGS裁剪K相同，虚线重合并用交错蓝金线段显示，未偏移数据。\n\n'
        '质量刻度保持Replica原版：PSNR为0–40，SSIM为0–1，LPIPS为反向1–0，均为等距刻度。'
        '相同指标在不同数据集使用同样质量刻度；点数上限按各数据集最大平均K整万向上取整。\n\n'
        '保留K不是逐视角可见或参与Alpha合成的点数。各方法按自身N确定预算，属于同Ratio、不同K比较。'
        '沿用现有未校正原图评价结果及各自轨迹；帧数、来源和逐场景指标保存在CSV。'
        '随机删除只有seed0，不代表多随机种子稳定性。本轮不重新训练或渲染。\n',encoding='utf-8')
    assert all(len(list(figures.glob('*.'+ext)))==7 for ext in ['png','svg','pdf'])
    print(title,'完成：',len(per_scene),'条逐场景数据，',len(rows),'条均值，7张图，K上限',k_limit,flush=True)
    return provenance


def main():
    assert not base.EXPANDED_SCALE
    base.configure_font();provenance={tag:plot_dataset(tag,config) for tag,config in DATASETS.items()}
    shared_data=ROOT/'data';shared_data.mkdir(exist_ok=True)
    if SOURCE.resolve()!=(shared_data/'METHODS_ALL_METRICS.csv').resolve():
        shutil.copyfile(SOURCE,shared_data/'METHODS_ALL_METRICS.csv')
    receipt=dict(complete=True,source_unchanged=base.digest(SOURCE)==SOURCE_SHA,datasets=provenance,
                 png=14,svg=14,pdf=14,
                 figures={str(p.relative_to(ROOT)).replace('\\','/'):base.digest(p) for p in ROOT.glob('*/figures/*') if p.is_file()})
    (ROOT/'VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'README_结果说明.md').write_text('# TUM与VECtor Ratio雷达图\n\n'
        'TUM使用5场景单目与RGB-D共10组；VECtor使用3场景单目与双目共6组。'
        '每个数据集分别输出PSNR、SSIM、LPIPS主图、分模式图及三指标总览，均附虚线保留点数与数值表。'
        '质量保持原等距刻度，颜色及线型与Replica一致。详细场景、口径、数据与来源见各数据集子目录。\n',encoding='utf-8')
    with zipfile.ZipFile(ROOT/'TUM_VECTOR_RATIO_RADAR_WITH_COUNTS.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ['TUM','VECTOR','data']:
            for path in (ROOT/folder).rglob('*'):
                if path.is_file():archive.write(path,path.relative_to(ROOT))
        for name in ['plot_tum_vector_radar.py','VERIFIED.json','README_结果说明.md']:archive.write(ROOT/name,name)
        for name in ['plot_replica_radar.py','plot_replica_radar_with_counts.py']:archive.write(STYLE/name,name)
    print('TUM_VECTOR_RADAR_COMPLETE：14张图，PNG／SVG／PDF及数据已打包',flush=True)


if __name__=='__main__':main()
