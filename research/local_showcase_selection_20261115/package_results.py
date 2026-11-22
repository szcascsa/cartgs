"""核验并整理四组正式图片、两组备选和全部排名证据。"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent


def load(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ranking=load(ROOT/'reports/RANKING.json')
    assert load(ROOT/'RENDER_QUEUE.json')['state']=='complete'
    receipts=list((ROOT/'images').rglob('RENDER_VERIFIED.json'))
    assert len(receipts)==6
    exports=[];all_records=[]
    for path in receipts:
        receipt=load(path);assert receipt['checkpoint_unchanged']
        for name,digest in receipt['checkpoint_sha256'].items():assert sha(Path(name))==digest
        all_records.extend(receipt['records'])
        best=receipt['candidates'][0]
        label=receipt['records'][0]['label'];source=path.parent/f"rgb_{best['rgb_index']:06d}"
        destination=ROOT/'selected'/label/receipt['group']
        destination.mkdir(parents=True,exist_ok=True)
        for name in ['GT.png','ratio_01.png','ratio_05.png','ratio_10.png','ratio_30.png']:
            shutil.copyfile(source/name,destination/name)
        records=[record for record in receipt['records'] if record['rgb_index']==best['rgb_index']]
        assert len(records)==4
        exports.append(dict(id=receipt['group'],label=label,frame=best,
                            directory=str(destination),records=records))
    assert len(all_records)==72 and len(exports)==6
    assert all(record['selection_identical'] and max(record['abs_errors'].values())==0 for record in all_records)
    for name,digest in ranking['input_sha256'].items():assert sha(Path(name))==digest
    receipt=dict(complete=True,groups_scanned=26,views_scanned=ranking['frame_count'],
                 historical_metric_rows=ranking['frame_count']*4,exported_best_groups=4,
                 alternative_groups=2,verified_renders=72,all_selection_hashes_equal=True,
                 historical_metric_max_abs_error=0.,checkpoints_and_inputs_unchanged=True,
                 exports=exports)
    (ROOT/'reports/EXPORT_VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    rows=[]
    for group in exports:
        for record in group['records']:
            rows.append(dict(id=group['id'],label=group['label'],rgb_index=record['rgb_index'],
                             trajectory_index=record['trajectory_index'],evaluation_ordinal=record['evaluation_ordinal'],
                             rgb_timestamp=group['frame']['rgb_timestamp'],ratio=record['ratio'],
                             N=record['N'],candidate_N=record['candidates'],K=record['K'],
                             **record['scores'],image=str(Path(group['directory'])/f"ratio_{int(record['ratio']*100):02d}.png")))
    with (ROOT/'reports/SELECTED_VIEW_METRICS.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Replica／TUM局部渲染最佳视角筛选','',
           '覆盖Replica 8个场景×Mono／RGB-D，以及TUM 5个场景×Mono／RGB-D，共26组、49,740个评价视角。'
           '每个视角检查1%、5%、10%、30%四档局部渲染，共198,960条历史指标。','',
           '## 固定视角筛选规则','',
           '每组先按同一视角四档PSNR的算术平均降序排名，LPIPS为并列时的次级排序。'
           '随后在Replica RGB-D、Replica Mono、TUM RGB-D、TUM Mono四类中各取最高者。'
           '这代表四档整体最优共用视角，不代表该帧在每一个Ratio分别达到全轨迹最大值。', '',
           '选图使用完整图像；没有裁剪、改亮、锐化或图像修复。每个Ratio复用相同相机、GT关联与冻结模型。'
           '图上的PSNR来自保存PNG前的FP32渲染，与原评价相同；PNG为8位展示图。', '',
           '## 四组数值最优','',
           '| 数据集／场景／模式 | RGB索引（0起） | 轨迹索引（0起） | 1% PSNR／K | 5% PSNR／K | 10% PSNR／K | 30% PSNR／K |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label in ['best','alternatives']:
        if label=='alternatives':
            lines+=['','## 更明亮的Replica备选','',
                    'Replica最高PSNR视角集中在较暗、低纹理区域，不宜用高PSNR直接推断细节恢复更强。'
                    '额外保留office4 RGB-D、office0 Mono各自的四档整体最优帧，便于展示选择；它们不是绝对PSNR冠军。','',
                    '| 数据集／场景／模式 | RGB索引（0起） | 轨迹索引（0起） | 1% PSNR／K | 5% PSNR／K | 10% PSNR／K | 30% PSNR／K |',
                    '|---|---:|---:|---:|---:|---:|---:|']
        for item in sorted([item for item in exports if item['label']==label],key=lambda item:item['id']):
            b=item['frame'];values=' | '.join(f"{b['psnr_'+str(n)]:.2f}／{b['K_'+str(n)]}" for n in [1,5,10,30])
            lines.append(f"| {item['id']} | {b['rgb_index']} | {b['trajectory_index']} | {values} |")
    lines+=['','## 全部26组的最佳共用视角','',
            '| 场景／模式 | RGB索引 | 1% PSNR | 5% PSNR | 10% PSNR | 30% PSNR | 平均PSNR |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for group in sorted(ranking['groups'],key=lambda group:(group['dataset'],group['mode'],-group['best']['mean_psnr'])):
        b=group['best'];values=' | '.join(f"{b['psnr_'+str(n)]:.2f}" for n in [1,5,10,30])
        lines.append(f"| {group['id']} | {b['rgb_index']} | {values} | {b['mean_psnr']:.2f} |")
    lines+=['','## 语义与正确性','',
            '- 局部Ratio按当前名义Anchor视锥候选池计算预算，K是实际送入渲染的点数；1%不代表全图Master数量N的1%。',
            '- 双网络真实视角＋LI=0.1，沿用已完成正式批次的checkpoint；没有重训或改变选点／补偿。',
            '- Mono沿用原评价尺度处理，RGB-D沿用原评价处理；每行四档共用同一估计相机视角。',
            '- 共补渲染6组×3个候选视角×4档=72张；选点hash、K以及PSNR／SSIM／LPIPS与历史记录全部相同，最大指标差值为0。',
            '- 所有候选模型与原指标输入的SHA256保持不变；原评价目录和生产配置未修改。',
            '- 此处是明确挑选的最佳展示案例，不是数据集平均表现。Replica两种模式最佳恰好都是office1，TUM两种模式最佳恰好都是freiburg2_xyz，因此四组不等于四个不同场景名称。',
            '- 全部帧排名见ALL_FRAME_RANKING.csv；每组每Ratio各自最优见PER_RATIO_BEST_VIEW.csv；导出指标和实际K见SELECTED_VIEW_METRICS.csv。','',
            '## 图片位置','',
            'selected/best/<数据集>/<场景>/<模式>/：四组正式最佳视角，各有GT.png及ratio_01／05／10／30.png。',
            'selected/alternatives/：两组Replica明亮备选；images/：每组前三个高质量候选的原尺寸PNG。']
    (ROOT/'reports/LOCAL_RENDER_BEST_VIEWS_REPORT.md').write_text('\n'.join(lines)+'\n')
    with zipfile.ZipFile(ROOT/'LOCAL_BEST_VIEWS_RESULTS.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ['reports','selected','preview']:
            for path in (ROOT/folder).rglob('*'):
                if path.is_file():archive.write(path,path.relative_to(ROOT))
        for name in ['rank_views.py','render_candidates.py','run_render_queue.py','package_results.py']:
            archive.write(ROOT/name,name)
    print('PACKAGED',json.dumps({key:value for key,value in receipt.items() if key!='exports'}),flush=True)


if __name__=='__main__':
    main()
