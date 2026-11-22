"""运行独立队列并核验训练、评价、部署和源文件完整性。"""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import parallel_batch


def main():
    parallel_batch.main()
    state = json.loads((ROOT/'reports/BATCH_STATUS.json').read_text())
    scenes = json.loads((ROOT/'reports/SCENES.json').read_text())
    protocol = json.loads((ROOT/'reports/PROTOCOL.json').read_text())
    rows = []
    checks = []
    for scene in scenes:
        job = next(j for j in state['scenes'] if j['id']==scene['id'])
        folder = ROOT/'reports'/scene['slug']
        record = dict(scene=scene['id'],status=job['status'],error=job.get('error',''),
            training_valid=False,evaluation_valid=False,deployment_pass=False)
        try:
            preflight=json.loads((folder/'PREFLIGHT.json').read_text())
            assert preflight['passed'] and len(preflight['test_views'])==32
            assert not set(preflight['test_views']) & set(preflight['validation_views'])
            updates=0
            for stage in protocol['stages']:
                out=ROOT/'models'/scene['slug']/stage['name']
                config=json.loads((out/'configuration.json').read_text())
                log=list(csv.DictReader((out/'training.csv').open()))
                assert [int(r['step']) for r in log]==list(range(1,stage['steps']+1))
                for k,v in [('seed',0),('learning_rate',stage['lr']),('image_every',stage['image_every']),
                    ('image_weight',stage['image_weight']),('residual_weight',stage['residual_weight']),('ratio_jitter',stage['ratio_jitter'])]:
                    assert config[k]==v,(scene['id'],stage['name'],k)
                assert config['teacher_fingerprint']==preflight['fingerprint']
                updates+=len(log)
            record.update(training_valid=True,updates=updates)
            quality=json.loads((folder/'QUALITY_SUMMARY.json').read_text())
            for name,identity in quality['identities'].items():
                assert hashlib.sha256((ROOT/'models'/scene['slug']/name/'best.pt').read_bytes()).hexdigest()==identity['sha256']
            views=list(csv.DictReader((folder/'QUALITY_PER_VIEW.csv').open()))
            assert len(views)==2816
            groups={}
            for row in views:
                groups.setdefault((row['frame'],row['ratio'],row['local']),[]).append(row)
            assert len(groups)==704 and all(len(group)==4 and len({(r['K'],r['selection_sha256']) for r in group})==1 for group in groups.values())
            record.update(evaluation_valid=True,per_view_rows=len(views))
            for item in quality['summaries']:
                rows.append(dict(dataset=scene['dataset_kind'],mode=scene['mode'],N=scene['gaussians'],**item))
            deployment=folder/'DEPLOYMENT_VALIDATION.json'
            if deployment.exists():
                assert json.loads(deployment.read_text())['passed']
                assert json.loads((folder/'GRAPH_PARITY.json').read_text())['passed']
                record['deployment_pass']=True
        except Exception as error:
            record['validation_error']=repr(error)
        checks.append(record)
    if rows:
        with (ROOT/'reports/DISTILLATION_METRICS.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    integrity=subprocess.run([sys.executable,str(HERE/'integrity.py')],cwd=ROOT,capture_output=True,text=True)
    (ROOT/'logs/final_integrity.log').write_text(integrity.stdout+'\n'+integrity.stderr)
    summary=dict(scenes=len(scenes),training_valid=sum(x['training_valid'] for x in checks),
        evaluation_valid=sum(x['evaluation_valid'] for x in checks),deployment_pass=sum(x['deployment_pass'] for x in checks),
        integrity_pass=integrity.returncode==0,checks=checks)
    (ROOT/'reports/RESULTS_VALIDATION.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    lines=['# TUM 与 VECtor Compensator 蒸馏报告','',
        f"训练核验 {summary['training_valid']}/{len(scenes)}，评价核验 {summary['evaluation_valid']}/{len(scenes)}，部署验收 {summary['deployment_pass']}/{len(scenes)}，保护文件完整性 {summary['integrity_pass']}。",'',
        '每场景使用独立32/32学生，四阶段共17600步；教师、Master、Selector与Anchor Feature冻结。训练六档Ratio并在连续阶段采样中间比例，按每5帧留出学生未见视角，32帧测试。教师和地图已使用同轨迹。','',
        '| 场景 | 相机 | 高斯数 | 全局PSNR损失 | 局部PSNR损失 | 通过门限 | 部署 |','|---|---|---:|---:|---:|---:|---|']
    for scene in scenes:
        group=[r for r in rows if r['scene']==scene['id'] and r['student']=='continuous_calibration']
        check=next(x for x in checks if x['scene']==scene['id'])
        if group:
            g=statistics.mean(r['mean_loss'] for r in group if not r['local'])
            l=statistics.mean(r['mean_loss'] for r in group if r['local'])
            lines.append(f"| {scene['scene']} | {scene['mode']} | {scene['gaussians']} | {g:.5f} | {l:.5f} | {sum(r['gate'] for r in group)}/22 | {check['deployment_pass']} |")
        else:
            lines.append(f"| {scene['scene']} | {scene['mode']} | {scene['gaussians']} | 未完成 | 未完成 | — | {check['error']} {check.get('validation_error','')} |")
    lines += ['', 'PSNR损失为同视角、同选点与实际K下的Teacher PSNR减去Student PSNR，正值表示学生退化。质量门限依次为平均损失≤0.01 dB、p99≤0.05 dB、最差≤0.1 dB。每条件32视角，统计范围有限。CUDA部署一致性与蒸馏质量是独立检查。','',
        '逐Ratio PSNR、SSIM、LPIPS、实际K与长尾见DISTILLATION_METRICS.csv及各场景QUALITY_PER_VIEW.csv和ATTRIBUTES_PER_VIEW.csv。每组模型位于models/<场景>/continuous_calibration/best.pt；完整身份、错误及校验见RESULTS_VALIDATION.json。训练为两路并发，未做独占GPU速度基准。不保存渲染图像。']
    (ROOT/'reports/VECTOR_TUM_DISTILLATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('FINAL_VALIDATION',json.dumps({k:v for k,v in summary.items() if k!='checks'}),flush=True)


if __name__=='__main__':
    main()
