"""固定完成标准的最终检查；未通过不能以PSNR接近替代。"""
from pathlib import Path
import ast,hashlib,json,re,time
import pandas as pd

HERE=Path(__file__).resolve().parent


def read(path):return json.loads((HERE/path).read_text())


def main():
    result=read('FINAL_VERIFICATION.json');q=read('CAMERA_GRAPH_FINAL_VERIFICATION.json');integrity=read('reports/INTEGRITY_FINAL.json')
    async_path=HERE/'CAMERA_ASYNC_FINAL_VERIFICATION.json';async_passed=async_path.exists() and read('CAMERA_ASYNC_FINAL_VERIFICATION.json')['passed']
    assert result['passed_new_numerics'] and result['full_G_original_A_pass'] and result['full_G_previous_B_pass']
    assert q['passed'] and integrity['passed']
    required=['edges04','sanitizer_mem04','sanitizer_init04','coverage02','trajectory01','async01',
        'sanitizer_async_mem01','sanitizer_async_sync01','async_expansion01','camera_graph_edges01',
        'sanitizer_camera_graph01','camera_trajectory01','delivery_smoke01']
    if async_passed:required += ['async_camera_edges01','sanitizer_camera_async01','async_camera_trajectory01','delivery_smoke02']
    for name in required:assert read(f'reports/{name}/SUMMARY.json')['passed'],name
    sanitizers=['sanitizer_mem04','sanitizer_init04','sanitizer_async_mem01','sanitizer_async_sync01','sanitizer_camera_graph01']
    if async_passed:sanitizers.append('sanitizer_camera_async01')
    for name in sanitizers:
        path=HERE/'logs'/(name+'_details.log')
        if path.exists():text=path.read_text()
        else:text=__import__('gzip').decompress(path.with_suffix('.log.gz').read_bytes()).decode()
        assert re.search(r'ERROR SUMMARY: 0 errors',text),name
    jobs=['formal02','formal04','preparation01','preparation02','camera_final01','camera_final02','profile_suite02','profile_q01']
    if async_passed:jobs += read('CAMERA_ASYNC_FINAL_VERIFICATION.json')['formal_names']+['profile_r01']
    for name in jobs:
        job=read(f'reports/jobs/{name}.json');assert job['state']=='complete' and job['measurement_valid'] and job['foreign_active_samples']==0,name
    raster=pd.read_csv(HERE/'RASTERIZER_RESULTS.csv');pipe=pd.read_csv(HERE/('FINAL_ASYNC_PIPELINE_RESULTS.csv' if async_passed else 'FINAL_CAMERA_GRAPH_PIPELINE_RESULTS.csv'))
    assert raster.scene.nunique()==pipe.scene.nunique()==24
    assert len(raster)==24*14*4 and len(pipe)==24*14*(6 if async_passed else 5)
    assert not raster[['median_ms','p95_ms','K_median','N']].isna().any().any()
    assert not pipe[['median_ms','p95_ms','K_median','N']].isna().any().any()
    assert (raster.p95_ms>=raster.median_ms).all() and (pipe.p95_ms>=pipe.median_ms).all()
    assert (raster.K_max<=raster.N).all() and (pipe.K_max<=pipe.N).all()
    for path in [HERE/'FULL_GATES.csv',HERE/'FINAL_CAMERA_GRAPH_FULL_GATES.csv']:
        data=pd.read_csv(path);assert data.scene.nunique()==24 and set(data['round'])=={1,2}
        assert data[[name for name in data if name.startswith('pass_')]].all().all()
    if async_passed:
        data=pd.read_csv(HERE/'FINAL_ASYNC_FULL_GATES.csv');assert data[[name for name in data if name.startswith('pass_')]].all().all()
    figures=HERE/'figures'
    assert len(list((figures/'final_scenes').glob('*.png')))==24
    assert len(list((figures/'final_scenes').glob('*.svg')))==24
    for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    report=HERE/'LOW_POINT_RENDERING_SIX_HOUR_REPORT.md';text=report.read_text(encoding='utf-8')
    assert '未部署' in text and '反例' in text and '356,856' in text
    result=dict(passed=True,time=time.time(),scenes=24,raster_rows=len(raster),pipeline_rows=len(pipe),
        correctness_logs=required,production_integrity=integrity['passed'],all_two_round_full_gates_passed=True,
        report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),final_scene_figures=24,
        no_render_images=True,no_commit_push_deploy=True)
    (HERE/'DELIVERY_VERIFICATION.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
