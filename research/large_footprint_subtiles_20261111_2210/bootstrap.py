"""只在独立目录复制F2，失败候选与原基线均保留。"""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'render_count_scaling_20261111_2022'
source=HERE/'source'
assert not source.exists()
shutil.copytree(PREVIOUS/'background_stage/source',source)
for name in ('build','reports','logs'):(HERE/name).mkdir(exist_ok=True)
for name in ('launch.py','job_runner.py'):shutil.copy2(PREVIOUS/name,HERE/name)
manifest={}
for base in (PREVIOUS/'source',PREVIOUS/'background_stage/source'):
    for p in base.rglob('*'):
        if p.is_file():manifest[str(p)]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for p in [PREVIOUS/'build/mycartgs_render_count_scaling_v1.so',
          PREVIOUS/'background_stage/build/mycartgs_render_background_scaling_v1.so']:
    manifest[str(p)]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
(HERE/'reports/BASELINE_SHA256.json').write_text(json.dumps(manifest,indent=2))
main=source/'graph_main.cu'
text=main.read_text().replace('BackgroundScalingRasterizer','SubtileRasterizer')
text=text.replace('#include "packed_pixels.cuh"','#include "packed_pixels.cuh"\n#include "subtile_pixels.cuh"')
text=text.replace('pixel_mode<=18','pixel_mode<=22')
text=text.replace('pixel_mode_==18?inputs:nullptr','pixel_mode_>=18?inputs:nullptr')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==18)renderScalingPixels<9>' in x)
extra='\n'.join(f'      if(pixel_mode_=={mode})renderSubtilePixels<{kind}><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);'
                for mode,kind in [(19,0),(20,1),(21,2),(22,3)])
main.write_text(text.replace(line,line+'\n'+extra))
shutil.copy2(HERE/'subtile_pixels.cuh',source/'subtile_pixels.cuh')
print('SUBTILE_SOURCE_READY',len(manifest),flush=True)
