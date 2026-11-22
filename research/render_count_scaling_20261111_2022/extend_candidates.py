"""第二组候选：减小微型请求填充，或展开原顺序像素循环。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
backup=HERE/'source_v1'
assert not backup.exists()
shutil.copytree(source,backup)
binary=HERE/'build/mycartgs_render_count_scaling_v1.so'
shutil.copy2(binary,HERE/'reports/candidates_v1.so')
main=source/'graph_main.cu'
text=main.read_text()
text=text.replace('pixel_mode<=9','pixel_mode<=12')
anchor='if(pixel_mode_==9)renderScalingPixels<4>'
assert anchor in text
line=next(x for x in text.splitlines() if anchor in x)
extra='\n'.join(line.replace('==9','=='+str(mode)).replace('<4>','<'+str(mode-5)+'>') for mode in (10,11,12))
text=text.replace(line,line+'\n'+extra)
text=text.replace('plan_granularity==0||plan_granularity==512','plan_granularity==0||plan_granularity==512||plan_granularity==32')
text=text.replace('int capacity=512;if(plan_granularity==512)', 'int capacity=plan_granularity==32?32:512;if(plan_granularity==32)capacity=points<=512?((points+31)/32)*32:((points+511)/512)*512;else if(plan_granularity==512)')
text=text.replace('if(plan_granularity==512)for(int k=512;k<=pool_capacity;k+=512)capacities.push_back(k);',
"""if(plan_granularity==32){
      for(int k=32;k<=std::min(512,pool_capacity);k+=32)capacities.push_back(k);
      for(int k=1024;k<=pool_capacity;k+=512)capacities.push_back(k);
    } else if(plan_granularity==512)for(int k=512;k<=pool_capacity;k+=512)capacities.push_back(k);""")
text=text.replace('int capacity=512;\n    if(plan_granularity==512)',
"""int capacity=plan_granularity==32?32:512;
    if(plan_granularity==32)capacity=P<=512?((P+31)/32)*32:((P+511)/512)*512;
    else if(plan_granularity==512)""")
assert 'if(plan_granularity==32)capacity=P' in text
assert 'if(plan_granularity==32)capacity=points' in text
main.write_text(text)
pixels=source/'scaling_pixels.cuh'
text=pixels.read_text().replace('    for(int j=0;j<count;++j) {',
"""    constexpr int Unroll=Mode==5?2:Mode==6?4:Mode==7?8:1;
    #pragma unroll Unroll
    for(int j=0;j<count;++j) {""")
assert '#pragma unroll Unroll' in text
pixels.write_text(text)
print('CANDIDATES_V2_READY',flush=True)
