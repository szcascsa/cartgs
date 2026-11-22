"""交付核验：768条数值、32张图的真实Ratio坐标及归档完整性。"""
from pathlib import Path
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent


def main():
    data=json.loads((ROOT/'data/QUALITY_RENDERED_COUNTS.json').read_text(encoding='utf-8'))
    source=ROOT.parent/'all_scene_quality_figures_20261108/data/ALL_SCENE_RESULTS.json'
    original=json.loads(source.read_text(encoding='utf-8'))
    assert data['groups']==original['groups']
    assert data['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert data['source_hashes_unchanged'] and data['images_saved']==0 and data['training_steps']==0
    assert len(data['counts'])==768
    table=list(csv.DictReader((ROOT/'QUALITY_RENDERED_COUNTS.csv').open(encoding='utf-8-sig')))
    assert len(table)==768
    lookup={(row['group'],row['scope'],float(row['ratio'])):row for row in table}
    assert len(lookup)==768
    groups={group['id']:group for group in original['groups']}
    for row in data['counts']:
        output=lookup[(row['group'],row['scope'],row['ratio'])]
        group=groups[row['group']]
        reference=group['vanilla'] if row['scope']=='vanilla' else next(
            item for item in group['ours'] if item['scope']==row['scope'] and item['ratio']==row['ratio'])
        for key in ['psnr','ssim','lpips']:
            assert row[key]==reference[key],(row['group'],row['scope'],key)
        for key in ['psnr','ssim','lpips','submitted_K_mean','submitted_K_min','submitted_K_max',
                    'projected_K_mean','projected_K_min','projected_K_max',
                    'contributing_K_mean','contributing_K_min','contributing_K_max']:
            assert float(output[key])==row[key],(row['group'],key)
        assert float(output['source_map_N'])==row['model_total_N']
        expected=row['submitted_K_mean'] if row['scope']=='global' else row['model_total_N']
        assert float(output['render_model_N'])==expected
        for suffix in ['mean','min','max']:
            assert 0<=row['contributing_K_'+suffix]<=row['projected_K_'+suffix]<=row['submitted_K_'+suffix]<=row['model_total_N']
        assert len(row['checks'])==3 and all(c['render_and_radii_bitwise_equal'] for c in row['checks'])
    curves=0
    ratio=np.array([.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7])
    expected=(ratio-ratio[0])/(ratio[-1]-ratio[0])
    for group in data['groups']:
        stem=group['id'].replace('/','__');folder=ROOT/'figures'/group['dataset']
        with Image.open(folder/(stem+'.png')) as image:
            assert image.size[0]==6660 and abs(image.size[1]-7605)<=1
            assert abs(image.info['dpi'][0]-450)<.02
        svg=ET.parse(folder/(stem+'.svg')).getroot()
        for index in range(1,6):
            axes=next(node for node in svg.iter('{http://www.w3.org/2000/svg}g') if node.attrib.get('id')==f'axes_{index}')
            found=0
            for node in axes.iter('{http://www.w3.org/2000/svg}path'):
                style=node.attrib.get('style','');path=node.attrib.get('d','')
                if any('stroke: '+color in style for color in ['#0077bb','#ee7733']) and path.count('L')==10:
                    xy=np.array([float(n) for n in re.findall(r'-?\d+(?:\.\d+)?',path)]).reshape(-1,2)
                    actual=(xy[:,0]-xy[0,0])/(xy[-1,0]-xy[0,0])
                    assert np.max(abs(actual-expected))<1e-7,(group['id'],index)
                    found+=1
            assert found==2,(group['id'],index,found)
            curves+=found
    assert curves==320
    overview=ROOT/'figures/ALL_SCENES_OVERVIEW.png'
    assert overview.is_file() and overview.with_suffix('.svg').is_file()
    archive=ROOT/'QUALITY_RENDERED_COUNTS_RESULTS.zip'
    with zipfile.ZipFile(archive) as packed:
        assert packed.testzip() is None
        assert len([n for n in packed.namelist() if n.endswith('.png')])==33
        assert len([n for n in packed.namelist() if n.endswith('.svg')])==33
    receipt=dict(verified=True,scene_figures=32,overview_figures=1,metric_records=768,
                 per_view_count_records=data['total_rendered_views'],ratio_curves_verified=curves,
                 render_bitwise_probe_cases=sum(len(row['checks']) for row in data['counts']),
                 quality_changes=0,model_and_source_hashes_unchanged=True,
                 archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (ROOT/'DELIVERY_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
