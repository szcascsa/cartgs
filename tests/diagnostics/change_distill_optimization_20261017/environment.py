from pathlib import Path
import hashlib
import json
import subprocess
import torch

D = Path(__file__).resolve().parent
EVENT = Path('/home/prmi/zwj/mycartgs/results/change_distill_ablation/20261017-2038/events_supplement/local_loop/event_6207_1')
data = {'torch_python': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'torch_config': torch.__config__.show(),
        'nvidia_smi': subprocess.check_output(['nvidia-smi', '-q'], text=True),
        'nvcc': subprocess.check_output(['/usr/local/cuda-11.8/bin/nvcc', '--version'], text=True),
        'compiler': subprocess.check_output(['g++', '--version'], text=True)}
(D / 'environment.json').write_text(json.dumps(data, indent=2))
hashes = {}
for name in ['event.json', 'snapshot.pt', 'selector.pt', 'transform_field.pt']:
    with (EVENT / name).open('rb') as file:
        hashes[name] = hashlib.file_digest(file, 'sha256').hexdigest()
(D / 'input_sha256.json').write_text(json.dumps(hashes, indent=2))
reuse = {}
for stage in ['baseline', 'stage_a', 'stage_a_grouped', 'stage_b', 'stage_c']:
    reuse[stage] = {}
    for source in (D / stage / 'source/src').glob('*.cu'):
        obj = D / stage / 'objects_abi1' / (source.name + '.o')
        original = D / 'baseline/source/src' / source.name
        reuse[stage][source.name] = {'unchanged_from_baseline': source.read_bytes() == original.read_bytes(),
                                    'object_sha256': hashlib.sha256(obj.read_bytes()).hexdigest()}
(D / 'cuda_objects.json').write_text(json.dumps(reuse, indent=2))
