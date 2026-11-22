"""保留每个Tensor kernel的真实HMMA指令证据，不保留冗余大SASS。"""
from pathlib import Path
import subprocess,re,json,hashlib
HERE=Path(__file__).resolve().parent;binary=HERE/'build/mycartgs_tensor_deployment_v1.so'
raw=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],text=True)
functions=[]
for part in re.split(r'Function : ',raw)[1:]:
    name=part.splitlines()[0]
    if 'selectorTensor' not in name and 'compensationTensor' not in name:continue
    instructions=[line.strip() for line in part.splitlines() if 'HMMA.' in line]
    assert instructions,name
    functions.append(dict(function=name,HMMA_instructions=len(instructions),FFMA_instructions=len(re.findall(r'\bFFMA\b',part)),examples=instructions[:4]))
assert functions
(HERE/'reports/TENSOR_SASS_EVIDENCE.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),functions=functions),indent=2))
print('TENSOR_SASS_PASS',len(functions),'FUNCTIONS',sum(f['HMMA_instructions'] for f in functions),'HMMA')
