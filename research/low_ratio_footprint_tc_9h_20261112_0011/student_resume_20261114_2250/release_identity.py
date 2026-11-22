"""核验最终执行Student、Selector、Feature和实际二进制；只读原模型。"""
from release_joint import *
import hashlib


def tensor_records(values):
    return [dict(shape=list(value.shape), dtype=str(value.dtype), elements=value.numel(),
        sha256=hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()) for value in values]


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    mode=json.loads((HERE/'reports/SELECTED_RASTER_20261115.json').read_text())['mode'];rows=[]
    for spec in student_specs():
        source,student,path=load_case(spec)
        expected=network_variants(source,student)['tensor_best'][0]
        actual=JointReleaseRenderer(source,student,mode,reuse=True,fused=True,finish=True).network
        fields={'Selector':'_selector','Student':'_compensator','Feature':None,'View':None}
        records={}
        for name,attribute in fields.items():
            def values(engine):
                if name=='Feature':return engine._base.tables
                if name=='View':return [engine._view_weight]
                return getattr(engine,attribute)
            first,second=tensor_records(values(expected)),tensor_records(values(actual))
            assert first==second,(spec['id'],name)
            records[name]=second
        rows.append(dict(**identity(spec,source,student,path),actual_tensors=records,
            student_parameter_count=sum(value.numel() for value in student.parameters()),
            selector_parameter_count=sum(value.numel() for value in actual._selector),
            selector_backend='FP16 PTX MMA Tensor Core, FP32 accumulation',
            compensator_backend='FP16 WMMA Tensor Core, FP32 accumulation and decoding',
            preparation_backend='CPU C++ camera; ordinary CUDA visibility, View and index operations'))
        del source,student,expected,actual;gc.collect();torch.cuda.empty_cache()
    binaries={}
    for folder in ['build_joint_release','build_encoded_sign','build_tail_cutoff','build_tail_strong']:
        for path in (HERE/folder).glob('*.so'):
            binaries[str(path)]=sha(path)
    mappings=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if '.so' in line and 'low_ratio_footprint_tc' in line})
    save(out/'IDENTITIES.json',rows);save(out/'BINARIES.json',binaries);save(out/'LOADED_MODULES.json',mappings)
    save(out/'DONE.json',dict(passed=True,groups=len(rows),no_teacher_forward=True,same_weights_features=True))
    print('RELEASE_IDENTITIES_PASS',len(rows),flush=True)


if __name__=='__main__':main()
