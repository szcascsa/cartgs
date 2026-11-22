"""直接核验执行入口实际消费的FP32 Selector、Student参数和Feature表。"""
from joint_final_study import *
import hashlib


def tensors(values):
    return [dict(shape=list(value.shape),dtype=str(value.dtype),elements=value.numel(),
        sha256=hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()) for value in values]


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[]
    config=json.loads((HERE/'reports/FROZEN_STUDENT.json').read_text())['config']
    for spec in student_specs():
        source,student,path=load_case(spec);networks=network_variants(source,student)
        final=FinalStudentRenderer(source,student,182,**config)
        engines={name:item[0] for name,item in networks.items()};engines['final']=final.network
        expected_selector=tensors(engines['tensor_best']._selector)
        expected_student=tensors(engines['tensor_best']._compensator)
        expected_features=tensors(engines['tensor_best']._base.tables)
        records={}
        for name,engine in engines.items():
            actual_selector=tensors(engine._selector);actual_student=tensors(engine._compensator);actual_features=tensors(engine._base.tables)
            assert actual_selector==expected_selector,(spec['id'],name,'Selector')
            assert actual_student==expected_student,(spec['id'],name,'Student')
            assert actual_features==expected_features,(spec['id'],name,'Feature')
            records[name]=dict(selector=True,student=True,features=True)
        rows.append(dict(**identity(spec,source,student,path),selector_execution=expected_selector,
            selector_parameter_count=sum(x['elements'] for x in expected_selector),student_execution=expected_student,
            feature_tables=expected_features,execution_backends=records,student_fingerprint_matches=True))
        del engines,networks,final,source,student;gc.collect();torch.cuda.empty_cache()
    save(out/'EXECUTION_IDENTITIES.json',rows);save(out/'DONE.json',dict(passed=True,groups=len(rows),models_same=True))
    print('EXECUTION_IDENTITY_PASS',len(rows),flush=True)


if __name__=='__main__':main()
