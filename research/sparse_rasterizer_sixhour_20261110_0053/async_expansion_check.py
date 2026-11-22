"""在原版oracle全部完成后连续提交扩容/分辨率变化，排除oracle隐藏同步。"""
from harness import *
from final_benchmark import native_modules
from final_edges import packet as make_packet
from runtime import SparseRenderer
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    torch.manual_seed(71319);bg=torch.tensor([.13,-0.,.76],device='cuda');cases=[]
    sizes=[1,257,33,1025,513,4097,255,8193,31,513,1025,33]
    for index,P in enumerate(sizes):
        camera=MiniCam(97+index*3,65+(index%3)*2,1.2,1.,np.eye(4))
        packet=make_packet(P,['random','huge','thin','overlap'][index%4])
        reference=render_prepared(camera,packet,bg,return_radii=True);cases.append((camera,packet,reference))
    torch.cuda.synchronize();pending=[]
    c=SparseRenderer(**config['regular']);g=GraphRenderer(**config['graph'])
    # 未预留大容量；连续改变W/H/P，旧图和旧Workspace失效。
    for index,(camera,packet,reference) in enumerate(cases):
        for name,engine in [('C',c),('G',g)]:pending.append((index,name,engine.render(camera,packet,bg),reference))
    torch.cuda.synchronize();rows=[]
    for index,name,result,reference in pending:
        row=dict(case=index,variant=name,K=len(result[1]),rgb_bitwise=bits_equal(result[0],reference['render']),
            radii_bitwise=bits_equal(result[1],reference['radii']))
        rows.append(row);assert row['rgb_bitwise'] and row['radii_bitwise'],row
    write_csv(output/'checks.csv',rows);write_json(output/'SUMMARY.json',dict(passed=True,comparisons=len(rows),
        explicit_synchronizations=2,oracle_before_candidate_burst=True,changing_resolution_and_capacity=True,
        all_outputs_retained=True,graph_stats=g.native.stats()))
    print('ASYNC_EXPANSION_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
