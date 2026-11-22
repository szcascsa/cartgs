"""由实际日志和计时生成失败候选清单；不修改已完成实验。"""
from pathlib import Path
import json
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
NOTES={
 'TAIL_DEVELOPMENT02':'并行有序Tile中心后缀扫描；合格阈值慢约9%，更激进质量失败。',
 'NORMAL_DEVELOPMENT':'前缀归一化/当前或下一Gaussian颜色补尾；合格阈值没有稳定收益。',
 'COMPLETION_DEVELOPMENT':'补尾移到退出后；.04最差条件PSNR损失0.522dB，失败。',
 'WARP_HALF_DEVELOPMENT':'warp私有缓存重复装载属性；FP16颜色累加没有减少主依赖链。',
 'LOG_DEVELOPMENT':'预计算log-opacity及成员阈值；224项逐位一致，但完整速度基本持平。',
 'UNROLL_DEVELOPMENT':'循环展开1/2/4/8/16；最好仅约0.7%，不足以保留。',
 'PREFIX_DEVELOPMENT':'4lane扫描8个Gaussian并行合成；Shuffle/归约开销抵消收益，约慢40%。',
 'SERIAL_TAIL_DEVELOPMENT':'每Tile一线程串行尾色；额外Workspace与预处理过重。',
 'BRANCHLESS_PACKED_DEVELOPMENT':'half2打包Alpha/T/RGB慢约20%；误改fmaf的生成器已修复，首次编译日志保留。',
 'ENCODED_DEVELOPMENT':'bit-exp线性变换提升到Tile装载；本轮有效组件，Horner重排没有进一步收益。',
 'COHERENT_DEVELOPMENT':'8x8布局与T-weight组合有效；单独布局和极小T判断存在负例。',
 'ENCODED_SIGN_DEVELOPMENT':'182组合仅在开发集锁定，然后进行独立保留验证。',
 'EXECUTION_DEVELOPMENT':'协作排序首版遗漏容量尾部sentinel，memcheck定位compactTileBits越界；未作为可接受实现。',
 'EXECUTION_DEVELOPMENT02':'修复协作排序通过边界但主要Ratio慢约10%；空Tile退出收益约1%且不稳定。联合smoke数值检查完成后CSV列不齐失败，改为统一字段输出。',
 'REGISTER_DEVELOPMENT':'PTX寄存器MMA占用128寄存器，相比原WMMA44个；补偿13us增至29us。',
 'DEVELOPMENT':'减少warp组线程和persistent任务调度；逐位一致但完整速度普遍慢4%–6%。',
}


def main():
    rows=[]
    for stage,note in NOTES.items():
        f=HERE/'reports'/(stage+'_CONTROLLER.json')
        if not f.exists():continue
        state=json.loads(f.read_text())
        rows.append(dict(stage=stage,state=state['state'],accepted=state.get('accepted',[]),rejected=state.get('rejected',[]),note=note))
    (HERE/'reports/CANDIDATE_DECISIONS.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    lines=['# 候选及失败归因','','全部结论以对应控制器、原始CSV与日志为准；单独组件不得重复算作组合新增收益。','']
    for row in rows:lines.append(f"- **{row['stage']}**（{row['state']}）：{row['note']}")
    (HERE/'reports/FAILED_CANDIDATES.md').write_text('\n'.join(lines),encoding='utf-8');print('CANDIDATE_DECISIONS_SAVED',len(rows))


if __name__=='__main__':main()
