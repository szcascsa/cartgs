"""正式评价清单全部预先登记，固定终点，不按test挑选。"""
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    common=dict(frozen_before_nonbank=True,registered_utc=datetime.now(timezone.utc).isoformat(),
        design='all registered comparisons retained; final12000; Bank-only decisions',primary='same_k')
    groups=[('confirm','fr1_xyz',3100,'R3',['Reference59','RawPure','PhysicalFiLM','SpatialFiLM','CrossAttentionFiLM','SpatialOutputKnots'],[5100,5101]),
        ('reference_control','fr1_xyz',3100,'R3',['Reference59TrainEncoder'],[5100,5101]),
        ('controls','fr1_xyz',3100,'R3',['SpatialNoFiLM','SpatialNoFiLMCapacity','SpatialCapacityControl','SpatialLocalXYZ','SpatialKnotsBands10'],[5100,5101]),
        ('fr2_confirm','fr2_xyz',4100,'R3',['Reference59TrainEncoder','SpatialFiLM','SpatialOutputKnots'],[6100,6101]),
        ('baseline_sets','fr1_xyz',3100,'Baseline59',['Reference59TrainEncoder','SpatialFiLM','SpatialOutputKnots'],[5100,5101]),
        ('long36k','fr1_xyz',3100,'R3',['Reference59TrainEncoderLong','SpatialFiLMLong','SpatialOutputKnotsLong'],[5100,5101]),
        ('node_clock','fr1_xyz',3100,'R3',['SpatialKnotsSparseClock'],[5100,5101]),
        ('folded','fr1_xyz',3100,'R3',['SpatialKnotsFolded'],[5100,5101]),
        ('input_cause','fr1_xyz',3100,'R3',['NoWorldControl','NoShapeControl','NormalizedLocalOnly'],[5100,5101]),
        ('repeat','fr1_xyz',3100,'R3',['Reference59TrainEncoder','SpatialFiLM','SpatialOutputKnots'],[5100]),
        ('fr2_local_followup','fr2_xyz',4100,'R3',['SpatialLocalXYZ'],[6100,6101])]
    for tier,scene,source,family,names,seeds in groups:
        plan=dict(common,scene=scene,source_seed=source,source_family=family,models=[
            dict(directory=str(ROOT/'results'/tier/scene/name/str(seed))) for seed in seeds for name in names])
        path=ROOT/('plan_'+tier+'.json')
        if not path.exists():path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print('PLANS_REGISTERED',flush=True)


if __name__=='__main__':main()
