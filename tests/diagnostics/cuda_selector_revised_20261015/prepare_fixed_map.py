"""从冻结的旧在线 checkpoint 准备四场景真实固定地图，不改旧夹具。"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('scene', choices=['room', 'desk', 'xyz', 'long'])
args = parser.parse_args()
dest = Path(__file__).resolve().parent
root = dest.parents[2]
datasets = dict(room='freiburg1_room', desk='freiburg1_desk', xyz='freiburg2_xyz',
                long='freiburg3_long_office_household')
checkpoint, = (root/'results/selector_online_ab_final_20261015'/args.scene/'legacy/repeat1').glob('*_shutdown/ply')
fixture = dest/'fixed_map'/args.scene
fixture.mkdir(parents=True, exist_ok=True)
# Reuse exactly the previous parser, camera transform and image resize.
source = (dest.parent/'cuda_selector_implementation_20261015/prepare_current_fixture.py').read_text()
source = source.replace('dest = Path(__file__).resolve().parent', f'dest = Path({str(fixture)!r})')
source = source.replace('root = dest.parents[2]', f'root = Path({str(root)!r})')
old = "checkpoint = Path('/home/prmi/zwj/mycartgs/results/cudatfnewprotection/tum_rgbd/roomfixednice/6900_shutdown/ply')"
assert old in source
source = source.replace(old, f'checkpoint = Path({str(checkpoint)!r})')
source = source.replace('rgbd_dataset_freiburg1_room', 'rgbd_dataset_'+datasets[args.scene])
exec(compile(source, __file__, 'exec'))
