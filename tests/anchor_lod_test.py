"""混合LoD的调度、身份、冻结缓存与原选择/补偿路径逐值回归。"""
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from anchor_lod import MixedAnchorLodRenderer, schedule_levels
from anchor_frustum import anchor_frustum_mask, compact_gaussians
from gaussian_model import GaussianModel
from renderer import prepare_render_data
from selector import AnchorSelectorNetwork, compensated_attributes, slot_eligibility
from spatial_compensation import SpatialCompensation


def fixture(empty=False):
    torch.manual_seed(73)
    a, n = (0, 0) if empty else (6, 50)
    model = AnchorSelectorNetwork(a, n, gaussian_attributes=True, explicit_blocks=True, feature_dim=0)
    anchors = model.anchors
    if not empty:
        anchors.centers.copy_(torch.tensor([[0., 0., 2.], [0., 0., 4.], [0., 0., 8.],
                                            [0., 0., 16.], [100., 0., 4.], [0., 0., 5.]]))
        anchors.rows.copy_(torch.arange(5).repeat_interleave(10))
    anchors.offset_scales.fill_(.2)
    anchors.compensation_values = [.2, .4, .6, .8]
    anchors.compensation_residual_mode = 1
    anchors.register_buffer('compensation_ratio_knots', torch.tensor(anchors.compensation_values, dtype=torch.float64))
    anchors.register_buffer('slot_inactive', torch.zeros(n, dtype=torch.bool))
    anchors.register_buffer('slot_recovered', torch.zeros(n, dtype=torch.bool))
    for j in range(4):
        anchors.register_parameter('compensation_features_' + str(j), nn.Parameter(torch.randn(a, 48)*.02))
    model.spatial_compensation = SpatialCompensation(fourier_bands=0, ratio_feature_width=48,
        ratio_code=True, ratio_bounds=(.2, .8), offset_only=True)
    model.spatial_compensation.normalization_ready.fill_(1)
    model.spatial_compensation.register_buffer('softplus_scale_schema', torch.tensor(1))
    for head in model.spatial_compensation.heads.values():
        nn.init.normal_(head[-1].weight, 0, .01)
        nn.init.normal_(head[-1].bias, 0, .005)
    model.eval()
    pc = GaussianModel(3)
    pc.active_sh_degree = 3
    pc._xyz = anchors.centers[anchors.rows].clone()
    pc._xyz[:, 0] += torch.linspace(-.02, .02, n)
    pc._scaling = torch.full((n, 3), math.log(.01))
    pc._rotation = torch.zeros(n, 4);pc._rotation[:, 0] = 1
    pc._features_dc = torch.randn(n, 1, 3)*.1
    pc._features_rest = torch.randn(n, 15, 3)*.001
    pc._opacity = torch.full((n, 1), -.2)
    if n:
        pc._opacity[0] = -10
        anchors.slot_inactive[0] = True
    return model, pc


def camera(x=0., z=0.):
    view = torch.eye(4);view[3, 0] = -x;view[3, 2] = -z
    projection = torch.zeros(4, 4)
    projection[0, 0] = projection[1, 1] = 1
    projection[2, 2] = 100/99.99;projection[3, 2] = -.01*100/99.99;projection[2, 3] = 1
    return SimpleNamespace(camera_center=torch.tensor([x, 0., z]), world_view_transform=view,
        full_proj_transform=view@projection, FoVy=math.pi/2, znear=.01)


class AnchorLodTest(unittest.TestCase):
    def setUp(self):
        self.model, self.pc = fixture()
        self.renderer = MixedAnchorLodRenderer(self.model, self.pc, dict(selection_mode='flexgs'),
                                               [1., 3., 6., 12.], screen_guard=False)

    def equal_data(self, actual, expected):
        self.assertEqual(actual['sh_degree'], expected['sh_degree'])
        for key, value in actual['inputs'].items():
            if value is None:self.assertIsNone(expected['inputs'][key])
            else:self.assertTrue(torch.equal(value, expected['inputs'][key]), key)

    def test_threshold_validation(self):
        for distances in ([1., 2.], [1., 1., 2., 3.], [0., 1., 2., 3.], [1., 2., 3., float('nan')]):
            with self.assertRaises(ValueError):
                MixedAnchorLodRenderer(self.model, self.pc, dict(selection_mode='flexgs'), distances)

    def test_distance_levels_and_hysteresis(self):
        limits = torch.tensor([2., 4.])
        schedule = lambda d, p: schedule_levels(torch.tensor(d), torch.zeros(len(d)), torch.tensor(p), limits, None, .1)
        self.assertEqual(schedule([1., 3., 5.], [-1, -1, -1]).tolist(), [0, 1, 2])
        self.assertEqual(schedule([2.1, 1.9, 2.3, 1.7], [0, 1, 0, 1]).tolist(), [0, 1, 1, 0])
        self.assertEqual(schedule([10., .1], [0, 2]).tolist(), [2, 0])

    def test_screen_guard_and_hysteresis(self):
        limits, screen = torch.tensor([2., 4.]), torch.tensor([.2, .1])
        result = schedule_levels(torch.full((3,), 10.), torch.tensor([.3, .15, .05]),
                                 torch.full((3,), -1), limits, screen, .1)
        self.assertEqual(result.tolist(), [0, 1, 2])
        result = schedule_levels(torch.full((4,), 10.), torch.tensor([.19, .21, .17, .23]),
                                 torch.tensor([0, 1, 0, 1]), limits, screen, .1)
        self.assertEqual(result.tolist(), [0, 1, 1, 0])

    def test_nominal_culling_and_empty_anchor(self):
        visible = anchor_frustum_mask(self.model.anchors.centers, self.model.anchors.offset_scales,
                                      camera().full_proj_transform, self.renderer.occupied)
        self.assertEqual(visible.tolist(), [True, True, True, True, False, False])
        _, ids, stats = self.renderer.prepare(camera())
        self.assertEqual(stats['visible_anchors'], 4)
        self.assertEqual(stats['candidate_gaussians'], 40)
        self.assertTrue((ids < 40).all())
        self.assertEqual(self.renderer.previous_levels[4:].tolist(), [-1, -1])

    def test_all_full_matches_same_nominal_reference(self):
        visible = self.renderer.occupied
        actual, ids, _ = self.renderer._assemble(visible, torch.zeros(6, dtype=torch.long))
        self.equal_data(actual, prepare_render_data(self.pc))
        self.assertEqual(ids.tolist(), list(range(50)))  # Full retains the inactive row too.

    def test_every_fixed_level_matches_original_formula_and_decode(self):
        model, pc, r = self.model, self.pc, self.renderer
        for level, ratio in enumerate(r.ratios, 1):
            with torch.no_grad():
                _, shared, logits = model.forward_selection_features(pc.get_xyz, pc.get_scaling,
                    pc.get_features_dc, pc.get_opacity, pc.get_xyz.new_tensor([[ratio]]), return_logits=True)
                scores = (logits+r.noise(level)).softmax(-1)[:, 1]
                eligible = slot_eligibility(model, pc.get_opacity)
                scores[~eligible] = -float('inf')
                expected_mask = eligible & (scores >= scores.sort().values[int((1-ratio)*(len(scores)-1))])
                attributes = compensated_attributes(model, pc, expected_mask, ratio, shared)
                attributes['scaling'] = attributes['scaling'].log().exp()
                attributes['rotation'] = pc.rotation_activation(attributes['rotation'])
                expected = prepare_render_data(pc, expected_mask, attributes)
                actual, ids, _ = r._assemble(r.occupied, torch.full((6,), level))
            self.equal_data(actual, expected)
            self.assertTrue(torch.equal(ids, torch.nonzero(expected_mask).flatten()))

    def test_mixed_levels_preserve_unique_global_ids_and_matching_attributes(self):
        r = self.renderer
        levels = torch.tensor([0, 1, 3, 4, 2, -1])
        actual, ids, _ = r._assemble(r.occupied, levels)
        expected_ids = torch.tensor([i for i in range(50) if bool(r.masks[levels[r.owners[i]], i])])
        self.assertTrue(torch.equal(ids, expected_ids))
        self.assertEqual(len(ids), len(torch.unique(ids)))
        for output_row, global_row in enumerate(ids.tolist()):
            level = int(levels[r.owners[global_row]])
            self.assertTrue(torch.equal(actual['inputs']['means3D'][output_row], r.attributes[level, global_row, :3]))
            self.assertTrue(torch.equal(actual['inputs']['shs'][output_row], self.pc.get_features_rest[global_row]))

    def test_cached_prepare_has_no_network_calls_or_state_changes(self):
        from unittest.mock import patch
        before = {n: x.clone() for n, x in self.model.state_dict().items()}
        with patch.object(self.model, 'forward_selection_features', side_effect=AssertionError('network called')):
            for _ in range(3):self.renderer.prepare(camera())
        self.assertTrue(all(torch.equal(x, before[n]) for n, x in self.model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_build_preserves_rng_and_slot_recovery_state(self):
        self.model.anchors.slot_inactive[1] = True  # High opacity: eligible, without mutating flags.
        before = {n: x.clone() for n, x in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        self.renderer.rebuild_cache()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(x, before[n]) for n, x in self.model.state_dict().items()))
        self.assertTrue(self.renderer.eligible[1])

    def test_parameter_and_topology_updates_invalidate_cache(self):
        with torch.no_grad():self.model.soft_net.bias.add_(.01)
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):self.renderer.prepare(camera())
        self.renderer.rebuild_cache()
        self.renderer.prepare(camera())
        with torch.no_grad():self.model.anchors.block_uids.add_(100)
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):self.renderer.prepare(camera())

    def test_master_replacement_and_reorder_require_rebuild(self):
        order = torch.arange(49, -1, -1)
        for name in ['_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest']:
            setattr(self.pc, name, getattr(self.pc, name)[order])
        self.model.anchors.rows.copy_(self.model.anchors.rows[order])
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):self.renderer.prepare(camera())
        self.renderer.rebuild_cache()
        self.assertTrue((self.renderer.previous_levels == -1).all())
        _, ids, _ = self.renderer.prepare(camera())
        self.assertTrue((self.model.anchors.rows[ids] < 4).all())

    def test_near_plane_promotes_full(self):
        self.renderer.prepare(camera(z=1.99))
        self.assertEqual(int(self.renderer.previous_levels[0]), 0)

    def test_variable_member_counts_and_no_eligible_compressed_points(self):
        rows = torch.tensor([0, 10, 11, 20, 21, 22, 23, 24, 30, 31, 32])
        pc = compact_gaussians(self.pc, rows)
        self.model.anchors.rows = self.model.anchors.rows[rows]
        self.model.anchors.slot_inactive = torch.ones(len(rows), dtype=torch.bool)
        self.model.anchors.slot_recovered = torch.zeros(len(rows), dtype=torch.bool)
        pc._opacity.fill_(-10)
        renderer = MixedAnchorLodRenderer(self.model, pc, dict(selection_mode='flexgs'), [1., 3., 6., 12.])
        self.assertEqual(torch.bincount(renderer.owners).tolist(), [1, 2, 5, 3])
        data, ids, _ = renderer._assemble(renderer.occupied, torch.ones(6, dtype=torch.long))
        self.assertEqual(len(ids), 0)
        data, ids, _ = renderer._assemble(renderer.occupied, torch.zeros(6, dtype=torch.long))
        self.assertEqual(len(ids), len(rows))

    def test_empty_view_and_empty_map(self):
        data, ids, stats = self.renderer.prepare(camera(x=1000.))
        self.assertEqual(len(ids), 0)
        self.assertEqual(data['inputs']['means3D'].shape, (0, 3))
        self.assertEqual(stats['visible_anchors'], 0)
        model, pc = fixture(empty=True)
        empty = MixedAnchorLodRenderer(model, pc, dict(selection_mode='flexgs'), [1., 3., 6., 12.])
        data, ids, stats = empty.prepare(camera())
        self.assertEqual(len(ids), 0)
        self.assertEqual(stats['active_gaussians'], 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
