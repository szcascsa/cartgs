"""Anchor排序分档：投影/深度口径、预算、原网络解码和只读生命周期。"""
from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from anchor_rank_lod import RankedAnchorRenderer, anchor_projection, rank_levels
from anchor_budget import FixedBudgetRenderer, exact_budget_mask
from anchor_frustum import compact_gaussians
from selector import slot_eligibility, compensated_attributes
from selector_sampling import gumbel_selection
from renderer import prepare_render_data
from anchor_lod_test import fixture, camera


def cam(**kwargs):
    value = camera(**kwargs);value.image_width = 640;value.image_height = 480
    return value


class RankedAnchorLodTest(unittest.TestCase):
    def setUp(self):
        self.model, self.pc = fixture()
        self.base = FixedBudgetRenderer(self.model, self.pc, dict(selection_mode='flexgs'))
        self.renderer = RankedAnchorRenderer(self.base)

    def equal_data(self, a, b):
        for name, value in a['inputs'].items():
            if value is not None:self.assertTrue(torch.equal(value, b['inputs'][name]), name)

    def test_projection_and_depth_are_distinct_and_camera_relative(self):
        centers = torch.tensor([[0., 0., 2.], [0., 0., 4.]])
        sizes = torch.tensor([[.1], [2.]])
        screen, depth, near = anchor_projection(centers, sizes, cam())
        self.assertGreater(screen[1], screen[0])
        self.assertEqual(depth.tolist(), [2., 4.])
        self.assertFalse(bool(near.any()))
        moved = anchor_projection(centers, sizes, cam(z=1.))[1]
        self.assertEqual(moved.tolist(), [1., 3.])

    def test_projection_near_plane_is_conservative_and_edge_is_unclipped(self):
        centers = torch.tensor([[0., 0., .02], [1.9, 0., 2.]])
        sizes = torch.tensor([[.1], [.4]])
        screen, _, near = anchor_projection(centers, sizes, cam())
        self.assertTrue(math.isinf(float(screen[0])))
        self.assertTrue(bool(near[0]))
        original = anchor_projection(centers[1:]*torch.tensor([[0., 1., 1.]]), sizes[1:], cam())[0]
        self.assertGreaterEqual(float(screen[1]), float(original[0]))

    def test_weighted_midpoint_not_anchor_count(self):
        levels, order = rank_levels(torch.tensor([4., 3., 2., 1.]), torch.tensor([2, 8, 20, 70]), torch.arange(4))
        self.assertEqual(order.tolist(), [0, 1, 2, 3])
        self.assertEqual(levels.tolist(), [0, 0, 1, 2])

    def test_ties_use_stable_uid_even_after_reorder(self):
        uids = torch.tensor([30, 10, 20, 40])
        levels, order = rank_levels(torch.ones(4), torch.ones(4, dtype=torch.long), uids)
        self.assertEqual(uids[order].tolist(), [10, 20, 30, 40])
        permutation = torch.tensor([2, 0, 3, 1])
        other, _ = rank_levels(torch.ones(4), torch.ones(4, dtype=torch.long), uids[permutation])
        self.assertTrue(torch.equal(levels[permutation], other))

    def test_hysteresis_retains_boundary_and_reset_is_deterministic(self):
        metric, counts, uids = torch.arange(10).float(), torch.ones(10, dtype=torch.long), torch.arange(10)
        initial, _ = rank_levels(metric, counts, uids, (.2, .3, .5))
        changed, _ = rank_levels(metric, counts, uids, (.3, .2, .5), previous=initial, hysteresis=.11)
        self.assertTrue(torch.equal(changed, initial))
        direct, _ = rank_levels(metric, counts, uids, (.3, .2, .5))
        self.assertFalse(torch.equal(direct, initial))

    def test_three_groups_disjoint_cover_visible_and_strict_k(self):
        for sorting in ('projection', 'depth'):
            _, ids, stats = self.renderer.prepare(cam(), sorting=sorting)
            self.assertEqual(sum(g['candidate_gaussians'] for g in stats['lod_groups']), 40)
            self.assertEqual(sum(g['anchors'] for g in stats['lod_groups']), 4)
            self.assertEqual(len(ids), sum(min(math.floor(g['ratio']*g['candidate_gaussians']), g['eligible_candidates']) for g in stats['lod_groups']))
            self.assertEqual(ids.tolist(), sorted(set(ids.tolist())))
            self.assertNotIn(0, ids.tolist())

    def test_single_nonempty_level_matches_original_visible_ratio(self):
        for shares, ratio in [((1., 0., 0.), .5), ((0., 1., 0.), .3), ((0., 0., 1.), .1)]:
            for compensation in (False, True):
                a, ids, _ = self.renderer.prepare(cam(), shares=shares, compensation=compensation)
                b, other, _ = self.base.prepare(cam(), ratio_budget=ratio, compensation=compensation)
                self.assertTrue(torch.equal(ids, other));self.equal_data(a, b)

    def test_grouped_output_matches_independent_network_calls(self):
        result, ids, stats = self.renderer.prepare(cam())
        expected, all_ids = [], []
        for level, ratio in enumerate((.5, .3, .1)):
            rows = torch.nonzero(self.renderer.previous[self.model.anchors.rows] == level).flatten()
            pc = compact_gaussians(self.pc, rows)
            with torch.no_grad():
                _, shared, logits = self.model.forward_selection_features(pc.get_xyz, pc.get_scaling,
                    pc.get_features_dc, pc.get_opacity, torch.tensor([[ratio]]), source_rows=rows, return_logits=True)
                mask = exact_budget_mask(gumbel_selection(logits, hard=False, noise=self.base.noise[rows]),
                    slot_eligibility(self.model, pc.get_opacity, rows), math.floor(ratio*len(rows)))
                attributes = compensated_attributes(self.model, pc, mask, ratio, shared, source_rows=rows)
                if attributes is not None:
                    attributes['scaling'] = attributes['scaling'].log().exp()
                    attributes['rotation'] = pc.rotation_activation(attributes['rotation'])
                expected.append(prepare_render_data(pc, mask, attributes));all_ids.append(rows[mask])
        expected_ids = torch.cat(all_ids);order = expected_ids.argsort()
        self.assertTrue(torch.equal(ids, expected_ids[order]))
        merged = dict(inputs={name:None if value is None else torch.cat([e['inputs'][name] for e in expected])[order]
            for name, value in expected[0]['inputs'].items()})
        self.equal_data(result, merged)

    def test_four_ablations_pair_ids_and_k(self):
        for sorting in ('projection', 'depth'):
            results = {}
            for selection in ('learned', 'random'):
                for compensation in (False, True):
                    data, ids, stats = self.renderer.prepare(cam(), sorting=sorting, selection=selection, compensation=compensation)
                    results[selection, compensation] = ids
                    if not compensation:self.equal_data(data, prepare_render_data(compact_gaussians(self.pc, ids)))
            self.assertEqual(len({len(ids) for ids in results.values()}), 1)
            for selection in ('learned', 'random'):
                self.assertTrue(torch.equal(results[selection, False], results[selection, True]))

    def test_debug_colors_do_not_replace_quality_inputs(self):
        a, ids, _ = self.renderer.prepare(cam(), colorize=True)
        b, other, _ = self.renderer.prepare(cam())
        self.assertEqual(a['lod_colors'].shape, (len(ids), 3))
        self.assertTrue(torch.equal(ids, other));self.equal_data(a, b)

    def test_empty_map_view_and_unavailable_slots(self):
        data, ids, _ = self.renderer.prepare(cam(x=1000.))
        self.assertEqual(len(ids), 0)
        self.assertEqual(len(data['inputs']['means3D']), 0)
        model, pc = fixture(empty=True)
        r = RankedAnchorRenderer(FixedBudgetRenderer(model, pc, dict(selection_mode='flexgs')))
        self.assertEqual(len(r.prepare(cam())[1]), 0)
        with torch.no_grad():
            self.model.anchors.slot_inactive.fill_(True);self.pc._opacity.fill_(-20)
        self.base.rebuild()
        self.assertEqual(len(self.renderer.prepare(cam())[1]), 0)

    def test_validation_and_rebuild(self):
        for options in [dict(sorting='distance'), dict(ratios=[.1,.3,.5]), dict(shares=[.1,.1,.1]),
                        dict(ratios=[True,.3,.1]), dict(hysteresis=True), dict(hysteresis=float('nan')),
                        dict(shares=[-.1,.6,.5]), dict(colorize=1)]:
            with self.assertRaises(ValueError):self.renderer.prepare(cam(), **options)
        with torch.no_grad():self.model.anchors.compensation_features_0.add_(.01)
        with self.assertRaises(RuntimeError):self.renderer.prepare(cam())
        self.base.rebuild();self.renderer.prepare(cam())

    def test_no_model_or_rng_changes(self):
        state = {name:value.clone() for name,value in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        self.renderer.prepare(cam(), hysteresis=.02);self.renderer.prepare(cam(z=.1), sorting='depth')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, state[name]) for name,value in self.model.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))


if __name__ == '__main__':
    torch.set_num_threads(2);unittest.main()
