"""固定预算的真实网络数据流、严格K和只读状态回归。"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from anchor_budget import FixedBudgetRenderer, exact_budget_mask
from anchor_lod_test import fixture, camera


class AnchorBudgetTest(unittest.TestCase):
    def setUp(self):
        self.model, self.pc = fixture()
        self.renderer = FixedBudgetRenderer(self.model, self.pc, dict(selection_mode='flexgs'))

    def test_exact_count_zero_ties_and_stable_row_order(self):
        scores = torch.ones(17)
        eligible = torch.arange(17) % 2 == 0
        for b in [0, 1, 3, 8, 9, 30]:
            mask = exact_budget_mask(scores, eligible, b)
            self.assertEqual(torch.nonzero(mask).flatten().tolist(), torch.nonzero(eligible).flatten()[:b].tolist())

    def test_ratio_uses_visible_population_and_strict_budget(self):
        for b in [0, 1, 7, 39, 40, 100]:
            data, ids, stats = self.renderer.prepare(camera(), b)
            self.assertEqual(stats['candidate_gaussians'], 40)
            self.assertEqual(stats['eligible_candidates'], 39)
            self.assertEqual(len(ids), min(b, 39))
            self.assertEqual(len(data['inputs']['means3D']), min(b, 39))
            self.assertEqual(stats['ratio'], min(1., b/40) if b else None)
            self.assertFalse(bool((ids == 0).any()))
            self.assertTrue(torch.equal(ids, ids.sort().values))

    def test_both_networks_receive_same_dynamic_ratio(self):
        selector_ratios, compensation_ratios = [], []
        original = self.model.forward_selection_features
        def selection(*args, **kwargs):
            selector_ratios.append(float(args[4].item()))
            return original(*args, **kwargs)
        hook = self.model.spatial_compensation.register_forward_pre_hook(
            lambda module, args: compensation_ratios.append(float(args[7])))
        with patch.object(self.model, 'forward_selection_features', side_effect=selection):
            self.renderer.prepare(camera(), 10)
            self.renderer.prepare(camera(), 20)
        hook.remove()
        self.assertEqual(selector_ratios, [.25, .5])
        self.assertEqual(compensation_ratios, [.25, .5])

    def test_ratio_budget_uses_visible_count_and_preserves_requested_ratio(self):
        for ratio in [0., .01, .125, .333, .5, 1.]:
            data, ids, stats = self.renderer.prepare(camera(), ratio_budget=ratio)
            self.assertEqual(len(ids), min(int(ratio*40), 39))
            self.assertEqual(stats['ratio'], ratio if ratio else None)
            self.assertEqual(stats['budget_unit'], 'ratio')
            self.assertEqual(stats['requested_budget'], int(ratio*40))

    def test_global_ratio_matches_original_inference_without_frustum(self):
        from selector_sampling import flexgs_inference_mask
        from selector import slot_eligibility
        with torch.no_grad():
            _, _, logits = self.model.forward_selection_features(self.pc.get_xyz, self.pc.get_scaling,
                self.pc.get_features_dc, self.pc.get_opacity, torch.tensor([[.4]]), return_logits=True)
            expected, _, _ = flexgs_inference_mask(logits, .4,
                eligible=slot_eligibility(self.model, self.pc.get_opacity), noise=self.renderer.noise)
        with patch('anchor_budget.anchor_frustum_mask', side_effect=AssertionError('frustum used')):
            _, ids, stats = self.renderer.prepare(None, global_ratio=.4)
        self.assertTrue(torch.equal(ids, torch.nonzero(expected).flatten()))
        self.assertEqual(stats['candidate_gaussians'], 50)
        self.assertEqual(stats['selection_scope'], 'global')

    def test_global_cache_is_bounded_and_invalidated_by_model_changes(self):
        data, ids, stats = self.renderer.prepare(None, global_ratio=.4)
        with patch.object(self.model, 'forward_selection_features', side_effect=AssertionError('network called')):
            cached, other_ids, repeated = self.renderer.prepare(None, global_ratio=.4)
        self.assertIs(data, cached)
        self.assertTrue(torch.equal(ids, other_ids))
        self.assertTrue(repeated['cache_hit'])
        for i in range(1, 12):self.renderer.prepare(None, global_ratio=i/12)
        self.assertEqual(len(self.renderer.global_cache), 8)
        with torch.no_grad():next(self.model.parameters()).add_(.001)
        with self.assertRaises(RuntimeError):self.renderer.prepare(None, global_ratio=.4)
        self.renderer.rebuild()
        self.assertEqual(len(self.renderer.global_cache), 0)

    def test_ratio_validation_and_exclusive_budget_inputs(self):
        for value in [-.1, 1.1, True, '0.4', float('nan'), float('inf')]:
            with self.assertRaises(ValueError):self.renderer.prepare(camera(), ratio_budget=value)
            with self.assertRaises(ValueError):self.renderer.prepare(None, global_ratio=value)
        with self.assertRaises(ValueError):self.renderer.prepare(camera(), 5, ratio_budget=.4)
        with self.assertRaises(ValueError):self.renderer.prepare(None, global_ratio=0)

    def test_four_ablations_preserve_k_and_compensation_pair_ids(self):
        from renderer import prepare_render_data
        from anchor_frustum import compact_gaussians
        for options in [dict(budget=11), dict(ratio_budget=.3), dict(global_ratio=.4)]:
            results={}
            for selection in ('learned','random'):
                for compensation in (True,False):
                    data,ids,stats=self.renderer.prepare(camera(),selection=selection,compensation=compensation,**options)
                    results[selection,compensation]=ids
                    if not compensation:
                        expected=prepare_render_data(compact_gaussians(self.pc,ids))
                        for name,value in data['inputs'].items():
                            if value is not None:self.assertTrue(torch.equal(value,expected['inputs'][name]),name)
            self.assertEqual(len({len(ids) for ids in results.values()}),1)
            for selection in ('learned','random'):
                self.assertTrue(torch.equal(results[selection,True],results[selection,False]))

    def test_random_without_compensation_skips_both_networks(self):
        with (patch.object(self.model,'forward_selection_features',side_effect=AssertionError('selector called')),
              patch('selector.compensated_attributes',side_effect=AssertionError('compensator called'))):
            _,ids,stats=self.renderer.prepare(camera(),11,selection='random',compensation=False)
        self.assertEqual(len(ids),11)
        self.assertEqual(stats['compensation_ms'],0.)
        self.assertEqual(stats['ablation_group'],'D')

    def test_repeated_query_is_identical_without_model_or_rng_changes(self):
        before = {n: t.clone() for n, t in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        a, ids_a, _ = self.renderer.prepare(camera(), 11)
        b, ids_b, _ = self.renderer.prepare(camera(), 11)
        self.assertTrue(torch.equal(ids_a, ids_b))
        for k, t in a['inputs'].items():
            if t is not None:self.assertTrue(torch.equal(t, b['inputs'][k]), k)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(t, before[n]) for n, t in self.model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_zero_budget_and_empty_view_skip_network(self):
        with patch.object(self.model, 'forward_selection_features', side_effect=AssertionError('network called')):
            for cam, budget in [(camera(), 0), (camera(x=1000.), 100)]:
                _, ids, stats = self.renderer.prepare(cam, budget)
                self.assertEqual(len(ids), 0)
                self.assertIsNone(stats['ratio'])

    def test_empty_map(self):
        model, pc = fixture(empty=True)
        r = FixedBudgetRenderer(model, pc, dict(selection_mode='flexgs'))
        _, ids, stats = r.prepare(camera(), 100)
        self.assertEqual(len(ids), 0)
        self.assertEqual(stats['candidate_gaussians'], 0)

    def test_invalid_budgets_rejected(self):
        for b in [-1, 1.5, True, None, float('inf')]:
            with self.assertRaises(ValueError):self.renderer.prepare(camera(), b)

    def test_slot_shortfall_is_explicit(self):
        with torch.no_grad():
            self.pc._opacity.fill_(-10)
            self.model.anchors.slot_inactive.fill_(True)
        self.renderer.rebuild()
        _, ids, stats = self.renderer.prepare(camera(), 10)
        self.assertEqual(len(ids), 0)
        self.assertEqual(stats['ratio'], .25)
        self.assertEqual(stats['budget_shortfall'], 10)

    def test_state_updates_require_rebuild(self):
        with torch.no_grad():self.model.soft_net.bias.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'changed'):self.renderer.prepare(camera(), 10)
        self.renderer.rebuild()
        self.renderer.prepare(camera(), 10)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
