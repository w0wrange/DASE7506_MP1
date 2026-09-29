"""Tests for backward-compatible AdamW parameter grouping."""
import copy
import unittest

import torch

from student import build_model
from train import learning_rate_at_step, optimizer_groups


def config():
    return dict(vocab=2048, width=32, heads=4, depth=2, context=256,
                ffn_type='swiglu')


class OptimizerGroupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = build_model(config())

    def test_matrix_groups_are_complete_disjoint_and_shape_based(self):
        groups, stats = optimizer_groups(self.model, .1, 'matrix')
        decayed = groups[0]['params']
        nondecayed = groups[1]['params']
        all_grouped = decayed + nondecayed
        trainable = [parameter for parameter in self.model.parameters()
                     if parameter.requires_grad]
        self.assertEqual(len(all_grouped), len(trainable))
        self.assertEqual({id(parameter) for parameter in all_grouped},
                         {id(parameter) for parameter in trainable})
        self.assertEqual(len(all_grouped), len({id(parameter) for parameter in all_grouped}))
        self.assertTrue(all(parameter.ndim >= 2 for parameter in decayed))
        self.assertTrue(all(parameter.ndim < 2 for parameter in nondecayed))
        self.assertEqual(groups[0]['weight_decay'], .1)
        self.assertEqual(groups[1]['weight_decay'], 0.)
        self.assertEqual(stats['decayed_parameters'] + stats['nondecayed_parameters'],
                         sum(parameter.numel() for parameter in trainable))
        self.assertIn('token.weight', stats['decayed_names'])
        self.assertNotIn('head.weight', stats['decayed_names'])  # tied parameter is deduplicated
        self.assertIn('blocks.0.norm1.weight', stats['nondecayed_names'])
        self.assertIn('blocks.0.qkv.bias', stats['nondecayed_names'])

    def test_all_mode_matches_legacy_adamw_update_exactly(self):
        candidate = copy.deepcopy(self.model)
        legacy = copy.deepcopy(self.model)
        candidate_groups, stats = optimizer_groups(candidate, .1, 'all')
        candidate_optimizer = torch.optim.AdamW(candidate_groups, lr=.001,
                                                 weight_decay=0.)
        legacy_optimizer = torch.optim.AdamW(legacy.parameters(), lr=.001,
                                              weight_decay=.1)
        ids = torch.randint(0, 2048, (2, 13))
        for model, optimizer in ((candidate, candidate_optimizer),
                                 (legacy, legacy_optimizer)):
            loss = torch.nn.functional.cross_entropy(
                model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        self.assertEqual(stats['nondecayed_parameters'], 0)
        for actual, expected in zip(candidate.parameters(), legacy.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_gradient_step_decays_only_matrix_parameters(self):
        before = {name: parameter.detach().clone()
                  for name, parameter in self.model.named_parameters()}
        groups, _ = optimizer_groups(self.model, .1, 'matrix')
        optimizer = torch.optim.AdamW(groups, lr=.01, weight_decay=0.)
        for parameter in self.model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        for name, parameter in self.model.named_parameters():
            if parameter.ndim >= 2:
                torch.testing.assert_close(parameter, before[name] * .999,
                                           rtol=1e-6, atol=1e-8)
            else:
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_invalid_mode_fails(self):
        with self.assertRaises(ValueError):
            optimizer_groups(self.model, .1, 'unknown')

    def test_default_learning_rate_schedule_matches_legacy_formula(self):
        for step in (0, 1, 99, 100, 300, 600, 900, 1199):
            expected = .001 * min(1., (step + 1) / 100) * (
                .1 + .9 * .5 * (1 + torch.cos(torch.tensor(
                    torch.pi * step / 1200, dtype=torch.float64)).item())
            )
            actual = learning_rate_at_step(step, 1200, .001, 100, .1)
            self.assertAlmostEqual(actual, expected, places=18)

    def test_zero_warmup_starts_at_peak_learning_rate(self):
        self.assertEqual(learning_rate_at_step(0, 1200, .001, 0, .1), .001)


if __name__ == '__main__':
    unittest.main()
