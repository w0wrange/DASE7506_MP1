import math
import unittest

import torch
from torch.nn import functional as F

import test_contract
from student import RMSNorm, build_model, causal_cache_probs
from train import average_state, embedding_lr_groups, learning_rate_at_step, optimizer_groups


def config(**kwargs):
    return dict(vocab=2048, width=32, heads=4, depth=2, context=256,
                ffn_type='swiglu', position_encoding='rope', **kwargs)


class CacheContractTests(test_contract.ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = build_model(config(cache_weight=.1, cache_theta=5.)).eval()


class RMSContractTests(test_contract.ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = build_model(config(norm_type='rmsnorm')).eval()


class MethodsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_scaled_initialization_changes_only_residual_weights(self):
        control = build_model(config())
        torch.manual_seed(17)
        scaled = build_model(config(residual_init_scale=.5))
        for name, value in control.state_dict().items():
            factor = .5 if name.endswith(('proj.weight', 'mlp.down.weight')) else 1.
            torch.testing.assert_close(scaled.state_dict()[name], value * factor, rtol=0, atol=0)

    def test_embedding_lr_changes_only_embedding_update_scale(self):
        model = build_model(config())
        groups, _ = optimizer_groups(model, .1, 'matrix')
        groups = embedding_lr_groups(groups, model, .25)
        flattened = [p for g in groups for p in g['params']]
        self.assertEqual(len(flattened), len(set(map(id, flattened))))
        self.assertEqual(set(map(id, flattened)), set(map(id, model.parameters())))
        before = {id(p): p.detach().clone() for p in flattened}
        for g in groups:
            g['lr'] = .01 * g['lr_scale']
            for p in g['params']:
                p.grad = torch.ones_like(p)
        opt = torch.optim.SGD(groups, lr=.01, weight_decay=0.)
        opt.step()
        for g in groups:
            for p in g['params']:
                scale = .25 if p is model.token.weight else 1.
                expected = before[id(p)] - .01 * scale * (1 + g['weight_decay'] * before[id(p)])
                torch.testing.assert_close(p, expected)

    def test_wsd_endpoints_and_continuity(self):
        lr = lambda s: learning_rate_at_step(s, 1200, .0012, 100, .1, 'wsd', 800)
        self.assertAlmostEqual(lr(0), .000012)
        self.assertAlmostEqual(lr(99), .0012)
        self.assertAlmostEqual(lr(799), .0012)
        self.assertLess(lr(800), lr(799))
        self.assertAlmostEqual(lr(1199), .00012)
        self.assertTrue(all(lr(s + 1) <= lr(s) for s in range(799, 1199)))

    def test_checkpoint_average_is_arithmetic_mean_without_mutating_source(self):
        a = {'weight': torch.tensor([1., 4.]), 'counter': torch.tensor(1)}
        b = {'weight': torch.tensor([3., 2.]), 'counter': torch.tensor(2)}
        c = {'weight': torch.tensor([8., 0.]), 'counter': torch.tensor(3)}
        avg = average_state(None, a, 0)
        avg = average_state(avg, b, 1)
        avg = average_state(avg, c, 2)
        torch.testing.assert_close(avg['weight'], torch.tensor([4., 2.]))
        torch.testing.assert_close(a['weight'], torch.tensor([1., 4.]))
        self.assertEqual(avg['counter'].item(), 3)

    def test_cache_matches_explicit_prefix_retrieval_and_repeated_token_counts(self):
        features = torch.randn(2, 6, 8)
        ids = torch.tensor([[1, 3, 3, 4, 2, 5], [7, 8, 9, 8, 8, 7]])
        actual = causal_cache_probs(features, ids, 5.)
        unit = F.normalize(features, dim=-1)
        expected = torch.zeros_like(actual)
        for b in range(2):
            for t in range(1, 6):
                weights = (5 * (unit[b, :t] @ unit[b, t])).softmax(-1)
                for j in range(t):
                    expected[b, t, ids[b, j + 1]] += weights[j]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual[:, 0].count_nonzero().item(), 0)
        torch.testing.assert_close(actual[:, 1:].sum(-1), torch.ones(2, 5))
        self.assertEqual(actual[0, 1, 3].item(), 1.)

    def test_cache_no_future_leak_full_context_or_cross_call_state(self):
        model = build_model(config(cache_weight=.1)).eval()
        ids = torch.randint(2048, (2, 256))
        changed = ids.clone()
        changed[:, 128:] = (changed[:, 128:] + 51) % 2048
        with torch.no_grad():
            p = model.predict_log_probs(ids)
            q = model.predict_log_probs(changed)
            again = model.predict_log_probs(ids)
            prefix = model.predict_log_probs(ids[:, :128])
        torch.testing.assert_close(p[:, :128], q[:, :128], rtol=0, atol=0)
        torch.testing.assert_close(p, again, rtol=0, atol=0)
        torch.testing.assert_close(p[:, :128], prefix, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(p.logsumexp(-1), torch.zeros(2, 256), atol=1e-6, rtol=0)

    def test_rmsnorm_matches_reference_and_has_finite_gradients(self):
        norm = RMSNorm(8)
        x = torch.randn(2, 4, 8, requires_grad=True)
        expected = x / (x.square().mean(-1, keepdim=True) + 1e-5).sqrt()
        torch.testing.assert_close(norm(x), expected)
        norm(x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_new_variants_roundtrip_and_untied_heads_diverge_under_training(self):
        ids = torch.randint(2048, (2, 15))
        for override in ({'norm_type': 'rmsnorm'}, {'residual_init_scale': .5},
                         {'tie_embeddings': False}, {'cache_weight': .1}):
            with self.subTest(override=override):
                model = build_model(config(**override))
                opt = torch.optim.AdamW(model.parameters(), lr=.001)
                loss = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
                loss.backward()
                opt.step()
                if override.get('tie_embeddings') is False:
                    self.assertFalse(torch.equal(model.head.weight, model.token.weight))
                restored = build_model(config(**override))
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(model.predict_log_probs(ids),
                                               restored.predict_log_probs(ids), rtol=0, atol=0)

    def test_invalid_new_configurations(self):
        for override in ({'norm_type': 'bad'}, {'residual_init_scale': 0},
                         {'residual_init_scale': math.nan}, {'cache_weight': 1},
                         {'cache_weight': -.1}, {'cache_theta': math.inf}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_model(config(**override))


if __name__ == '__main__':
    unittest.main()
