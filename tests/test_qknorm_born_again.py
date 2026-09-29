import unittest

import torch
from torch.nn import functional as F

from student_qknorm_rotary import ReferenceRotary
from train_born_again_19200 import recipe_for


class QKNormBornAgainTests(unittest.TestCase):
    def test_qk_norm_receives_gradients_without_future_token_leakage(self):
        torch.manual_seed(7)
        config = dict(vocab=2048, width=32, heads=4, depth=2, ffn_hidden=88,
                      context=256, residual_dropout=.05, qk_norm=True,
                      cache_weight=.05, cache_theta=13., neural_temperature=1.1)
        model = ReferenceRotary(config).eval()
        ids = torch.randint(2048, (2, 16))
        changed = ids.clone()
        changed[:, 8:] = torch.randint(2048, (2, 8))
        with torch.no_grad():
            original = model.predict_log_probs(ids)
            later_changed = model.predict_log_probs(changed)
            torch.testing.assert_close(original[:, :8], later_changed[:, :8], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(original.logsumexp(-1), torch.zeros(2, 16), atol=2e-6, rtol=0)
        loss = F.cross_entropy(model(ids).flatten(0, 1), ids.flatten())
        loss.backward()
        for block in model.blocks:
            self.assertGreater(block.q_norm.weight.grad.abs().sum().item(), 0.)
            self.assertGreater(block.k_norm.weight.grad.abs().sum().item(), 0.)

    def test_19200_step_recipe_preserves_the_existing_distillation_objective(self):
        recipe = recipe_for('student', 37, {'width': 320}, [{'seed': 29}, {'seed': 31}],
                            steps=19200, weight_decay=.05,
                            implementation='student_qknorm_rotary')
        self.assertEqual(recipe['average_steps'], [17760, 18240, 18720, 19200])
        self.assertEqual(recipe['teacher_weight'], .5)
        self.assertEqual(recipe['temperature'], 1.)
        self.assertEqual(recipe['weight_decay'], .05)
        self.assertTrue(recipe['random_initialization'])


if __name__ == '__main__':
    unittest.main()
