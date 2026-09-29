"""The submitted CPU implementation must preserve trained model predictions."""
import unittest

import torch

from student_qknorm_rotary import ReferenceRotary as TrainedRotary
from student_qknorm_rotary_inference import ReferenceRotary as InferenceRotary


class FinalInferenceTests(unittest.TestCase):
    def test_complex_rope_preserves_full_probability_distribution(self):
        torch.manual_seed(37)
        config = dict(vocab=2048, width=48, heads=6, depth=2,
                      ffn_hidden=88, context=256, residual_dropout=.05,
                      qk_norm=True, cache_weight=.05, cache_theta=13.,
                      neural_temperature=1.1)
        trained = TrainedRotary(config).eval()
        inference = InferenceRotary(config).eval()
        inference.load_state_dict(trained.state_dict())
        ids = torch.randint(2048, (2, 32))
        with torch.no_grad():
            expected = trained.predict_log_probs(ids)
            actual = inference.predict_log_probs(ids)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual.logsumexp(-1), torch.zeros(2, 32),
                                   atol=2e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
