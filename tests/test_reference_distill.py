import unittest
import torch
from torch.nn import functional as F
from train_reference_distill import accumulate_average, late_steps, objective, teacher_probabilities


class FixedTeacher(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32), requires_grad=False)
        self.eval()

    def forward(self, inputs):
        return self.logits.expand(*inputs.shape, -1)


class ReferenceDistillationTests(unittest.TestCase):
    def test_probability_mixture_is_not_mean_logits(self):
        teachers = [FixedTeacher([5., 0., -1.]), FixedTeacher([0., 1., 2.])]
        inputs = torch.zeros(2, 3, dtype=torch.long)
        mixed = teacher_probabilities(teachers, inputs, 1.)
        expected = .5 * (teachers[0](inputs).softmax(-1) + teachers[1](inputs).softmax(-1))
        torch.testing.assert_close(mixed, expected)
        self.assertFalse(torch.allclose(mixed, (.5 * (teachers[0](inputs) + teachers[1](inputs))).softmax(-1)))
        torch.testing.assert_close(mixed.sum(-1), torch.ones_like(inputs, dtype=torch.float32))

    def test_soft_ce_and_kl_have_same_student_gradients(self):
        torch.manual_seed(4)
        targets = torch.tensor([[0, 1], [2, 0]])
        for temperature in (1., 2.):
            teacher = torch.randn(2, 2, 3)
            q = (teacher / temperature).softmax(-1)
            logits = torch.randn_like(teacher, requires_grad=True)
            loss, _, _ = objective(logits, q, targets, temperature)
            reference = .5 * F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            reference += .5 * temperature ** 2 * F.kl_div(
                F.log_softmax(logits / temperature, -1), q, reduction='none').sum(-1).mean()
            actual = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            expected = torch.autograd.grad(reference, logits)[0]
            torch.testing.assert_close(actual, expected)

    def test_teacher_is_frozen_and_detached(self):
        teacher = FixedTeacher([1., -1., 0.])
        q = teacher_probabilities([teacher], torch.zeros(2, 2, dtype=torch.long), 2.)
        logits = torch.randn_like(q, requires_grad=True)
        objective(logits, q, torch.zeros(2, 2, dtype=torch.long), 2.)[0].backward()
        self.assertIsNone(teacher.logits.grad)
        self.assertFalse(q.requires_grad)
        teacher.train()
        with self.assertRaises(ValueError):
            teacher_probabilities([teacher], torch.zeros(1, 1, dtype=torch.long), 1.)

    def test_fp64_average_avoids_float32_cancellation(self):
        average = None
        for value in (1e8, 1., -1e8, 3.):
            average = accumulate_average(average, {'weight': torch.tensor([value])})
        self.assertEqual(average['weight'].dtype, torch.float64)
        self.assertEqual(float((average['weight'] / 4.).float()), 1.)

    def test_four_predeclared_late_states(self):
        self.assertEqual(late_steps(19200), [17760, 18240, 18720, 19200])
        self.assertEqual(late_steps(1200), [1110, 1140, 1170, 1200])
        with self.assertRaises(ValueError):
            late_steps(41)
