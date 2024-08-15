import unittest

import torch
from torchmetrics.functional.classification import multilabel_precision_recall_curve

import src.metrics
from src.metrics import compute_all_metrics, compute_thresholds


class TestComputeMetrics(unittest.TestCase):
    def setUp(self):
        src.metrics.top_ks = [1, 2, 3]
        predictions_and_labels = torch.tensor([
            [[0.5, 1], [0.6, 1], [0.5, 0]],
            [[0.4, 1], [0.5, 1], [0.4, 0]],
            [[0.3, 0], [0.4, 1], [0.3, 0]],
            [[0.2, 0], [0.3, 1], [0.2, 0]]
        ]).permute(2, 0, 1)
        self.predictions = predictions_and_labels[0]
        self.labels = predictions_and_labels[1].long()

    def test_compute_tuned_metrics(self):
        pr_curve_result = multilabel_precision_recall_curve(self.predictions, self.labels, 3)
        thresholds = compute_thresholds(pr_curve_result, self.labels.shape[-1])
        results = compute_all_metrics(self.predictions, self.labels, thresholds, '')

        self.assertAlmostEqual(results['MicroF1'].item(), 0.2857, 4)
        self.assertAlmostEqual(results['MacroF1'].item(), 0.4000, 4)

        self.assertAlmostEqual(results['MicroPrecision'].item(), 1)
        self.assertAlmostEqual(results['MacroPrecision'].item(), 1)

        self.assertAlmostEqual(results['MicroRecall'].item(), 0.1667, 4)
        self.assertAlmostEqual(results['MacroRecall'].item(), 0.1250, 4)

        self.assertAlmostEqual(results['MicroAccuracy'].item(), 0.5833, 4)
        self.assertAlmostEqual(results['MacroAccuracy'].item(), 0.5833, 4)

        self.assertAlmostEqual(results['TunedMicroF1'].item(), 1)
        self.assertAlmostEqual(results['TunedMacroF1'].item(), 1)

        self.assertAlmostEqual(results['TunedMicroPrecision'].item(), 1)
        self.assertAlmostEqual(results['TunedMacroPrecision'].item(), 1)

        self.assertAlmostEqual(results['TunedMicroRecall'].item(), 1)
        self.assertAlmostEqual(results['TunedMacroRecall'].item(), 1)

        self.assertAlmostEqual(results['TunedMicroAccuracy'].item(), 1)
        self.assertAlmostEqual(results['TunedMacroAccuracy'].item(), 1)

        self.assertTrue(all(thresholds[:2] == torch.tensor([0.4, 0.3])))
        self.assertAlmostEqual(thresholds[2].item(), 0.5, 6)


if __name__ == '__main__':
    unittest.main()
