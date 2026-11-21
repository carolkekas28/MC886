"""Small checks for independent gradients and safe reuse of recorded experiments."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import convergencia as experiment


class ConvergenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.raw = {"Synthetic": (rng.integers(0, 256, (80, 4), dtype=np.uint8),
                                   np.tile(np.arange(10), 8))}
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.options = dict(lambda_grid=[0, 1e-3], n_splits=2,
                            num_threads=2, checkpoint_every=2)

    def test_independent_gradients_and_objectives(self):
        candidates = experiment._candidates(self.options["lambda_grid"])
        models = experiment._CandidateModels(4, 10, candidates, 42)
        X = torch.tensor(self.raw["Synthetic"][0][:30] / 255, dtype=torch.float32)
        y = torch.tensor(self.raw["Synthetic"][1][:30])
        metrics = experiment._evaluate(models, X, y, X, y, candidates)
        for i, (name, strength) in enumerate(candidates):
            torch.manual_seed(42)
            model = nn.Linear(4, 10)
            loss = nn.functional.cross_entropy(model(X), y)
            if name == "L1":
                loss = loss + strength * model.weight.abs().sum()
            elif name == "L2":
                loss = loss + strength * model.weight.square().sum()
            loss.backward()
            torch.testing.assert_close(models.weight.grad[i], model.weight.grad,
                                       rtol=2e-6, atol=1e-7)
            torch.testing.assert_close(models.bias.grad[i], model.bias.grad,
                                       rtol=2e-6, atol=1e-7)
            torch.testing.assert_close(torch.tensor(metrics["train_objective"][i]), loss)

    def test_resume_is_identical_and_complete_cache_does_not_retrain(self):
        path, _ = experiment.run_experiment(self.raw, self.root / "resume", max_iter=4,
                                             **self.options)
        path, resumed = experiment.run_experiment(self.raw, self.root / "resume", max_iter=9,
                                                   **self.options)
        _, continuous = experiment.run_experiment(self.raw, self.root / "continuous", max_iter=9,
                                                   **self.options)
        columns = [name for name in resumed if name != "elapsed_seconds"]
        pd.testing.assert_frame_equal(resumed[columns], continuous[columns], check_exact=True)
        times = {p: p.stat().st_mtime_ns for p in path.glob("*/fold_*/*.csv")}
        experiment.run_experiment(self.raw, self.root / "resume", max_iter=9, **self.options)
        self.assertEqual(times, {p: p.stat().st_mtime_ns for p in times})
        self.assertEqual(len(resumed), 2 * 3 * 10)
        self.assertTrue(np.isfinite(resumed.select_dtypes("number").to_numpy()).all())

    def test_uncommitted_rows_are_ignored_and_analysis_has_real_fold_counts(self):
        path, history = experiment.run_experiment(self.raw, self.root, max_iter=4, **self.options)
        csv = path / "Synthetic/fold_1/iterations.csv"
        committed = pd.read_csv(csv, float_precision="round_trip")
        extras = committed[committed.iteration == 4].assign(iteration=5)
        experiment._atomic_csv(csv, pd.concat([committed, extras]))
        loaded = experiment.load_history(path)
        self.assertEqual(len(loaded), len(history))
        _, resumed = experiment.run_experiment(self.raw, self.root, max_iter=9, **self.options)
        _, reference = experiment.run_experiment(self.raw, self.root / "reference", max_iter=9,
                                                  **self.options)
        columns = [name for name in resumed if name != "elapsed_seconds"]
        pd.testing.assert_frame_equal(resumed[columns], reference[columns], check_exact=True)
        aggregate, selected, diagnostics = experiment.summarize(resumed, path,
                                                                budgets=[4, 9], window=3)
        self.assertTrue((aggregate.folds == 2).all())
        self.assertEqual(len(aggregate), 2 * 2 * 10)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(diagnostics), 2 * 2 * 2)
        baseline = resumed[(resumed.regularizer == "Baseline") & (resumed.iteration == 9)]
        aggregated = aggregate[(aggregate.regularizer == "L1") & (aggregate["lambda"] == 0)
                               & (aggregate.iteration == 9)].iloc[0]
        self.assertAlmostEqual(aggregated.val_accuracy_mean, baseline.val_accuracy.mean())
        self.assertAlmostEqual(aggregated.val_accuracy_std, baseline.val_accuracy.std(ddof=1))


if __name__ == "__main__":
    unittest.main()
