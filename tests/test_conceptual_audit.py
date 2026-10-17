"""Regression checks for the final FIA methodology audit."""
import unittest

import pandas as pd

from src.feature_ablation import DropColumns, FEATURE_SETS, summarize_ablation


class ConceptualAuditTests(unittest.TestCase):
    def test_feature_ablation_does_not_mutate_input(self):
        original = pd.DataFrame({
            "Group": [1, 2], "Surname": ["Rossi", "Verdi"],
            "Age": [20, 40],
        })
        output = DropColumns(columns=("Group", "Surname")).fit_transform(original)
        self.assertEqual(list(output.columns), ["Age"])
        self.assertEqual(list(original.columns), ["Group", "Surname", "Age"])

    def test_feature_variants_preserve_baseline(self):
        self.assertEqual(FEATURE_SETS["baseline"], ())
        self.assertEqual(FEATURE_SETS["no_group_id_or_surname"], ("Group", "Surname"))

    def test_feature_ablation_deltas_use_model_specific_baseline(self):
        data = pd.DataFrame([
            {"model": "svc", "variant": "baseline", "fold": 1, "accuracy": .8},
            {"model": "svc", "variant": "baseline", "fold": 2, "accuracy": .6},
            {"model": "svc", "variant": "no_group_id", "fold": 1, "accuracy": .7},
            {"model": "svc", "variant": "no_group_id", "fold": 2, "accuracy": .5},
            {"model": "xgboost", "variant": "baseline", "fold": 1, "accuracy": .9},
            {"model": "xgboost", "variant": "baseline", "fold": 2, "accuracy": .9},
        ])
        out = summarize_ablation(data)
        svc = out[(out["model"] == "svc") & (out["variant"] == "no_group_id")].iloc[0]
        self.assertAlmostEqual(svc["delta_vs_baseline"], -.1)
        xgb = out[(out["model"] == "xgboost") & (out["variant"] == "baseline")].iloc[0]
        self.assertAlmostEqual(xgb["delta_vs_baseline"], 0.0)


if __name__ == "__main__":
    unittest.main()
