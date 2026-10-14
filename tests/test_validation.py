"""VAL-01 tests: unseen validation rows cannot change fitted preprocessing."""
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.fold_preprocessing import FoldSafePreprocessor
from src.model import ModelConfig, _build_pipeline_for_model, build_model_zoo


def passengers(n=12):
    rows = []
    for i in range(n):
        rows.append({
            "PassengerId": f"{i+1:04d}_01",
            "HomePlanet": "Earth" if i % 2 else "Europa",
            "CryoSleep": False,
            "Cabin": f"G/{i+1}/S",
            "Destination": "TRAPPIST-1e",
            "Age": float(10 + i),
            "VIP": False,
            "RoomService": float(i),
            "FoodCourt": float(i * 2),
            "ShoppingMall": 0.0,
            "Spa": 0.0,
            "VRDeck": 0.0,
            "Name": f"Person{i} Family{i}",
        })
    return pd.DataFrame(rows)


class FoldSafeTests(unittest.TestCase):
    def test_holdout_age_cannot_change_fitted_training_median(self):
        train = passengers(6)
        train.loc[0, "Age"] = np.nan
        a = FoldSafePreprocessor().fit(train)
        validation = passengers(2)
        validation["Age"] = [np.nan, 100000.0]
        transformed = a.transform(validation)
        self.assertEqual(transformed.loc[0, "Age"], a.age_median_)
        self.assertEqual(a.age_median_, 13.0)

    def test_unseen_group_does_not_count_validation_rows(self):
        train = passengers(3)
        validation = passengers(2)
        validation["PassengerId"] = ["9999_01", "9999_02"]
        output = FoldSafePreprocessor().fit(train).transform(validation)
        self.assertEqual(output["Group_size"].tolist(), [1, 1])
        self.assertEqual(output["Solo"].tolist(), [1, 1])

    def test_validation_expense_does_not_update_training_median(self):
        train = passengers(4)
        train["RoomService"] = [10., 20., 30., 40.]
        validation = passengers(2)
        validation["RoomService"] = [np.nan, 999999.]
        validation["HomePlanet"] = "Mars"  # no matching training subgroup
        prep = FoldSafePreprocessor().fit(train)
        out = prep.transform(validation)
        self.assertEqual(prep.spending_medians_["RoomService"], 25.)
        self.assertEqual(out.loc[0, "RoomService"], 25.)

    def test_cross_validation_pipeline_accepts_raw_rows(self):
        X = passengers(20)
        y = pd.Series([0, 1] * 10)
        cfg = ModelConfig(model_names=("logistic_regression",), n_jobs=1)
        pipeline = _build_pipeline_for_model(X, "logistic_regression", build_model_zoo(cfg)["logistic_regression"])
        scores = cross_validate(
            pipeline, X, y, cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
            scoring="accuracy", n_jobs=1, error_score="raise",
        )
        self.assertEqual(len(scores["test_score"]), 2)
        self.assertTrue(np.isfinite(scores["test_score"]).all())


if __name__ == "__main__":
    unittest.main()
