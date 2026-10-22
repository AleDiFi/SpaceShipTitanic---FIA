"""EXP-02 regression tests for categorical fill and group-aware validation."""
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing import PreprocessorePasseggeri
from src.robustness_experiment import development_partition, summarize_results


def passenger_frame():
    return pd.DataFrame({
        "PassengerId": ["0001_01", "0001_02", "0002_01", "0003_01"],
        "HomePlanet": ["Earth", None, "Europa", "Earth"],
        "CryoSleep": pd.Series([True, None, False, None], dtype=object),
        "Cabin": ["G/1/S", None, "A/3/P", "G/5/S"],
        "Destination": ["TRAPPIST-1e", None, "55 Cancri e", None],
        "Age": [20.0, np.nan, 25.0, 30.0],
        "VIP": pd.Series([False, None, False, True], dtype=object),
        "RoomService": [0.0, np.nan, 10.0, 0.0],
        "FoodCourt": [0.0, 0.0, 10.0, 0.0],
        "ShoppingMall": [0.0, 0.0, 10.0, 0.0],
        "Spa": [0.0, 0.0, 10.0, 0.0],
        "VRDeck": [0.0, 0.0, 10.0, 0.0],
        "Name": ["Anna Rossi", None, "Luigi Verdi", "Pia Neri"],
    })


class ExperimentRegressionTests(unittest.TestCase):
    def test_categorical_imputation_keeps_boolean_values(self):
        raw = passenger_frame()
        preprocessor = PreprocessorePasseggeri().fit(raw)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FutureWarning)
            out = preprocessor.transform(raw)
        downcast = [w for w in caught if issubclass(w.category, FutureWarning)
                    and "Downcasting" in str(w.message)]
        self.assertEqual(downcast, [])
        self.assertEqual(bool(out.loc[0, "CryoSleep"]), True)
        self.assertEqual(bool(out.loc[2, "CryoSleep"]), False)
        self.assertEqual(out.loc[1, "CryoSleep"], preprocessor.category_modes_["CryoSleep"])
        self.assertEqual(bool(out.loc[1, "VIP"]), False)
        self.assertEqual(out.loc[1, "HomePlanet"], "Earth")
        self.assertEqual(out.loc[1, "Destination"], preprocessor.category_modes_["Destination"])

    def test_fractional_cabin_median_fills_nullable_integer_column(self):
        raw = passenger_frame()
        raw["Cabin"] = ["G/432/S", "G/433/P", None, None]
        processor = PreprocessorePasseggeri().fit(raw)
        self.assertEqual(processor.cabin_median_, 432.5)
        transformed = processor.transform(raw)
        self.assertEqual(transformed["Cabin_number"].tolist(), [432.0, 433.0, 432.5, 432.5])
        self.assertTrue(pd.api.types.is_float_dtype(transformed["Cabin_number"]))

    def test_holdout_partition_is_reproducible(self):
        sample = pd.concat([passenger_frame()] * 10, ignore_index=True)
        sample["Transported"] = [0, 1] * 20
        a, ya = development_partition(sample)
        b, yb = development_partition(sample)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_series_equal(ya, yb)
        self.assertEqual(len(a), 32)

    def test_grouped_cv_disjoint_travel_groups(self):
        sample = pd.concat([passenger_frame()] * 10, ignore_index=True)
        sample["PassengerId"] = [f"{i // 2:04d}_{i % 2 + 1:02d}" for i in range(40)]
        groups = sample["PassengerId"].str.split("_").str[0]
        y = pd.Series([0, 1] * 20)
        for t, v in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(sample, y, groups):
            self.assertFalse(set(groups.iloc[t]) & set(groups.iloc[v]))

    def test_summary_reports_both_schemes(self):
        rows = pd.DataFrame([
            {"scheme": scheme, "model": model, "accuracy": .8, "f1": .7, "shared_groups": 0}
            for scheme in ("stratified", "stratified_group")
            for model in ("svc", "xgboost")
        ])
        output = summarize_results(rows)
        self.assertEqual(len(output), 4)
        self.assertEqual(set(output["model"]), {"svc", "xgboost"})


if __name__ == "__main__":
    unittest.main()
