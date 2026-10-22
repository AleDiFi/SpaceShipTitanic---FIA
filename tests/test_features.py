"""Feature-engineering regression tests against the ONLY production preprocessor.

Historical FE-01 and FE-02 behavior is tested via PreprocessorePasseggeri,
not via obsolete feature-engineering and transductive preprocessing modules.
"""
import unittest

import numpy as np
import pandas as pd

from src.preprocessing import (PreprocessorePasseggeri, FeatureTransformer, FeatureEngineeringPipeline,
                               GroupTransformer, SurnameTransformer, CabinTransformer,
                               AgeGroupTransformer, SpendingTransformer)


EXPENSES = ("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")


def passengers(ids, names=None, cryosleep=None):
    n = len(ids)
    return pd.DataFrame({
        "PassengerId": ids,
        "HomePlanet": ["Earth"] * n,
        "CryoSleep": cryosleep if cryosleep is not None else [False] * n,
        "Cabin": [f"G/{i+1}/S" for i in range(n)],
        "Destination": ["TRAPPIST-1e"] * n,
        "Age": [25.] * n,
        "VIP": [False] * n,
        "RoomService": [0.] * n,
        "FoodCourt": [0.] * n,
        "ShoppingMall": [0.] * n,
        "Spa": [0.] * n,
        "VRDeck": [0.] * n,
        "Name": names if names is not None else [f"Person{i} Rossi" for i in range(n)],
    })


class StrategyPatternTests(unittest.TestCase):
    def test_feature_transformers_implement_strategy_interface(self):
        for strategy in (GroupTransformer(), SurnameTransformer(), CabinTransformer(),
                         AgeGroupTransformer(), SpendingTransformer()):
            self.assertIsInstance(strategy, FeatureTransformer)

    def test_pipeline_applies_strategies_in_order(self):
        frame = passengers(["0001_01"], names=["Anna Rossi"])
        out = FeatureEngineeringPipeline([
            GroupTransformer(), SurnameTransformer(), CabinTransformer()
        ]).transform(frame.copy())
        self.assertEqual(out.loc[0, "Group"], 1)
        self.assertEqual(out.loc[0, "Surname"], "Rossi")
        self.assertEqual(out.loc[0, "Cabin_deck"], "G")


class GroupFamilyTests(unittest.TestCase):
    def test_group_count_and_solo_within_fitted_cohort(self):
        raw = passengers(["0042_01", "0042_02", "0050_01"])
        result = PreprocessorePasseggeri().fit_transform(raw)
        self.assertEqual(result["Group_size"].tolist(), [2, 2, 1])
        self.assertEqual(result["Solo"].tolist(), [0, 0, 1])

    def test_unseen_group_does_not_learn_validation_companions(self):
        train = passengers(["0042_01"])
        valid = passengers(["0099_01", "0099_02"])
        result = PreprocessorePasseggeri().fit(train).transform(valid)
        self.assertEqual(result["Group_size"].tolist(), [1, 1])

    def test_missing_names_are_not_a_shared_family(self):
        train = passengers(["0001_01", "0002_01"], [None, None])
        result = PreprocessorePasseggeri().fit_transform(train)
        self.assertEqual(result["Family_size"].tolist(), [0, 0])

    def test_unambiguous_surname_inference_recomputes_family_frequency(self):
        raw = passengers(["0001_01", "0001_02", "0002_01"],
                         ["Ana Rossi", None, "Leo Rossi"])
        result = PreprocessorePasseggeri().fit_transform(raw)
        self.assertEqual(result["Surname"].tolist(), ["Rossi"] * 3)
        self.assertEqual(result["Family_size"].tolist(), [3, 3, 3])

    def test_ambiguous_group_does_not_infer_surname(self):
        raw = passengers(["0001_01", "0001_02", "0001_03"],
                         ["Ana Rossi", "Leo Verdi", None])
        result = PreprocessorePasseggeri().fit_transform(raw)
        self.assertEqual(result.loc[2, "Surname"], "Unknown")
        self.assertEqual(result.loc[2, "Family_size"], 0)


class SpendingTests(unittest.TestCase):
    def test_known_zero_expenses_identify_no_spending(self):
        raw = passengers(["0001_01"])
        result = PreprocessorePasseggeri().fit_transform(raw)
        self.assertEqual(result.loc[0, "Expenditure"], 0)
        self.assertEqual(result.loc[0, "No_spending"], 1)

    def test_missing_expense_uses_trained_median_not_zero(self):
        raw = passengers(["0001_01", "0002_01", "0003_01"])
        raw["FoodCourt"] = [np.nan, 20., 40.]
        result = PreprocessorePasseggeri().fit_transform(raw)
        self.assertEqual(result.loc[0, "FoodCourt"], 30.)
        self.assertEqual(result.loc[0, "Expenditure"], 30.)
        self.assertEqual(result.loc[0, "No_spending"], 0)

    def test_cryosleep_missing_expense_is_zero(self):
        raw = passengers(["0001_01", "0002_01"], cryosleep=[True, False])
        raw.loc[0, "FoodCourt"] = np.nan
        raw.loc[1, "FoodCourt"] = 50.
        result = PreprocessorePasseggeri().fit_transform(raw)
        self.assertEqual(result.loc[0, "FoodCourt"], 0)
        self.assertEqual(result.loc[0, "No_spending"], 1)


if __name__ == "__main__":
    unittest.main()
