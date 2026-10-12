"""Regression tests for FE-01 spending and missing-value handling."""
import unittest

import numpy as np
import pandas as pd

from src.feature_engineering import ExpenditureTransformer
from src.preprocessing import PreprocessConfig, impute_spending


EXPENSES = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


class SpendingTests(unittest.TestCase):
    def test_missing_expense_is_not_zero_spending(self):
        df = pd.DataFrame([[0, np.nan, 0, np.nan, 0]], columns=EXPENSES)
        result = ExpenditureTransformer().transform(df)
        self.assertTrue(pd.isna(result.loc[0, "Expenditure"]))
        self.assertEqual(result.loc[0, "No_spending"], 0)

    def test_all_known_zero_expenses(self):
        df = pd.DataFrame([[0, 0, 0, 0, 0]], columns=EXPENSES)
        result = ExpenditureTransformer().transform(df)
        self.assertEqual(result.loc[0, "Expenditure"], 0)
        self.assertEqual(result.loc[0, "No_spending"], 1)

    def test_missing_expense_uses_observed_median_not_zero_flag(self):
        df = pd.DataFrame([
            [0, np.nan, 0, 0, 0],
            [0, 20, 0, 0, 0],
            [0, 40, 0, 0, 0],
        ], columns=EXPENSES)
        df = ExpenditureTransformer().transform(df)
        result = impute_spending(df, PreprocessConfig(apply_domain_rules=True))
        self.assertEqual(result.loc[0, "FoodCourt"], 30)
        self.assertEqual(result.loc[0, "Expenditure"], 30)
        self.assertEqual(result.loc[0, "No_spending"], 0)

    def test_cryosleep_missing_expense_is_zero(self):
        df = pd.DataFrame([[0, np.nan, 0, 0, 0]], columns=EXPENSES)
        df["CryoSleep"] = True
        result = impute_spending(ExpenditureTransformer().transform(df), PreprocessConfig())
        self.assertEqual(result.loc[0, "FoodCourt"], 0)
        self.assertEqual(result.loc[0, "No_spending"], 1)


if __name__ == "__main__":
    unittest.main()
