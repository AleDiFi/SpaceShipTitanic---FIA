"""FE-02 regression tests: groups and surname-frequency features."""
import unittest

import numpy as np
import pandas as pd

from src.feature_engineering import FeatureEngineeringPipeline, GroupTransformer, FamilySizeTransformer
from src.preprocessing import PreprocessConfig, impute_surname, run_preprocessing


class GroupFamilyTests(unittest.TestCase):
    def test_group_split_across_train_and_test(self):
        train = pd.DataFrame({"PassengerId": ["0042_01", "0042_02", "0050_01"],
                              "Name": ["Ana Rossi", "Lia Rossi", "Mia Bianchi"],
                              "Transported": [True, False, True]})
        test = pd.DataFrame({"PassengerId": ["0042_03", "0051_01"],
                             "Name": ["Leo Rossi", "Noa Verdi"]})
        pipeline = FeatureEngineeringPipeline([GroupTransformer(), FamilySizeTransformer()])
        train_fe, test_fe = pipeline.fit_transform(train, test)
        self.assertEqual(train_fe.loc[0, "Group_size"], 3)
        self.assertEqual(test_fe.loc[0, "Group_size"], 3)
        self.assertEqual(test_fe.loc[0, "Solo"], 0)
        self.assertEqual(test_fe.loc[1, "Solo"], 1)
        self.assertEqual(train_fe.loc[0, "Family_size"], 3)
        self.assertEqual(test_fe.loc[0, "Family_size"], 3)
        self.assertEqual(train_fe["Transported"].tolist(), [True, False, True])
        self.assertNotIn("Transported", test_fe.columns)

    def test_missing_names_never_become_unknown_family(self):
        train = pd.DataFrame({"PassengerId": ["0001_01", "0002_01"],
                              "Name": [np.nan, np.nan]})
        test = pd.DataFrame({"PassengerId": ["0003_01"], "Name": [np.nan]})
        train_fe, test_fe = FeatureEngineeringPipeline([GroupTransformer(), FamilySizeTransformer()]).fit_transform(train, test)
        self.assertTrue(train_fe["Surname"].isna().all())
        self.assertTrue(test_fe["Family_size"].isna().all())

    def test_impute_only_unambiguous_group_surnames(self):
        df = pd.DataFrame({"Group": [1, 1, 2, 2, 2],
                           "Surname": ["Rossi", np.nan, "Verdi", "Bianchi", np.nan]})
        out = impute_surname(df, PreprocessConfig())
        self.assertEqual(out.loc[1, "Surname"], "Rossi")
        self.assertTrue(pd.isna(out.loc[4, "Surname"]))

    def test_family_size_recomputed_after_surname_inference(self):
        train = pd.DataFrame({"Group": [1, 1, 2], "Surname": ["Rossi", np.nan, "Rossi"],
                              "Family_size": [2, np.nan, 2], "Transported": [True, False, True]})
        test = pd.DataFrame({"Group": [3], "Surname": ["Verdi"], "Family_size": [1]})
        train_out, test_out, _ = run_preprocessing(train, test)
        self.assertEqual(train_out["Family_size"].tolist(), [3, 3, 3])
        self.assertEqual(test_out.loc[0, "Family_size"], 1)


if __name__ == "__main__":
    unittest.main()
