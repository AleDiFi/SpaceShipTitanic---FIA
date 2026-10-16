"""Fold-safe feature engineering and missing-data imputation for Spaceship Titanic.

All learned distributions are fit ONLY on the samples passed to fit().
transform() never learns statistics from the validation or Kaggle test rows.
Group and surname frequencies describe the fitted cohort; unseen groups
receive a conservative size of one, not a peek at validation passengers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

EXPENSE_COLUMNS = ("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")


def _mode(series: pd.Series, default):
    values = series.dropna().mode()
    return values.iloc[0] if not values.empty else default


def _age_groups(age: pd.Series) -> pd.Series:
    result = pd.Series("Unknown", index=age.index, dtype=object)
    for label, condition in (
        ("Age_0-12", age <= 12),
        ("Age_13-17", (age > 12) & (age < 18)),
        ("Age_18-25", (age >= 18) & (age <= 25)),
        ("Age_26-30", (age > 25) & (age <= 30)),
        ("Age_31-50", (age > 30) & (age <= 50)),
        ("Age_51+", age > 50),
    ):
        result.loc[condition.fillna(False)] = label
    return result


class FoldSafePreprocessor(BaseEstimator, TransformerMixin):
    """Learn train-fold-only statistics, and apply them to unseen passengers."""

    def _extract(self, X):
        frame = X.copy().reset_index(drop=True)
        frame["Group"] = frame["PassengerId"].astype("string").str.split("_").str[0].astype(int)
        frame["Surname"] = frame["Name"].astype("string").str.split().str[-1].astype(object)
        cabin = frame["Cabin"].astype("string").str.split("/", n=2, expand=True).reindex(columns=[0, 1, 2])
        frame["Cabin_deck"] = cabin[0].astype(object)
        frame["Cabin_number"] = pd.to_numeric(cabin[1], errors="coerce")
        frame["Cabin_side"] = cabin[2].astype(object)
        for n in range(1, 8):
            lower = (n - 1) * 300
            if n == 7:
                region = frame["Cabin_number"] > 1800
            elif n == 1:
                region = frame["Cabin_number"] <= 300
            else:
                region = (frame["Cabin_number"] > lower) & (frame["Cabin_number"] <= n * 300)
            frame[f"Cabin_region{n}"] = region.fillna(False).astype(int)
        frame = frame.drop(columns=["PassengerId", "Name", "Cabin"])
        return frame

    def fit(self, X, y=None):
        source = self._extract(X)
        self.group_counts_ = source["Group"].value_counts().to_dict()
        known = source.dropna(subset=["Surname"])
        unique = known.groupby("Group")["Surname"].nunique()
        compatible = known[known["Group"].isin(unique[unique == 1].index)]
        self.group_surname_ = compatible.groupby("Group")["Surname"].first().to_dict()
        source["Surname"] = source["Surname"].fillna(source["Group"].map(self.group_surname_))
        self.surname_counts_ = source["Surname"].value_counts().to_dict()
        known_planets = source.dropna(subset=["HomePlanet"])
        self.group_planet_ = known_planets.groupby("Group")["HomePlanet"].agg(
            lambda s: _mode(s, "Earth")
        ).to_dict()
        self.category_modes_ = {
            col: _mode(source[col], default)
            for col, default in (
                ("HomePlanet", "Earth"), ("Destination", "TRAPPIST-1e"),
                ("CryoSleep", False), ("VIP", False),
                ("Cabin_deck", "Unknown"), ("Cabin_side", "Unknown"),
            )
        }
        self.age_median_ = float(source["Age"].median()) if source["Age"].notna().any() else 28.0
        source = self._fill_basic(source)
        self.spending_medians_ = {}
        self.spending_by_group_ = {}
        for col in EXPENSE_COLUMNS:
            self.spending_medians_[col] = float(source[col].median()) if source[col].notna().any() else 0.0
            grouping = source.groupby(["Age_group", "HomePlanet", "VIP"])[col].median()
            self.spending_by_group_[col] = grouping.to_dict()
        self.cabin_median_ = (
            float(source["Cabin_number"].median())
            if source["Cabin_number"].notna().any() else 0.0
        )
        return self

    def _fill_basic(self, frame):
        frame = frame.copy()
        frame["Surname"] = frame["Surname"].fillna(frame["Group"].map(self.group_surname_))
        frame["Surname"] = frame["Surname"].fillna("Unknown").astype(object)
        frame["Group_size"] = frame["Group"].map(self.group_counts_).fillna(1).astype(int)
        frame["Solo"] = (frame["Group_size"] == 1).astype(int)
        frame["Family_size"] = frame["Surname"].map(self.surname_counts_).fillna(0).astype(int)

        planet = frame["HomePlanet"].fillna(frame["Group"].map(self.group_planet_))
        planet = planet.mask(planet.isna() & frame["Cabin_deck"].isin(["A", "B", "C", "T"]), "Europa")
        planet = planet.mask(planet.isna() & frame["Cabin_deck"].eq("G"), "Earth")
        frame["HomePlanet"] = planet.fillna(self.category_modes_["HomePlanet"]).astype(object)

        for col in ("Destination", "CryoSleep", "VIP", "Cabin_deck", "Cabin_side"):
            # Resolve missing categories explicitly: pandas fillna on an object
            # series can silently downcast booleans and emit FutureWarning.
            fallback = self.category_modes_[col]
            frame[col] = frame[col].map(
                lambda value: value if pd.notna(value) else fallback
            ).astype(object)
        frame["Age"] = frame["Age"].fillna(self.age_median_)
        frame["Age_group"] = _age_groups(frame["Age"])
        return frame

    def transform(self, X):
        check_is_fitted(self, ["group_counts_", "spending_medians_", "cabin_median_"])
        frame = self._fill_basic(self._extract(X))
        for col in EXPENSE_COLUMNS:
            # A passenger in cryosleep cannot incur onboard service charges:
            # an explicit domain assumption, not an inferred validation statistic.
            frame.loc[frame["CryoSleep"].eq(True) & frame[col].isna(), col] = 0
            grouped = self.spending_by_group_[col]
            keys = list(zip(frame["Age_group"], frame["HomePlanet"], frame["VIP"]))
            estimates = pd.Series([grouped.get(key, np.nan) for key in keys], index=frame.index)
            frame[col] = frame[col].fillna(estimates).fillna(self.spending_medians_[col])
        frame["Expenditure"] = frame[list(EXPENSE_COLUMNS)].sum(axis=1)
        frame["No_spending"] = frame[list(EXPENSE_COLUMNS)].eq(0).all(axis=1).astype(int)
        frame["Cabin_number"] = frame["Cabin_number"].fillna(self.cabin_median_)
        return frame
