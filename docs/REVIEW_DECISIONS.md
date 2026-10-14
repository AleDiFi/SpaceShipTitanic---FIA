# FIA — Technical decision log

## FE-01 — Spending
Problem: partial missing expenses could be misclassified as zero total spending.
Decision: total expenditure remains missing until every component is known; No_spending is true only when all observed components equal zero; do not fill missing expenses using that flag.
Keep CryoSleep zero-spending imputation as an explicit domain assumption.
Status: 4/4 FE-01 regression tests passed in the user's local run on 14 October 2026.

## FE-02 — Group and surname counts
Problem: group sizes and surname frequencies were computed separately in train and test; unknown-name placeholders distorted counts; post-imputation family counts were stale.
Decision: compute count-based features over the combined unlabeled train/test passenger table, preserve row order and target separation; preserve missing surnames; infer missing surnames only for travel groups with exactly one distinct known surname; recompute Family_size after imputation.
Terminology: Family_size is an observed surname-frequency proxy, not the number of verified relatives.
Status: 4/4 FE-02 regression tests passed after correcting the index and missing-name issues. User's local test output on 14 October 2026: 9 tests total, 0 errors/failures (including FE-01 and the existing sample test).
Tradeoff: use of the unlabeled Kaggle test set is transductive; during VAL-01 cross-validation must avoid using validation folds improperly.

## VAL-01 — Validation architecture
Planned: investigate learned imputation before CV and independent model assessment. No changes approved yet.

Release policy: main stays unchanged until approval. Never claim successful tests or Accuracy improvements without execution and evidence.