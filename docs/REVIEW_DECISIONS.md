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
Implemented on branch review/val-01-validation (awaiting test execution): a new FoldSafePreprocessor fits group counts, surname frequencies, categorical fallback values, and numerical medians exclusively on the training subset passed to fit(). It computes age bins after imputing age and applies trained statistics unchanged to validation/test rows. The model pipeline begins with this transformer, followed by OHE/scaling and the classifier, so RandomizedSearchCV refits the whole preprocessing within each training fold.
In main.py, the raw labeled train is divided into stratified development (80%) and final holdout (20%) before any learned preprocessing. Candidate models are tuned on development CV and selected by accuracy (Kaggle evaluation metric); the chosen model is evaluated on the untouched holdout, logged separately as outputs/holdout_results.csv, then refitted on all labeled training data for the Kaggle submission. F1/precision/recall remain supplementary metrics.
Scope/limitations: the new validation path does not reuse the legacy transductive train+Kaggle-test frequency counts; unseen groups are treated conservatively as solitary. Group peers may still occur in different random folds, so group-aware split evaluation remains a possible follow-up sensitivity analysis. CV-best accuracy is a selection statistic, not an unbiased evaluation; holdout is the independent estimate. Legacy run_preprocessing remains for compatibility with FE-01/FE-02 unit tests, not used by the new main training path. This refactor can change model accuracy in either direction; no improvement is claimed.
Status: user executed python -m unittest discover -s tests -v on 16 October 2026; all 13 tests passed in 0.716 s, with zero failures/errors (4 FE-01, 4 FE-02, 4 VAL-01, 1 sample test). This establishes regression-test success only; full Kaggle dataset run, independent holdout metrics and submission are not yet verified.

Release policy: main stays unchanged until approval. Never claim successful tests or Accuracy improvements without execution and evidence.
## EXP-01 — First complete Kaggle-dataset run (16 October 2026)
User supplied terminal output and generated CSV files. Complete pipeline exited successfully and created 4,277 prediction rows with no duplicate PassengerIds or nulls; Transported is boolean (2,210 True, 2,067 False). Model selection by development CV accuracy chose XGBoost (0.809175), with SVC very close (0.808024). Independent holdout accuracy for selected XGBoost: 0.809086; precision: 0.805618; recall: 0.818493; F1: 0.812005. These are NOT Kaggle leaderboard scores. The old pre-review result is not a controlled apples-to-apples comparison because evaluation design and preprocessing differ. The log contains 240 FutureWarning repetitions from src/fold_preprocessing.py line 109 (Pandas silent downcasting during fillna); a compatibility cleanup should be reviewed and tested before final release. Next actions: resolve warnings, inspect fairness of SVC comparison, run further stability checks; do not select or tune models against the held-out evaluation results.
