# Dry Bean Classifier Bench — ML Assignment 2

**Name:** _<your full name>_ · **BITS ID:** 2025AC05211
**Course:** Machine Learning (AIMLCZG565) · M.Tech AIML

---

## a. Problem statement

Seven registered varieties of dry bean — *Barbunya, Bombay, Cali, Dermason, Horoz,
Seker* and *Sira* — are visually similar once they leave the field, so grading them by
hand is slow and inconsistent. Each variety commands a different market price, and a
mislabelled sack is a direct financial loss to the trader.

This project treats varietal grading as a **supervised multi-class classification**
problem: given 16 shape and size measurements extracted from a single high-resolution
photograph of a bean, predict which of the seven varieties it is.

Five classical classifiers are trained on the same data and compared on six metrics, and
the whole comparison is exposed through a Streamlit web app so that a non-programmer can
upload a test CSV, choose a model, and read the results.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Dry Bean Dataset |
| Source | UCI Machine Learning Repository, dataset ID 602 — <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset> |
| Instances | 13,611 (requirement: ≥ 500) |
| Features | 16 numeric (requirement: ≥ 12) |
| Target | `Class` — 7 bean varieties |
| Missing values | None |
| Task type | Multi-class classification |

**How the data was produced.** 13,611 individual beans were photographed with a
high-resolution camera; a computer-vision pipeline segmented each bean and computed
12 dimensional and 4 shape-form features from the segmented region.

**Features**

| # | Feature | Meaning |
|---|---|---|
| 1 | `Area` | Pixel count inside the bean boundary |
| 2 | `Perimeter` | Length of the bean boundary |
| 3 | `MajorAxisLength` | Longest line that can be drawn through the bean |
| 4 | `MinorAxisLength` | Longest line perpendicular to the major axis |
| 5 | `AspectRation` | Major axis ÷ minor axis |
| 6 | `Eccentricity` | Eccentricity of the ellipse with the same moments |
| 7 | `ConvexArea` | Pixel count of the smallest convex hull containing the bean |
| 8 | `EquivDiameter` | Diameter of a circle of the same area |
| 9 | `Extent` | Bean area ÷ bounding-box area |
| 10 | `Solidity` | Bean area ÷ convex-hull area |
| 11 | `roundness` | 4π·Area ÷ Perimeter² |
| 12 | `Compactness` | EquivDiameter ÷ MajorAxisLength |
| 13–16 | `ShapeFactor1…4` | Four derived shape descriptors |

**Class balance (full dataset)**

| Class | Count | Share |
|---|---:|---:|
| DERMASON | 3,546 | 26.0 % |
| SIRA | 2,636 | 19.4 % |
| SEKER | 2,027 | 14.9 % |
| HOROZ | 1,928 | 14.2 % |
| CALI | 1,630 | 12.0 % |
| BARBUNYA | 1,322 | 9.7 % |
| BOMBAY | 522 | 3.8 % |

The classes are imbalanced by roughly 7:1, which is why **every metric below is
macro-averaged** (each variety counted equally) rather than micro-averaged, and why MCC
is used as the headline score — it stays honest under class imbalance where raw accuracy
does not.

**Preprocessing and protocol**

* Stratified 80 / 20 train–test split (`random_state = 2025`) → 10,888 train / 2,723 test.
* `StandardScaler` fitted **inside** each pipeline on the training fold only, so no test
  statistics leak into training.
* Target encoded with `LabelEncoder`; the fitted encoder is saved alongside the models.
* The 2,723-row test split is committed as `test_data.csv` and is the file the Streamlit
  app consumes.

## c. GitHub repository link

> **https://github.com/&lt;your-github-username&gt;/dry-bean-classifier-bench**
>
> _Replace this line with your actual public repository URL before submitting._

Repository contents:

```
dry-bean-classifier-bench/
├── app.py                     Streamlit application
├── requirements.txt           pinned dependencies
├── README.md                  this file
├── test_data.csv              2,723-row held-out test split (with labels)
├── data/
│   └── dry_bean.csv           full 13,611-row dataset converted from the UCI .arff
└── model/
    ├── train_models.py        training + evaluation script (regenerates everything)
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── label_encoder.joblib
    └── metrics.json           metrics + metadata consumed by the app
```

## d. Models used

All five models are scikit-learn implementations wrapped in a `Pipeline` with
`StandardScaler`, trained on the identical 10,888-row training split and scored on the
identical 2,723-row test split.

| # | Model | Key hyperparameters |
|---|---|---|
| 1 | Logistic Regression | multinomial softmax, `C=1.0`, `max_iter=3000`, lbfgs |
| 2 | Decision Tree | `criterion='entropy'`, `max_depth=12`, `min_samples_leaf=8` |
| 3 | kNN | `n_neighbors=11`, `weights='distance'`, Euclidean |
| 4 | Naive Bayes | `GaussianNB`, `var_smoothing=1e-9` |
| 5 | Random Forest (Ensemble) | 300 trees, unrestricted depth, `min_samples_leaf=1` |

### Comparison table — test set (2,723 samples), macro-averaged

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9247 | **0.9953** | 0.9366 | 0.9330 | 0.9347 | 0.9090 |
| Decision Tree | 0.9060 | 0.9758 | 0.9204 | 0.9159 | 0.9179 | 0.8863 |
| kNN | **0.9262** | 0.9905 | **0.9396** | **0.9347** | **0.9369** | **0.9108** |
| Naive Bayes | 0.9053 | 0.9931 | 0.9155 | 0.9153 | 0.9149 | 0.8858 |
| Random Forest (Ensemble) | 0.9258 | 0.9951 | 0.9380 | 0.9338 | 0.9358 | 0.9103 |

AUC is computed One-vs-Rest and macro-averaged over the seven classes.
Bold marks the best value in each column.

### Per-class F1 (test set)

| Model | BARBUNYA | BOMBAY | CALI | DERMASON | HOROZ | SEKER | SIRA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.914 | 1.000 | 0.930 | 0.926 | 0.961 | 0.936 | 0.876 |
| Decision Tree | 0.885 | 0.995 | 0.910 | 0.904 | 0.951 | 0.933 | 0.848 |
| kNN | 0.919 | 1.000 | 0.930 | 0.920 | 0.966 | 0.943 | 0.881 |
| Naive Bayes | 0.861 | 0.990 | 0.899 | 0.903 | 0.960 | 0.935 | 0.857 |
| Random Forest | 0.918 | 0.995 | 0.932 | 0.922 | 0.961 | 0.944 | 0.878 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Surprisingly strong for a linear model — 0.9247 accuracy and the **highest AUC of all five (0.9953)**. The bean classes are close to linearly separable in the standardised 16-D feature space, so a softmax boundary is already almost enough. Its top AUC means its *ranking* of class probabilities is the best in the set, even though its hard-label accuracy is marginally below kNN. Trains in 0.3 s and the model file is 2 KB, so it is by far the cheapest model to serve. Weak spot: SIRA (F1 0.876), where the linear boundary cannot bend around the DERMASON overlap. |
| **Decision Tree** | The weakest model overall (accuracy 0.9060, MCC 0.8863) and clearly the weakest on AUC (0.9758). A single tree emits coarse, piecewise-constant probabilities, which is exactly what AUC punishes. Depth was capped at 12 with `min_samples_leaf=8`; letting it grow unrestricted drove training accuracy to 1.0000 while test accuracy *fell* from 0.9060 to 0.9008 — textbook overfitting, and the reason for the cap. Its value here is interpretability, not accuracy: the tree splits first on `MajorAxisLength`, then on `MinorAxisLength` and `ShapeFactor1`, confirming that raw bean size carries most of the signal. |
| **kNN** | **Best model on five of the six metrics** (accuracy 0.9262, precision 0.9396, recall 0.9347, F1 0.9369, MCC 0.9108). The features are continuous, standardised, and geometrically meaningful, so Euclidean distance is a genuinely good similarity measure here — beans of the same variety really do cluster. `weights='distance'` with k=11 smooths the imbalanced regions. Cost: training is instant but *inference* is the slowest of the five and the pickle is 1.35 MB because the whole training set must be stored. |
| **Naive Bayes** | Lowest accuracy (0.9053) and lowest MCC (0.8858), which is the expected penalty for its conditional-independence assumption — features like `Area`, `ConvexArea`, `EquivDiameter` and `Perimeter` are all near-perfectly correlated measures of size, so the model effectively counts the same evidence four times and becomes overconfident. Interestingly its **AUC is still 0.9931**, third best: the class ordering it produces is good even where its hard decisions are wrong. Fastest to train (0.01 s) and a 3 KB model — a reasonable baseline, not a serious contender. |
| **Random Forest (Ensemble)** | Effectively tied with kNN at the top (accuracy 0.9258, MCC 0.9103, AUC 0.9951) and a **+2.0 point accuracy gain over the single Decision Tree it is built from**, which is the clearest demonstration of variance reduction through bagging in this experiment. It is the most balanced model: no class falls below F1 0.878, and it is the only model that is top-3 on every single metric. Costs 300× the training time of one tree (2.0 s) and a 7.6 MB artefact, but inference stays fast. |
| **Overall winner for this dataset** | **Random Forest**, with kNN a very close second. kNN edges it by 0.0004 on accuracy and 0.0005 on MCC — differences of roughly one test sample out of 2,723, i.e. statistical noise. Random Forest is chosen as the winner because it matches that accuracy while also delivering a near-top AUC (0.9951 vs kNN's 0.9905), giving better-calibrated probabilities; it needs no training data at inference time; it degrades gracefully on unseen feature ranges; and it supplies feature importances. If deployment memory were the binding constraint, Logistic Regression (2 KB, 0.9953 AUC) would be the pragmatic pick with only a 0.11 point accuracy sacrifice. |

**Where every model loses points — the SIRA / DERMASON boundary.** All five classifiers
share the same dominant error, visible in the confusion matrix: 48 SIRA beans are called
DERMASON and 35 DERMASON beans are called SIRA. The two varieties genuinely overlap in
size and roundness, so this is a limit of the 16 features rather than a modelling
failure — no amount of tuning moved it. The secondary confusion is BARBUNYA ↔ CALI
(15 and 13 samples). BOMBAY, by contrast, is classified essentially perfectly by every
model (F1 0.990–1.000) because it is roughly twice the area of any other variety and
therefore trivially separable.

## Streamlit application

**Live app:** _<paste your Streamlit Community Cloud URL here>_

Features implemented:

1. **Dataset upload (CSV)** — sidebar uploader that accepts any CSV carrying the 16
   feature columns; a bundled-test-data toggle is provided as a fallback. If the CSV has
   no `Class` column the app degrades gracefully to prediction-only mode.
2. **Model selection dropdown** — switch between all five trained classifiers; every
   panel recomputes live.
3. **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1 and MCC shown as
   metric tiles for the selected model, computed on the uploaded data.
4. **Confusion matrix and classification report** — annotated heatmap (raw counts or
   row-normalised) plus the full per-class precision/recall/F1/support table, with
   One-vs-Rest ROC curves per class.
5. **Extras** — a *Model comparison* tab that scores all five models on the uploaded file
   and highlights the winner, a *Predictions* tab with per-row confidence and a
   misclassified-rows-only filter, CSV download of predictions and of the comparison
   table, and a *Dataset* tab with class balance and feature statistics.

## Reproducing the results

```bash
pip install -r requirements.txt
python model/train_models.py     # retrains all five models, rewrites test_data.csv + metrics.json
streamlit run app.py             # launches the UI at http://localhost:8501
```

`random_state=2025` is fixed throughout, so the numbers in the tables above reproduce
exactly.
