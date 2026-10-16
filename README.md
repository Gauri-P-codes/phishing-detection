# Phishing Website Detection using AI & ML
**Course:** 23PCCE501 AIML / 23PCCE501L AIML Lab - Mini-Project (Group C-1: ML case study)

Classifies a URL as *phishing* or *legitimate* from lexical features (length, IP host, brand-in-subdomain,
suspicious keywords, entropy, TLD, ...). The URL is analysed as text only - the site is never visited.

## Quick start
```bash
pip install -r requirements.txt
python -m src.train            # full pipeline (~1-2 min); use --quick for a fast run
streamlit run app.py           # web app
mlflow ui                      # (optional) compare experiment runs
```
Real data: put a CSV with columns `url,label` (1=phishing, 0=legit) at `data/urls.csv`
(PhiUSIIL Phishing URL dataset on UCI, or Kaggle "Phishing Site URLs"). The pipeline picks it up automatically.

> **Important:** the bundled generator creates *synthetic* URLs (+3% label noise) so the code runs offline.
> Scores on it (~96% F1) are NOT real-world numbers. Report results on a real dataset in your submission.

## Syllabus mapping
| Syllabus unit | Where in the project |
|---|---|
| **I** Problem solving, state-space | Problem formulated as a search/classification task; feature selection = state-space search |
| **II** Heuristic search (Hill Climbing, Simulated Annealing, Beam) | `src/feature_search.py` - feature subset selection; state = bit-mask, move = flip a bit, h(n) = CV-F1 - penalty |
| **III** Data pre-processing | `src/preprocess.py` (cleaning, dedupe, scaling, discretisation, PCA, stratified split), `src/features.py` (transformation) |
| **IV** Supervised learning | Logistic Regression, k-NN (effect of k), Decision Tree, Naive Bayes in `src/train.py`; confusion matrix, accuracy, precision, recall, specificity, F1, ROC-AUC, MSE/RMSE, 5-fold CV |
| **V** Unsupervised | `src/unsupervised.py` - K-means (elbow, silhouette, purity), DBSCAN outliers, **Apriori from scratch** + association rules (`if kw_in_url, bad_tld -> PHISHING`) |
| **VI** Optimisation | `src/optimizers.py` - logistic regression from scratch with GD, SGD, Momentum, **RMSprop, Adam**; MLP with Adam; GridSearchCV hyper-parameter tuning |

## Self-study topics used
| Topic | Use here |
|---|---|
| Search Space Management | Hill-climbing's local-maximum problem vs **Beam, Tabu, Simulated Annealing** (`feature_search.py`) |
| Data Structure Management | **Trie** for O(L) keyword/brand lookup (`trie.py`); NumPy matrices/vectors; hash-map caching of search states |
| Model Deployment | **Streamlit** app (`app.py`) - Gradio/FastAPI are drop-in alternatives |
| Experiment Tracking | **MLflow** logging of every model's params/metrics (`log_mlflow` in `train.py`) |
| Version Control | Git/GitHub - suggested workflow below |
| MLOps | `Dockerfile` (containerised app); add **DVC** for dataset versioning (`dvc add data/urls.csv`) |
| AutoML | Optional extension: run `auto-sklearn`/`AutoGluon` on `features` and compare with your best model |
| Computer Vision | Optional extension: screenshot-based phishing detection with OpenCV / YOLO (logo spoofing) |

## Lab assignment coverage
A4 regression/prediction -> LogReg/optimisers - A5 k-NN with varying k -> `reports/knn_k.png` -
B1 Apriori -> `association_rules.csv` - B2 K-means with varying K -> `kmeans.csv`, `clusters.png` -
A3 pre-processing/sampling/normalisation -> `preprocess.py` - Group A/B heuristics -> `feature_search.csv`.

## Outputs (`reports/`)
`model_comparison.csv`, `confusion_matrix.png`, `knn_k.png`, `feature_importance.png`, `feature_search.csv`,
`optimizers.png/.csv`, `kmeans.csv`, `clusters.png`, `association_rules.csv`, `summary.json`.

## Git workflow (self-study)
```bash
git init && git add . && git commit -m "Initial phishing detection pipeline"
git checkout -b feature/real-dataset      # one branch per experiment
git push -u origin main                   # then open a Pull Request on GitHub
```

## Mini-project report outline
1. Problem domain & motivation  2. Literature + dataset survey  3. Preprocessing & feature engineering
4. Model study & experiments  5. Results (tables/plots from `reports/`)  6. Deployment & MLOps  7. Limitations & future work

## Limitations (mention in viva)
Lexical features miss content-based attacks and can be evaded; attackers adapt (concept drift) so models need retraining;
public datasets can have domain/time bias; false negatives are costlier than false positives, so prioritise **recall**.
