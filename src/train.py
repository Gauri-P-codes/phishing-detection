"""End-to-end pipeline:  python -m src.train [--data data/urls.csv] [--quick]
Outputs: models/best_model.joblib, reports/*.png, reports/*.csv, reports/summary.json
Speed note: slow steps (CV, k sweep, feature search, clustering) use a SAMPLE_N-row subsample;
final models are still trained on ALL training rows."""
import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, ConfusionMatrixDisplay, roc_auc_score)
from . import data_gen, features, preprocess as pp
from .feature_search import FeatureSearch
from .optimizers import LogReg
from .unsupervised import kmeans_study, dbscan_outliers, apriori, rules, to_transactions

R = "reports"
SAMPLE_N = 20000          # rows used for slow steps; lower it if training is still slow
SELECT_BY = "cv_f1"       # change to "cv_recall" to pick the final model by recall

def say(msg): print(msg, flush=True)

def metrics(y, pred, proba):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred), "recall(sensitivity)": recall_score(y, pred),
            "specificity": tn / (tn + fp), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, proba),
            "MSE": mean_squared_error(y, proba), "RMSE": mean_squared_error(y, proba) ** .5, "TP": tp, "FP": fp, "TN": tn, "FN": fn}

def log_mlflow(name, params, mets):
    try:
        import mlflow
        mlflow.set_experiment("phishing-detection")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params); mlflow.log_metrics({k: float(v) for k, v in mets.items()})
    except Exception:      # MLflow is optional (self-study: Experiment Tracking)
        pass

def main(a):
    os.makedirs(R, exist_ok=True); os.makedirs("models", exist_ok=True)
    # ---------- 1. data + preprocessing (Unit III) ----------
    if not os.path.exists(a.data):
        say("[data] no dataset found -> generating synthetic data"); os.makedirs("data", exist_ok=True)
        data_gen.generate(3000 if a.quick else 8000).to_csv(a.data, index=False)
    df = pp.clean(pd.read_csv(a.data)); y = df["label"].astype(int).values
    say("[features] extracting ... (can take a minute on large data)")
    F = features.featurize(df["url"]); names = list(F.columns)
    say(f"[features] {F.shape}, phishing ratio = {y.mean():.2f}")
    F.assign(label=y).describe().T.to_csv(f"{R}/feature_summary.csv")
    Xtr, Xte, ytr, yte = pp.split(F.values, y)
    pca = pp.pca_reduce(StandardScaler().fit_transform(Xtr))
    S = min(SAMPLE_N, len(Xtr)); Xs, ys = Xtr[:S], ytr[:S]          # data is already shuffled -> fair sample

    # ---------- 2. supervised models + CV (Unit IV) ----------
    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "kNN(k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=0),
        "NaiveBayes": make_pipeline(StandardScaler(), GaussianNB()),
        "MLP(Adam)": make_pipeline(StandardScaler(), MLPClassifier((32, 16), solver="adam", max_iter=100,
                                                                   early_stopping=True, random_state=0)),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=0); rows = []; fitted = {}
    for n, m in models.items():
        s = cross_validate(m, Xs, ys, cv=cv, scoring=["accuracy", "f1", "recall"])   # CV on the sample
        m.fit(Xtr, ytr); fitted[n] = m                                               # final fit on ALL train rows
        pr = m.predict_proba(Xte)[:, 1]; me = metrics(yte, m.predict(Xte), pr)
        rows.append({"model": n, "cv_acc": s["test_accuracy"].mean(), "cv_f1": s["test_f1"].mean(),
                     "cv_recall": s["test_recall"].mean(), **me})
        log_mlflow(n, {"model": n}, me)
        say(f"  {n:20s} CV-F1={s['test_f1'].mean():.3f}  test-F1={me['f1']:.3f}  recall={me['recall(sensitivity)']:.3f}")
    res = pd.DataFrame(rows).sort_values(SELECT_BY, ascending=False); res.to_csv(f"{R}/model_comparison.csv", index=False)

    # k-NN: effect of k (lab assignment A5) - on a subsample
    say("[knn] sweeping k ...")
    ks = list(range(1, 26, 2)); accs = []
    for k in ks:
        accs.append(make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(Xs, ys).score(Xte[:5000], yte[:5000]))
    plt.figure(figsize=(5, 3.5)); plt.plot(ks, accs, "o-"); plt.xlabel("k"); plt.ylabel("test accuracy"); plt.title("kNN: effect of k")
    plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{R}/knn_k.png", dpi=130); plt.close()

    # hyper-parameter tuning
    gs = GridSearchCV(DecisionTreeClassifier(random_state=0), {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 5, 20]},
                      cv=3, scoring="f1").fit(Xs, ys)
    say(f"[tuning] DecisionTree best: {gs.best_params_}  F1={gs.best_score_:.3f}")
    fitted["DecisionTree(tuned)"] = gs.best_estimator_

    # ---------- 3. best model: confusion matrix + save ----------
    best_name = res.iloc[0]["model"]; best = fitted[best_name]
    ConfusionMatrixDisplay.from_estimator(best, Xte[:5000], yte[:5000], display_labels=["Legit", "Phishing"], cmap="Blues")
    plt.title(f"Confusion matrix - {best_name}"); plt.tight_layout(); plt.savefig(f"{R}/confusion_matrix.png", dpi=130); plt.close()
    joblib.dump({"model": best, "features": names, "name": best_name}, "models/best_model.joblib")
    imp = pd.Series(fitted["DecisionTree"].feature_importances_, names).sort_values()[-12:]
    imp.plot.barh(figsize=(6, 4), title="Decision-tree feature importance"); plt.tight_layout()
    plt.savefig(f"{R}/feature_importance.png", dpi=130); plt.close()

    # ---------- 4. heuristic feature-subset search (Unit II / self-study) ----------
    srows = []
    if not a.quick:
        say("[search] feature-subset search (4 algorithms) ...")
        fs = FeatureSearch(Xs, ys)
        for nm, fn in [("HillClimbing", fs.hill_climbing), ("SimulatedAnnealing", fs.simulated_annealing),
                       ("BeamSearch(w=3)", fs.beam_search), ("TabuSearch", fs.tabu_search)]:
            t0 = time.time(); mask, sc = fn()
            srows.append({"algorithm": nm, "score": round(sc, 4), "n_features": sum(mask), "evals_so_far": fs.evals,
                          "sec": round(time.time() - t0, 1), "selected": ", ".join(n for n, m in zip(names, mask) if m)})
            say(f"  {nm}: score={sc:.4f}, {sum(mask)} features")
        pd.DataFrame(srows).to_csv(f"{R}/feature_search.csv", index=False)

    # ---------- 5. optimisers from scratch (Unit VI) ----------
    say("[optimizers] GD / SGD / Momentum / RMSprop / Adam ...")
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte); orows = []
    plt.figure(figsize=(5.5, 3.8))
    for opt, lr in [("gd", .5), ("sgd", .05), ("momentum", .01), ("rmsprop", .01), ("adam", .01)]:
        lr_ = LogReg(opt, lr=lr, epochs=15, batch=256).fit(A, ytr); plt.plot(lr_.history, label=opt)
        orows.append({"optimizer": opt, "lr": lr, "final_loss": lr_.history[-1], "test_acc": accuracy_score(yte, lr_.predict(B)),
                      "test_f1": f1_score(yte, lr_.predict(B))})
    plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.yscale("log"); plt.legend(); plt.title("Optimiser convergence"); plt.tight_layout()
    plt.savefig(f"{R}/optimizers.png", dpi=130); plt.close(); pd.DataFrame(orows).to_csv(f"{R}/optimizers.csv", index=False)

    # ---------- 6. unsupervised (Unit V) - on a subsample ----------
    say("[unsupervised] K-means, DBSCAN, Apriori ...")
    N = min(SAMPLE_N, len(F)); Fu, yu = F.iloc[:N], y[:N]
    Xall = StandardScaler().fit_transform(Fu.values)
    kres, km, purity = kmeans_study(Xall, yu); kres.to_csv(f"{R}/kmeans.csv", index=False)
    lab, out_frac = dbscan_outliers(Xall)
    T = to_transactions(Fu, yu); freq = apriori(T[:3000], min_support=0.12); rl = rules(freq, 0.85)
    rl_ph = [r for r in rl if r[1] == {"PHISHING"}][:10]
    pd.DataFrame([{"if": ", ".join(sorted(r[0])), "then": "PHISHING", "support": round(r[2], 3), "confidence": round(r[3], 3),
                   "lift": round(r[4], 2)} for r in rl_ph]).to_csv(f"{R}/association_rules.csv", index=False)
    pca2 = pp.PCA(2, random_state=0).fit_transform(Xall)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].scatter(*pca2.T, c=yu, s=4, cmap="coolwarm"); ax[0].set_title("True labels (PCA-2D)")
    ax[1].scatter(*pca2.T, c=km.labels_, s=4, cmap="viridis"); ax[1].set_title(f"K-means k=2 (purity {purity:.2f})")
    plt.tight_layout(); plt.savefig(f"{R}/clusters.png", dpi=130); plt.close()

    summary = {"best_model": best_name, "test_metrics": res.iloc[0].to_dict(), "pca_components_95": int(pca.n_components_),
               "kmeans_purity": purity, "dbscan_outlier_fraction": out_frac, "n_samples": len(df)}
    json.dump(summary, open(f"{R}/summary.json", "w"), indent=2, default=float)
    say(f"\nBest model: {best_name}\nK-means purity={purity:.3f}  DBSCAN outliers={out_frac:.1%}\nSaved -> models/best_model.joblib, reports/")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/urls.csv"); ap.add_argument("--quick", action="store_true")
    main(ap.parse_args())