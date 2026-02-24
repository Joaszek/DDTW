import numpy as np
import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

DATA_ROOT = Path(r"./ml-data")
RESULTS_DIR = Path(r"./results")
RESULTS_DIR.mkdir(exist_ok=True)
CATEGORIES = ["BAS", "B6", "B10", "B15"]

PARALLEL_FILES = {
    "RR":  "ABP_RR-CBFV_RR",
    "SPO": "ABP_SPO-CBFV_SPO",
    "SPP": "ABP_SPP-CBFV_SPP",
}

TEST_RATIO_PER_CLASS = 0.2
RANDOM_STATE = 42
WINDOW_LEN = 32
WINDOW_STEP = 8
USE_LOG1P = True
USE_NORM_BY_MEDIAN = True
USE_RESAMPLED_SHAPE = True
RESAMPLE_LEN = 16

def load_cost_vector(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "costs" in df.columns:
        x = df["costs"].values
    else:
        x = df.iloc[:, 0].values
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def find_parallel_file(folder: Path, key: str) -> Path:
    """
    Szuka pliku w folderze, który zawiera wzorzec np. 'ABP_RR-CBFV_RR'
    i kończy się .csv (w tym .csv.csv).
    """
    pattern = PARALLEL_FILES[key].lower()
    candidates = []
    for p in folder.iterdir():
        if p.is_file() and p.name.lower().endswith(".csv") and pattern in p.name.lower():
            candidates.append(p)

    if len(candidates) == 0:
        raise FileNotFoundError(f"Nie znaleziono pliku dla {key} w {folder}")
    candidates = sorted(candidates)
    return candidates[-1]

def preprocess_cost_vector(x: np.ndarray) -> np.ndarray:
    x = x.copy()

    if USE_LOG1P:
        x = np.log1p(np.clip(x, 0, None))

    if USE_NORM_BY_MEDIAN:
        med = np.median(x)
        x = x / (med + 1e-8)

    # opcjonalnie: lekkie obcięcie ekstremów
    x = np.clip(x, 0, np.percentile(x, 99.5))

    return x.astype(np.float32)

def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((target_len,), dtype=np.float32)
    if len(x) == 1:
        return np.full((target_len,), float(x[0]), dtype=np.float32)

    old_idx = np.linspace(0, 1, num=len(x), dtype=np.float32)
    new_idx = np.linspace(0, 1, num=target_len, dtype=np.float32)
    y = np.interp(new_idx, old_idx, x).astype(np.float32)
    return y

def window_stats(x: np.ndarray):
    """
    Statystyki z okna 1D
    """
    x = np.asarray(x, dtype=np.float32)

    mean = float(x.mean())
    std = float(x.std())
    mn = float(x.min())
    mx = float(x.max())
    med = float(np.median(x))
    p10 = float(np.percentile(x, 10))
    p25 = float(np.percentile(x, 25))
    p75 = float(np.percentile(x, 75))
    p90 = float(np.percentile(x, 90))
    iqr = float(p75 - p25)

    # trend (slope) z regresji liniowej
    t = np.arange(len(x), dtype=np.float32)
    t_mean = t.mean()
    denom = float(np.sum((t - t_mean) ** 2) + 1e-8)
    slope = float(np.sum((t - t_mean) * (x - mean)) / denom)

    energy = float(np.mean(x * x))

    return [mean, std, mn, mx, med, p10, p25, p75, p90, iqr, slope, energy]

def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sa = a.std()
    sb = b.std()
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def build_window_features(rr_win, spo_win, spp_win):
    """
    Buduje cechy z okna (RR, SPO, SPP).
    """
    feats = []

    # statystyki per kanał
    rr_f = window_stats(rr_win)
    spo_f = window_stats(spo_win)
    spp_f = window_stats(spp_win)

    feats += rr_f + spo_f + spp_f

    # relacje (mean i std)
    rr_mean, rr_std = rr_f[0], rr_f[1]
    spo_mean, spo_std = spo_f[0], spo_f[1]
    spp_mean, spp_std = spp_f[0], spp_f[1]

    eps = 1e-8

    # różnice mean
    feats += [
        rr_mean - spo_mean,
        rr_mean - spp_mean,
        spo_mean - spp_mean,
    ]

    # proporcje mean
    feats += [
        rr_mean / (spo_mean + eps),
        rr_mean / (spp_mean + eps),
        spo_mean / (spp_mean + eps),
    ]

    # różnice std
    feats += [
        rr_std - spo_std,
        rr_std - spp_std,
        spo_std - spp_std,
    ]

    # korelacje wewnątrz okna
    feats += [
        corr_safe(rr_win, spo_win),
        corr_safe(rr_win, spp_win),
        corr_safe(spo_win, spp_win),
    ]

    # shape (resampling) - często robi duży upgrade
    if USE_RESAMPLED_SHAPE:
        rr_shape = resample_1d(rr_win, RESAMPLE_LEN)
        spo_shape = resample_1d(spo_win, RESAMPLE_LEN)
        spp_shape = resample_1d(spp_win, RESAMPLE_LEN)

        # normalizacja shape (tylko kształt)
        rr_shape = (rr_shape - rr_shape.mean()) / (rr_shape.std() + 1e-8)
        spo_shape = (spo_shape - spo_shape.mean()) / (spo_shape.std() + 1e-8)
        spp_shape = (spp_shape - spp_shape.mean()) / (spp_shape.std() + 1e-8)

        feats += rr_shape.tolist() + spo_shape.tolist() + spp_shape.tolist()

    return np.array(feats, dtype=np.float32)

all_rows = []
CATEGORIES = ["BAS", "B6", "B10", "B15"]

for cat in CATEGORIES:
    cat_dir = DATA_ROOT / cat

    rr_path = find_parallel_file(cat_dir, "RR")
    spo_path = find_parallel_file(cat_dir, "SPO")
    spp_path = find_parallel_file(cat_dir, "SPP")

    rr = preprocess_cost_vector(load_cost_vector(rr_path))
    spo = preprocess_cost_vector(load_cost_vector(spo_path))
    spp = preprocess_cost_vector(load_cost_vector(spp_path))

    # wyrównaj długość
    L = min(len(rr), len(spo), len(spp))
    rr, spo, spp = rr[:L], spo[:L], spp[:L]

    # generuj okna
    windows = []
    t0_list = []

    for start in range(0, L - WINDOW_LEN + 1, WINDOW_STEP):
        rr_win = rr[start:start+WINDOW_LEN]
        spo_win = spo[start:start+WINDOW_LEN]
        spp_win = spp[start:start+WINDOW_LEN]

        feats = build_window_features(rr_win, spo_win, spp_win)
        windows.append(feats)
        t0_list.append(start)

    if len(windows) == 0:
        raise ValueError(f"Za krótki sygnał w klasie {cat} (L={L}) na WINDOW_LEN={WINDOW_LEN}")

    X_cat = np.vstack(windows)
    y_cat = np.array([cat] * len(windows))

    df_cat = pd.DataFrame({
        "label": y_cat,
        "t0": t0_list
    })
    all_rows.append((X_cat, df_cat))

X = np.vstack([x for x, _ in all_rows])
meta = pd.concat([m for _, m in all_rows], ignore_index=True)
y = meta["label"].values

print("X shape:", X.shape)
print("y counts:\n", meta["label"].value_counts())

train_idx = []
test_idx = []

offset = 0
for (X_cat, meta_cat) in all_rows:
    n = len(meta_cat)
    split = int(n * (1 - TEST_RATIO_PER_CLASS))

    idx = np.arange(offset, offset + n)
    train_idx.extend(idx[:split])
    test_idx.extend(idx[split:])

    offset += n

train_idx = np.array(train_idx)
test_idx = np.array(test_idx)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Train:", X_train.shape, "Test:", X_test.shape)

models = {
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000))
    ]),
    "SVM-RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale"))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=600,
        random_state=RANDOM_STATE
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=RANDOM_STATE
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print("\n" + "="*80)
    print("MODEL:", name)
    print("Accuracy:", acc)
    print(classification_report(y_test, pred, digits=4))

    cm = confusion_matrix(y_test, pred, labels=CATEGORIES)
    disp = ConfusionMatrixDisplay(cm, display_labels=CATEGORIES)
    disp.plot(values_format="d")
    plt.title(name)
    plt.savefig(RESULTS_DIR / f"{name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    ax.bar(range(len(y_test)), pred == y_test, width=1.0)
    ax.set_title(f"{name} — correct predictions")
    ax.set_xlabel("sample")
    ax.set_ylabel("correct")
    plt.savefig(RESULTS_DIR / f"{name}_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        fi_df = pd.DataFrame({"feature_idx": range(len(imp)), "importance": imp})
        fi_df = fi_df.sort_values("importance", ascending=False)
        fi_df.to_csv(RESULTS_DIR / f"{name}_feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        top_n = min(30, len(imp))
        top = fi_df.head(top_n)
        ax.barh(range(top_n), top["importance"].values)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([f"feat_{i}" for i in top["feature_idx"].values])
        ax.invert_yaxis()
        ax.set_title(f"{name} — top {top_n} feature importances")
        plt.savefig(RESULTS_DIR / f"{name}_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

try:
    from catboost import CatBoostClassifier

    cb = CatBoostClassifier(
        depth=8,
        learning_rate=0.05,
        iterations=1200,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=RANDOM_STATE,
        verbose=200
    )

    cb.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    pred = cb.predict(X_test).reshape(-1)

    acc = accuracy_score(y_test, pred)
    print("\n" + "="*80)
    print("MODEL: CatBoost")
    print("Accuracy:", acc)
    print(classification_report(y_test, pred, digits=4))

    cm = confusion_matrix(y_test, pred, labels=CATEGORIES)
    disp = ConfusionMatrixDisplay(cm, display_labels=CATEGORIES)
    disp.plot(values_format="d")
    plt.title("CatBoost")
    plt.savefig(RESULTS_DIR / "CatBoost_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    ax.bar(range(len(y_test)), pred == y_test, width=1.0)
    ax.set_title("CatBoost — correct predictions")
    ax.set_xlabel("sample")
    ax.set_ylabel("correct")
    plt.savefig(RESULTS_DIR / "CatBoostplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    imp = cb.get_feature_importance()
    fi_df = pd.DataFrame({"feature_idx": range(len(imp)), "importance": imp})
    fi_df = fi_df.sort_values("importance", ascending=False)
    fi_df.to_csv(RESULTS_DIR / "CatBoost_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(30, len(imp))
    top = fi_df.head(top_n)
    ax.barh(range(top_n), top["importance"].values)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"feat_{i}" for i in top["feature_idx"].values])
    ax.invert_yaxis()
    ax.set_title(f"CatBoost — top {top_n} feature importances")
    plt.savefig(RESULTS_DIR / "CatBoost_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

except Exception as e:
    print("\n[INFO] CatBoost pominięty (brak biblioteki albo błąd importu).")
    print("Jeśli chcesz CatBoost: pip install catboost")
    print("Błąd:", repr(e))

