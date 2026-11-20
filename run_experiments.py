"""
Reproduce the Iris / Wine results of Bai et al. (2014), "Sparse Extreme
Learning Machine for Classification", IEEE Trans. Cybernetics 44(10).

Examples
--------
    python run_experiments.py                 # paper's hyper-parameters (Table III), ~1 min
    python run_experiments.py --params cv     # re-select hyper-parameters by 5-fold CV
    python run_experiments.py --protocol fixed  # one fixed split, 20 shuffles (paper's style)

Protocol (Section VI of the paper)
    * 20 trials per dataset; Iris 75/75 and Wine 89/89 train/test samples
    * attributes linearly scaled to [-1, 1] with training-set factors
    * multiclass: OAO for SVM and sparse ELM, multi-output for unified ELM
    * L = 1000 random hidden nodes for sigmoid / sinusoid ELMs (multiclass)
    * C, sigma in {0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000},
      m in {1..5}, chosen by 5-fold CV on the training set
    * sparse ELM tolerance eps = 1e-3
"""
import argparse
import os
import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from elm import SparseELM, UnifiedELM, make_svm
from paper_results import PAPER_ACC, PAPER_NSV, PAPER_PARAMS, SPLIT

GRID = [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
DEGREES = [1, 2, 3, 4, 5]
CONFIGS = [("SVM", "gaussian"), ("SVM", "poly"),
           ("UELM", "gaussian"), ("UELM", "poly"), ("UELM", "sigmoid"), ("UELM", "sinusoid"),
           ("SELM", "gaussian"), ("SELM", "poly"), ("SELM", "sigmoid"), ("SELM", "sinusoid")]
NAMES = {"SVM": "SVM", "UELM": "Unified ELM", "SELM": "Sparse ELM"}
LOADERS = {"iris": load_iris, "wine": load_wine}


def build(method, kind, C, param, rng, n_hidden=1000):
    if method == "SVM":
        return make_svm(kind, C, param)
    cls = UnifiedELM if method == "UELM" else SparseELM
    return cls(kind, C, param, n_hidden=n_hidden, rng=rng)


def n_support(model):
    ns = model.n_support_
    return int(np.sum(ns)) if np.ndim(ns) else int(ns)


def scale(Xtr, Xte):
    sc = MinMaxScaler(feature_range=(-1, 1)).fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)   # test may fall slightly outside [-1, 1]


def cv_select(method, kind, X, y, seed, n_hidden):
    """5-fold CV on the (already scaled) training set, the paper's grid."""
    second = GRID if kind == "gaussian" else DEGREES if kind == "poly" else [None]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))
    results = []
    for C, p in product(GRID, second):
        accs = []
        for k, (tr, va) in enumerate(folds):
            rng = np.random.default_rng(seed * 7919 + k)
            m = build(method, kind, C, p, rng, n_hidden).fit(X[tr], y[tr])
            accs.append(np.mean(m.predict(X[va]) == y[va]))
        results.append(((C, p), float(np.mean(accs))))
    # With 60-70 training samples many grid points tie.  The paper does not say
    # how ties are broken; we prefer the simplest (smoothest) model:
    # largest sigma / lowest degree m, then smallest C.
    best_acc = max(a for _, a in results)
    tied = [cp for cp, a in results if a >= best_acc - 1e-12]

    def simplicity(cp):
        C, p = cp
        if kind == "gaussian":
            return (-p, C)
        if kind == "poly":
            return (p, C)
        return (C,)
    return min(tied, key=simplicity), best_acc


def splits(X, y, n_train, protocol, n_trials, seed):
    if protocol == "random":
        for t in range(n_trials):
            yield train_test_split(X, y, train_size=n_train, stratify=y,
                                   random_state=seed + t)
    else:  # one fixed split, then shuffle train and test sets separately
        Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=n_train,
                                              stratify=y, random_state=seed)
        rng = np.random.default_rng(seed)
        for _ in range(n_trials):
            p, q = rng.permutation(len(ytr)), rng.permutation(len(yte))
            yield Xtr[p], Xte[q], ytr[p], yte[q]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["iris", "wine"], choices=list(LOADERS))
    ap.add_argument("--params", choices=["paper", "cv"], default="paper",
                    help="paper = Table III values; cv = 5-fold CV on the first training split")
    ap.add_argument("--protocol", choices=["random", "fixed"], default="random",
                    help="random = new stratified split per trial; fixed = one split, 20 shuffles")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=1000, help="random hidden nodes L")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.params}_{args.protocol}"

    rows, chosen = [], []
    for ds in args.datasets:
        X, y = LOADERS[ds](return_X_y=True)
        n_train = SPLIT[ds]
        print(f"\n=== {ds}: {len(y)} samples, {X.shape[1]} features, "
              f"{len(np.unique(y))} classes, {n_train} train / {len(y) - n_train} test ===")

        # ---- hyper-parameters -------------------------------------------------
        params = {}
        if args.params == "paper":
            params = dict(PAPER_PARAMS[ds])
        else:
            Xtr, Xte, ytr, _ = train_test_split(X, y, train_size=n_train, stratify=y,
                                                random_state=args.seed)
            Xtr, _ = scale(Xtr, Xte)
            for cfg in CONFIGS:
                t0 = time.time()
                params[cfg], cv_acc = cv_select(*cfg, Xtr, ytr, args.seed, args.hidden)
                print(f"  CV {NAMES[cfg[0]]:12s} {cfg[1]:9s} -> C={params[cfg][0]}, "
                      f"param={params[cfg][1]}  (cv acc {100 * cv_acc:.2f}%, "
                      f"{time.time() - t0:.1f}s)")
        for cfg, (C, p) in params.items():
            chosen.append(dict(dataset=ds, method=cfg[0], kind=cfg[1], C=C, param=p,
                               paper_C=PAPER_PARAMS[ds][cfg][0],
                               paper_param=PAPER_PARAMS[ds][cfg][1]))

        # ---- 20 trials -----------------------------------------------------------
        for t, (Xtr, Xte, ytr, yte) in enumerate(
                splits(X, y, n_train, args.protocol, args.trials, args.seed)):
            Xtr, Xte = scale(Xtr, Xte)
            for cfg in CONFIGS:
                C, p = params[cfg]
                rng = np.random.default_rng(10_000 * args.seed + t)
                model = build(*cfg, C, p, rng, args.hidden)
                t0 = time.perf_counter()
                model.fit(Xtr, ytr)
                t_train = time.perf_counter() - t0
                t0 = time.perf_counter()
                pred = model.predict(Xte)
                t_test = time.perf_counter() - t0
                rows.append(dict(dataset=ds, trial=t, method=cfg[0], kind=cfg[1],
                                 train_acc=100 * np.mean(model.predict(Xtr) == ytr),
                                 test_acc=100 * np.mean(pred == yte),
                                 n_sv=n_support(model), train_time=t_train,
                                 test_time=t_test))
        print(f"  finished {args.trials} trials")

    per_trial = pd.DataFrame(rows)
    per_trial.to_csv(os.path.join(args.out, f"per_trial_{tag}.csv"), index=False)
    pd.DataFrame(chosen).to_csv(os.path.join(args.out, f"params_{tag}.csv"), index=False)

    # ---- summary vs. paper -------------------------------------------------------
    g = per_trial.groupby(["dataset", "method", "kind"], sort=False)
    summ = g.agg(train_acc=("train_acc", "mean"), test_acc=("test_acc", "mean"),
                 test_std=("test_acc", "std"), n_sv=("n_sv", "mean"),
                 train_time_ms=("train_time", lambda s: 1000 * s.mean())).reset_index()
    summ["paper_test_acc"] = [PAPER_ACC[d][(m, k)][1] for d, m, k in
                              zip(summ.dataset, summ.method, summ.kind)]
    summ["paper_test_std"] = [PAPER_ACC[d][(m, k)][2] for d, m, k in
                              zip(summ.dataset, summ.method, summ.kind)]
    summ["diff"] = summ.test_acc - summ.paper_test_acc
    summ["paper_n_sv"] = [PAPER_NSV[d].get((m, k), np.nan) for d, m, k in
                          zip(summ.dataset, summ.method, summ.kind)]
    summ.to_csv(os.path.join(args.out, f"summary_{tag}.csv"), index=False)

    lines = [f"# Results: params={args.params}, protocol={args.protocol}, "
             f"trials={args.trials}, seed={args.seed}\n"]
    for ds in args.datasets:
        lines.append(f"\n## {ds.capitalize()}\n")
        lines.append("| Method | Mapping | Test acc. (ours) | Test acc. (paper) | Diff | "
                     "#SV (ours) | #SV (paper) | Train acc. (ours) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in summ[summ.dataset == ds].iterrows():
            psv = "-" if np.isnan(r.paper_n_sv) else f"{r.paper_n_sv:g}"
            osv = "all (dense)" if r.method == "UELM" else f"{r.n_sv:.1f}"
            lines.append(f"| {NAMES[r.method]} | {r.kind} | {r.test_acc:.2f} ± {r.test_std:.2f} | "
                         f"{r.paper_test_acc:.2f} ± {r.paper_test_std:.2f} | {r['diff']:+.2f} | "
                         f"{osv} | {psv} | {r.train_acc:.2f} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, f"summary_{tag}.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"Saved per-trial CSV, summary CSV/Markdown and parameters to ./{args.out}/")


if __name__ == "__main__":
    main()
