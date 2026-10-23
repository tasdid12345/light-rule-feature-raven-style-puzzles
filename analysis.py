"""Simulated RAVEN-style puzzles: MLP candidate scorer vs LightGBM on rule-consistency features."""
import json, os, pickle, sys, time
from itertools import product
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier

SEED = 42
N_TRAIN, N_TEST = 4000, 2000
np.random.seed(SEED)


# ---------- simulation ----------
def gen_attr(rng, rule):
    """3x3 grid of one attribute (values 0-9). Rules: constant, progression, arithmetic, distribute-three."""
    g = np.zeros((3, 3), int)
    if rule == 0:
        g[:] = rng.integers(0, 10, 3)[:, None]
    elif rule == 1:
        d = rng.choice([-2, -1, 1, 2])
        lo, hi = (0, 9 - 2 * d) if d > 0 else (-2 * d, 9)
        g = rng.integers(lo, hi + 1, 3)[:, None] + d * np.arange(3)[None]
    elif rule == 2:
        s = rng.choice([1, -1])
        g[:, :2] = rng.integers(0, 10, (3, 2))
        g[:, 2] = (g[:, 0] + s * g[:, 1]) % 10
    else:
        vals = rng.choice(10, 3, replace=False)
        g = np.stack([rng.permutation(vals) for _ in range(3)])
    return g


BITS = np.array(list(product([0, 1], repeat=3)))  # 8 leaves of a balanced tree: level i flips attribute i


def make_puzzles(n, rng):
    ctx, cands, ys = [], [], []
    for _ in range(n):
        grids = [gen_attr(rng, rng.integers(0, 4)).ravel() for _ in range(3)]
        panels = np.stack(grids, 1)  # (9, 3): shape, size, colour
        ans = panels[8]
        alt = (ans + rng.integers(1, 10, 3)) % 10  # one changed value per attribute
        c = np.where(BITS == 1, alt, ans)[rng.permutation(8)] if False else None
        perm = rng.permutation(8)
        c = np.where(BITS == 1, alt, ans)[perm]
        ctx.append(panels[:8]); cands.append(c); ys.append(int(np.argmax(perm == 0)))
    return np.array(ctx), np.array(cands), np.array(ys)


# ---------- feature builders ----------
def onehot_X(ctx, cands):
    """Baseline input: one-hot of 8 context panels + one candidate (270 dims) per row."""
    p = np.concatenate([np.repeat(ctx, 8, 0), cands.reshape(-1, 1, 3)], 1)
    return np.eye(10, dtype=np.float32)[p].reshape(len(p), -1)


def hyp(a, b, c):
    """Does row (a,b,c) satisfy each of 5 rule hypotheses? Returns flags (M,5) and parameters (M,5,3)."""
    s = np.sort(np.stack([a, b, c], -1), -1)
    d1, d2 = b - a, c - b
    z = np.zeros_like(s)
    sat = np.stack([(a == b) & (b == c), (d1 == d2) & (np.abs(d1) >= 1) & (np.abs(d1) <= 2),
                    (a + b) % 10 == c, (a - b) % 10 == c, (s[:, 0] < s[:, 1]) & (s[:, 1] < s[:, 2])], 1)
    par = np.stack([z, np.repeat(d1[:, None], 3, 1), z, z, s], 1)
    return sat, par


def rule_X(ctx, cands):
    """Improved input: per attribute, which rules hold in rows 1-2, and whether the candidate row 3 obeys them."""
    cx, cd = np.repeat(ctx, 8, 0), cands.reshape(-1, 3)
    feats, viol = [], 0
    for j in range(3):
        v = cx[:, :, j]
        s1, p1 = hyp(v[:, 0], v[:, 1], v[:, 2])
        s2, p2 = hyp(v[:, 3], v[:, 4], v[:, 5])
        s3, p3 = hyp(v[:, 6], v[:, 7], cd[:, j])
        r12 = s1 & s2 & (p1 == p2).all(-1)
        r3 = s3 & (p3 == p1).all(-1)
        prod = r12 & r3
        broken = (r12 & ~r3).any(1)
        viol = viol + (r12.any(1) & ~prod.any(1))
        feats += [r12, r3, prod, prod.any(1)[:, None], broken[:, None]]
    feats.append(np.asarray(viol)[:, None])
    return np.concatenate(feats, 1).astype(np.float32)


# ---------- shared model helpers ----------
def puzzle_acc(model, X, y):
    """Pick the highest-scoring of the 8 candidates per puzzle; return accuracy."""
    s = model.predict_proba(X)[:, 1].reshape(-1, 8)
    return float((s.argmax(1) == y).mean())


def rows_of(idx):
    return (idx[:, None] * 8 + np.arange(8)).ravel()


def cv_select(make, grid, X, yp, y):
    """3-fold CV over puzzles on the training set; returns best params and its CV accuracy."""
    best = (-1, None)
    for params in grid:
        accs = []
        for tr, va in KFold(3, shuffle=True, random_state=SEED).split(y):
            m = make(params, SEED).fit(X[rows_of(tr)], yp[rows_of(tr)])
            accs.append(puzzle_acc(m, X[rows_of(va)], y[va]))
        print(f"  CV {params}: {np.mean(accs):.3f}")
        if np.mean(accs) > best[0]:
            best = (float(np.mean(accs)), params)
    return best[1], best[0]


def run(name, make, grid, Xtr, Xte, yp, ytr, yte, extra_time=0.0):
    print(f"[{name}] selecting hyper-parameters by CV")
    params, cv = cv_select(make, grid, Xtr, yp, ytr)
    accs, times, sizes = [], [], []
    for seed in (42, 43, 44):
        t = time.time()
        m = make(params, seed).fit(Xtr, yp)
        times.append(time.time() - t + extra_time)
        accs.append(puzzle_acc(m, Xte, yte))  # test set used once per final model
        sizes.append(len(pickle.dumps(m)))
        print(f"  seed {seed}: test acc {accs[-1]:.4f}, train {times[-1]:.1f}s")
    return dict(acc=accs, time=float(np.mean(times)), size=float(np.mean(sizes)), params=params, cv=cv)


def main():
    rng = np.random.default_rng(SEED)
    ctx, cands, y = make_puzzles(N_TRAIN + N_TEST, rng)
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, train_size=N_TRAIN, test_size=N_TEST, stratify=y, random_state=SEED)
    ytr, yte = y[itr], y[ite]
    yp = (np.arange(8)[None] == ytr[:, None]).ravel().astype(int)
    print(f"Simulated {len(y)} puzzles; train {len(itr)}, test {len(ite)}")

    Xb_tr, Xb_te = onehot_X(ctx[itr], cands[itr]), onehot_X(ctx[ite], cands[ite])
    t = time.time()
    Xr_tr = rule_X(ctx[itr], cands[itr])
    feat_time = time.time() - t  # feature extraction is counted in the improved model's training time
    Xr_te = rule_X(ctx[ite], cands[ite])

    mlp = lambda p, s: MLPClassifier(hidden_layer_sizes=p, max_iter=60, early_stopping=True, random_state=s)
    gbm = lambda p, s: LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=p, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, random_state=s, verbose=-1, n_jobs=2)
    base = run("MLP", mlp, [(128,), (256, 128)], Xb_tr, Xb_te, yp, ytr, yte)
    imp = run("LightGBM", gbm, [7, 31], Xr_tr, Xr_te, yp, ytr, yte, extra_time=feat_time)

    bs, is_ = float(np.mean(base["acc"])), float(np.mean(imp["acc"]))
    metrics = dict(
        baseline_score=bs, improved_score=is_,
        baseline_score_std=float(np.std(base["acc"])), improved_score_std=float(np.std(imp["acc"])),
        improvement=is_ - bs, chance_accuracy=1 / 8,
        baseline_cv_accuracy=base["cv"], improved_cv_accuracy=imp["cv"],
        baseline_train_time_s=base["time"], improved_train_time_s=imp["time"],
        baseline_model_bytes=base["size"], improved_model_bytes=imp["size"],
        train_speedup_x=base["time"] / imp["time"])
    if not all(np.isfinite(v) for v in metrics.values()):
        sys.exit("non-finite metrics")
    print({k: round(v, 4) for k, v in metrics.items()})

    os.makedirs("figures", exist_ok=True)
    plt.bar(["MLP scorer", "LightGBM rules"], [bs, is_], yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]])
    plt.axhline(1 / 8, color="gray", ls="--", label="chance")
    plt.ylabel("test answer accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig("figures/accuracy.png", dpi=120)

    res = dict(
        question="Can a small gradient-boosted ranker on rule-consistency features beat a standard MLP candidate scorer (stand-in for CPCNet) at picking the answer panel in simulated Raven-style matrices, at lower training cost?",
        method="Seeded simulation of RAVEN-style 3x3 puzzles (3 attributes, 4 row rules, I-RAVEN-style balanced-tree candidates). Both models train on 4000 puzzles and are tested once on 2000 held-out puzzles; the answer is the top-scoring of 8 candidates. Hyper-parameters chosen by 3-fold CV on the training set; 3 seeds each.",
        dataset=dict(name="Simulated RAVEN-style puzzles", source_url="generated in analysis.py (numpy, seed 42); no download",
                     n_rows=int(len(y)), n_train=int(len(itr)), n_test=int(len(ite)),
                     same_as_paper=False, baseline_is_papers_method=False),
        metric_name="answer accuracy (8-way, test set)", higher_is_better=True,
        baseline=f"sklearn MLPClassifier scoring one-hot (8 context panels + candidate); hidden layers {base['params']} chosen by CV. Not CPCNet.",
        improved=f"LightGBM binary scorer on per-attribute rule-consistency features (constant/progression/add/sub/distribute-three) from rows 1-2 and the candidate-completed row 3; num_leaves={imp['params']} chosen by CV.",
        metrics=metrics,
        notes="Simulation only: results do not transfer to real RAVEN/I-RAVEN/PGM or to CPCNet. The improved model uses hand-coded knowledge of the rule families (strong inductive bias), whereas the MLP must learn them from raw one-hot input. Rule-family overlap in random rows can make a few puzzles ambiguous. Test split is 33% (4000/2000 as specified, above the usual 20-30%). Train time of the improved model includes feature extraction; model size is the pickled size in bytes. Single process, small CPU budget for both.")
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
