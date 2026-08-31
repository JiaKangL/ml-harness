"""
Iteration: sequence axis -- SVD++-style liked-item profile as a target-dependent
history signal added on top of the FM baseline.

Mechanism: FM baseline scores (user, video) purely through a single static user
embedding. The ledger already proved pure user-side constants contribute exactly zero
to within-user ranking. But a user's *history* can act through the item side: we build
profile(u) = mean of FM video-embeddings the user long-viewed in train, then score
candidates by dot(profile(u), V[video]) -- a quantity that varies row-to-row within a
user's group because different candidates have different embeddings. This is a
lightweight target-attention / SVD++ implicit-feedback term, added as a second feature
blended with the FM logit via a 2-parameter logistic regression fit on train.

    python candidate.py --split valid --seed 42 --out scores.npy [--frac 0.01]
"""
import argparse

import numpy as np

from harness.data_guard import DataAPI

FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket")
VIDEO_COL = FIELDS.index("video_id")
K = 16
LR = 0.001
L2 = 1e-6
EPOCHS = 40
BATCH = 8192
PATIENCE = 4


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def raw_fields(api, split, edges):
    f = api.features(split)
    author = api.video_feature("author_id")[f["video_id"]]
    bucket = np.searchsorted(edges, f["duration_ms"].astype(np.float64))
    return [
        f["user_id"].astype(np.int64),
        f["video_id"].astype(np.int64),
        author.astype(np.int64),
        f["tab"].astype(np.int64),
        bucket.astype(np.int64),
    ]


def encode(api, splits):
    """Vocab per field from train, one UNK slot each, then a single flat index space."""
    edges = np.quantile(
        api.features("train")["duration_ms"].astype(np.float64),
        np.linspace(0, 1, 11)[1:-1],
    )
    train_cols = raw_fields(api, "train", edges)
    vocabs = [{v: i for i, v in enumerate(np.unique(col).tolist())} for col in train_cols]
    unk = [len(v) for v in vocabs]
    dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    out = {}
    for split in splits:
        cols = train_cols if split == "train" else raw_fields(api, split, edges)
        X = np.empty((cols[0].shape[0], len(FIELDS)), dtype=np.int32)
        for i, col in enumerate(cols):
            lookup = vocabs[i]
            X[:, i] = np.fromiter(
                (lookup.get(int(v), unk[i]) for v in col), dtype=np.int64, count=col.shape[0]
            ) + offsets[i]
        out[split] = X
    return out, int(sum(dims))


class FM:
    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def gauc_ndcg(users, labels, scores, k=5):
    """Local reimplementation used ONLY for epoch selection during training (fast);
    every reported score comes from the frozen evaluate.py via the harness."""
    order = np.lexsort((-scores, users))
    u = users[order]
    y = labels[order].astype(np.float64)
    starts = np.flatnonzero(np.concatenate(([True], u[1:] != u[:-1])))
    sizes = np.diff(np.concatenate((starts, [u.shape[0]])))
    gnum = gden = 0.0
    nd = []
    for s, n in zip(starts, sizes):
        labs = y[s:s + n]
        npos = labs.sum()
        if 0 < npos < n:
            ranks = np.arange(n, 0, -1, dtype=np.float64)
            gnum += npos * ((ranks[labs > 0].sum() - npos * (npos + 1) / 2) / (npos * (n - npos)))
            gden += npos
        top = labs[:k]
        disc = 1.0 / np.log2(np.arange(top.shape[0]) + 2)
        dcg = float((top * disc).sum())
        ideal = np.sort(labs)[::-1][:k]
        idcg = float((ideal * (1.0 / np.log2(np.arange(ideal.shape[0]) + 2))).sum())
        nd.append(dcg / idcg if idcg else 0.0)
    gauc = gnum / gden if gden else 0.5
    return 0.5 * (gauc + (sum(nd) / len(nd) if nd else 0.0))


def build_history(V, vid_idx_train, user_id_train, y_train, k, seed=0):
    """profile(u) = mean FM video-embedding over train impressions with label=1."""
    users_unique, inv = np.unique(user_id_train, return_inverse=True)
    n_u = len(users_unique)
    sum_vec = np.zeros((n_u, k), dtype=np.float64)
    pos_mask = y_train > 0.5
    np.add.at(sum_vec, inv[pos_mask], V[vid_idx_train[pos_mask]].astype(np.float64))
    counts = np.bincount(inv[pos_mask], minlength=n_u).astype(np.float64)
    hist = np.zeros_like(sum_vec)
    nz = counts > 0
    hist[nz] = sum_vec[nz] / counts[nz][:, None]
    user_map = {int(u): i for i, u in enumerate(users_unique.tolist())}
    return user_map, hist


def score_hist_for_rows(user_map, hist, user_ids, vid_idx, V, k):
    n = len(user_ids)
    idx = np.fromiter((user_map.get(int(u), -1) for u in user_ids), dtype=np.int64, count=n)
    vecs = np.zeros((n, k), dtype=np.float64)
    mask = idx >= 0
    if mask.any():
        vecs[mask] = hist[idx[mask]]
    return np.einsum("ij,ij->i", vecs, V[vid_idx].astype(np.float64))


def fit_blend(f1, f2, y, seed=0, iters=500, lr=0.3):
    """Tiny 2-feature logistic regression: sigmoid(w0*f1 + w1*f2 + b) -> y.
    f1, f2 are pre-standardized. Full-batch gradient descent, deterministic."""
    rng = np.random.default_rng(seed)
    w = np.zeros(2, dtype=np.float64)
    b = 0.0
    X = np.column_stack([f1, f2])
    n = len(y)
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = (p - y) / n
        gw = X.T @ g
        gb = g.sum()
        w -= lr * gw
        b -= lr * gb
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    args = ap.parse_args()

    api = DataAPI()
    wanted = {"train", "valid", args.split}
    enc, dim = encode(api, sorted(wanted))
    Xtr = enc["train"]
    ytr = api.labels("train").astype(np.float32)
    Xva, yva = enc["valid"], api.labels("valid")
    uva = api.features("valid")["user_id"]
    user_id_train_full = api.features("train")["user_id"]

    rng = np.random.default_rng(args.seed)
    if args.frac < 1.0:
        # Sample USERS, never rows, so impression groups stay intact.
        users_all = user_id_train_full
        uniq = np.unique(users_all)
        keep_users = rng.choice(uniq, size=max(1, int(len(uniq) * args.frac)), replace=False)
        mask = np.isin(users_all, keep_users)
        Xtr, ytr = Xtr[mask], ytr[mask]
        user_id_train = users_all[mask]
        epochs, patience = 2, 1
    else:
        user_id_train = user_id_train_full
        epochs, patience = EPOCHS, PATIENCE

    # ---- stage 1: train the FM exactly like the baseline ----
    model = FM(dim, seed=args.seed)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), BATCH):
            batch = idx[i:i + BATCH]
            model.step(Xtr[batch], ytr[batch])
        primary = gauc_ndcg(uva, yva, model.predict(Xva))
        if primary > best + 1e-5:
            best, bad = primary, 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.V, model.W, model.b = best_state

    V = model.V.astype(np.float64)

    # ---- stage 2: build the per-user liked-item profile from (possibly subsampled) train ----
    vid_idx_train = Xtr[:, VIDEO_COL]
    user_map, hist = build_history(V, vid_idx_train, user_id_train, ytr, K, seed=args.seed)

    # score_hist for train (to fit blend), valid (sanity), and target split
    score_hist_train = score_hist_for_rows(user_map, hist, user_id_train, vid_idx_train, V, K)
    fm_logit_train = model.predict(Xtr)

    vid_idx_va = Xva[:, VIDEO_COL]
    score_hist_va = score_hist_for_rows(user_map, hist, uva, vid_idx_va, V, K)
    fm_logit_va = model.predict(Xva)

    X_target = enc[args.split]
    f_target = api.features(args.split)
    user_id_target = f_target["user_id"]
    vid_idx_target = X_target[:, VIDEO_COL]
    score_hist_target = score_hist_for_rows(user_map, hist, user_id_target, vid_idx_target, V, K)
    fm_logit_target = model.predict(X_target)

    # ---- stage 3: standardize features using train statistics, fit 2-feature blend ----
    def standardize(x, mu, sd):
        return (x - mu) / (sd if sd > 1e-8 else 1.0)

    mu1, sd1 = float(fm_logit_train.mean()), float(fm_logit_train.std())
    mu2, sd2 = float(score_hist_train.mean()), float(score_hist_train.std())

    f1_tr = standardize(fm_logit_train, mu1, sd1)
    f2_tr = standardize(score_hist_train, mu2, sd2)
    w, b = fit_blend(f1_tr, f2_tr, ytr.astype(np.float64), seed=args.seed)

    f1_target = standardize(fm_logit_target, mu1, sd1)
    f2_target = standardize(score_hist_target, mu2, sd2)
    final = w[0] * f1_target + w[1] * f2_target + b

    np.save(args.out, final.astype(np.float64))


if __name__ == "__main__":
    main()
