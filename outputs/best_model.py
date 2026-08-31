"""FM baseline retrained with a pairwise BPR loss over within-user (pos, neg) pairs.

Hypothesis: pointwise logloss spends gradient calibrating group-constant terms
(bias b, first-order user weight) that provably cancel out of within-user ranking.
BPR's loss depends only on z_i - z_j for pairs drawn from the SAME user, so any
term shared identically between i and j receives exactly zero gradient by
construction -- forcing capacity into the cross terms that actually move rank
order. GAUC is itself a pairwise (Mann-Whitney) statistic, so this is a direct
alignment of the training objective with the scored metric.

    python candidate.py --split valid --seed 42 --out scores.npy [--frac 0.01]
"""
import argparse

import numpy as np

from harness.data_guard import DataAPI

FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket")
K = 16
LR = 0.001
L2 = 1e-6
EPOCHS = 40
BATCH = 4096
PATIENCE = 4
CAP_PER_USER = 20  # max pairs sampled per user per epoch, bounds pair-set size


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

    def _adam_apply(self, gV, gW, db):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * db

    def pair_step(self, Xi, Xj):
        """BPR update on within-user (positive, negative) pairs.

        d = z_i - z_j; L = -log sigmoid(d).
        dL/dz_i = sigmoid(d) - 1 ; dL/dz_j = 1 - sigmoid(d).
        Any term identical between i and j (bias b, a user's own first-order
        weight when i,j share the same user) contributes gi + gj == 0 exactly.
        """
        B = Xi.shape[0]
        zi, Ei, Si = self.logits(Xi)
        zj, Ej, Sj = self.logits(Xj)
        d = zi - zj
        sig = sigmoid(d)
        gi = ((sig - 1.0) / B).astype(np.float32)
        gj = ((1.0 - sig) / B).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, Xi, gi[:, None])
        np.add.at(gV, Xi, gi[:, None, None] * (Si[:, None, :] - Ei))
        np.add.at(gW, Xj, gj[:, None])
        np.add.at(gV, Xj, gj[:, None, None] * (Sj[:, None, :] - Ej))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        db = float(gi.sum() + gj.sum())  # analytically ~0 within a user's pair
        self._adam_apply(gV, gW, db)

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def gauc_ndcg(users, labels, scores, k=5):
    """Local reimplementation used ONLY for epoch selection (matches evaluate.py)."""
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


def build_pairs(users, y, rng, cap):
    """For each user with mixed labels, sample up to `cap` (pos, neg) row-index pairs."""
    order = np.argsort(users, kind="stable")
    us = users[order]
    ys = y[order]
    idx_sorted = order
    starts = np.flatnonzero(np.concatenate(([True], us[1:] != us[:-1])))
    ends = np.concatenate((starts[1:], [len(us)]))
    pos_list = []
    neg_list = []
    for s, e in zip(starts, ends):
        seg_idx = idx_sorted[s:e]
        seg_y = ys[s:e]
        pos = seg_idx[seg_y == 1]
        neg = seg_idx[seg_y == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        n = min(len(pos), len(neg), cap)
        pos = pos.copy()
        neg = neg.copy()
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos_list.append(pos[:n])
        neg_list.append(neg[:n])
    if not pos_list:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(pos_list), np.concatenate(neg_list)


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
    users_tr = api.features("train")["user_id"].astype(np.int64)
    Xva, yva = enc["valid"], api.labels("valid")
    uva = api.features("valid")["user_id"]

    rng = np.random.default_rng(args.seed)
    cap = CAP_PER_USER
    if args.frac < 1.0:
        uniq_users = np.unique(users_tr)
        keep_users = rng.choice(
            uniq_users, size=max(1, int(len(uniq_users) * args.frac)), replace=False
        )
        mask = np.isin(users_tr, keep_users)
        Xtr, ytr, users_tr = Xtr[mask], ytr[mask], users_tr[mask]
        epochs, patience, cap = 2, 1, 5
    else:
        epochs, patience = EPOCHS, PATIENCE

    model = FM(dim, seed=args.seed)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        pi, ni = build_pairs(users_tr, ytr, rng, cap)
        if len(pi) == 0:
            break
        perm = rng.permutation(len(pi))
        pi, ni = pi[perm], ni[perm]
        for i in range(0, len(pi), BATCH):
            bi = pi[i:i + BATCH]
            bj = ni[i:i + BATCH]
            model.pair_step(Xtr[bi], Xtr[bj])
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

    np.save(args.out, model.predict(enc[args.split]).astype(np.float64))


if __name__ == "__main__":
    main()
