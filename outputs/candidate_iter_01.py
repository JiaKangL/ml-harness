"""Iteration: FM with within-user listwise softmax cross-entropy loss.

Same 5-field FM (user_id, video_id, author_id, tab, dur_bucket), k=16, Adam, as the
baseline -- but instead of pointwise logloss per row, each training step computes a
softmax over the raw scores WITHIN each user's impression group and a cross-entropy
against the normalized label mass (y_i / sum(y) for that group). Groups with zero
positives contribute no gradient (there is no ranking signal there: nDCG is always 0
regardless of order, and GAUC excludes them by definition), which matches the metric's
own treatment of those users exactly.

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
PATIENCE = 4
BATCH_GROUPS = 200  # ~ mean 43.5 impressions/user -> ~8700 rows/batch, similar scale to baseline's 8192


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


def group_boundaries(gids):
    gids = np.asarray(gids)
    if gids.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    starts = np.flatnonzero(np.concatenate(([True], gids[1:] != gids[:-1])))
    sizes = np.diff(np.concatenate((starts, [gids.shape[0]])))
    return starts, sizes


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

    def _apply(self, gV, gW, g_b):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g_b

    def step_listwise(self, X, y, gid_local, n_groups):
        """One SGD step of within-user listwise softmax cross-entropy.

        gid_local: int array in [0, n_groups) mapping each row of X/y to its group
        (contiguous block within this batch). Groups with sum(y)==0 get zero gradient.
        """
        z, E, S = self.logits(X)
        y = y.astype(np.float64)
        z64 = z.astype(np.float64)

        segmax = np.full(n_groups, -np.inf, dtype=np.float64)
        np.maximum.at(segmax, gid_local, z64)
        ez = np.exp(z64 - segmax[gid_local])
        segsum = np.zeros(n_groups, dtype=np.float64)
        np.add.at(segsum, gid_local, ez)
        soft = ez / segsum[gid_local]

        sy = np.zeros(n_groups, dtype=np.float64)
        np.add.at(sy, gid_local, y)
        active = sy > 0
        active_row = active[gid_local]
        sy_safe = np.where(sy > 0, sy, 1.0)
        target = np.where(active_row, y / sy_safe[gid_local], 0.0)

        g = np.where(active_row, soft - target, 0.0)
        n_active = active.sum()
        if n_active == 0:
            return
        g = (g / n_active).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self._apply(gV, gW, float(g.sum()))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def gauc_ndcg(users, labels, scores, k=5):
    """Fast local reimplementation used ONLY for epoch selection (early stopping)."""
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
    train_gids = api.groups("train")

    Xva, yva = enc["valid"], api.labels("valid")
    uva = api.features("valid")["user_id"]

    rng = np.random.default_rng(args.seed)
    if args.frac < 1.0:
        # Sample USERS, never rows: rows must stay grouped intact for the listwise
        # loss (and for the metric) or the group structure is shredded.
        users = api.features("train")["user_id"]
        uniq_users = np.unique(users)
        keep_users = rng.choice(
            uniq_users, size=max(1, int(len(uniq_users) * args.frac)), replace=False
        )
        mask = np.isin(users, keep_users)
        Xtr, ytr = Xtr[mask], ytr[mask]
        train_gids = train_gids[mask]
        epochs, patience = 2, 1
    else:
        epochs, patience = EPOCHS, PATIENCE

    starts, sizes = group_boundaries(train_gids)
    n_groups = starts.shape[0]

    model = FM(dim, seed=args.seed)
    best, best_state, bad = -1.0, None, 0

    for _ in range(epochs):
        perm = rng.permutation(n_groups)
        sizes_p = sizes[perm]
        starts_p = starts[perm]
        row_idx = np.concatenate([np.arange(s, s + n) for s, n in zip(starts_p, sizes_p)]) \
            if n_groups > 0 else np.array([], dtype=np.int64)
        cum = np.concatenate(([0], np.cumsum(sizes_p))) if n_groups > 0 else np.array([0])

        n_batches = max(1, -(-n_groups // BATCH_GROUPS)) if n_groups > 0 else 0
        for b in range(n_batches):
            g0, g1 = b * BATCH_GROUPS, min((b + 1) * BATCH_GROUPS, n_groups)
            if g1 <= g0:
                continue
            r0, r1 = cum[g0], cum[g1]
            rows = row_idx[r0:r1]
            if rows.shape[0] == 0:
                continue
            Xb = Xtr[rows]
            yb = ytr[rows]
            sizes_b = sizes_p[g0:g1]
            gid_local = np.repeat(np.arange(len(sizes_b)), sizes_b)
            model.step_listwise(Xb, yb, gid_local, len(sizes_b))

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
