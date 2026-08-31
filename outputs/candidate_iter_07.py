"""FM baseline + a train-only auxiliary watch-fraction head with a one-sided
(censored) loss, sharing the main model's embedding table.

Hypothesis: the earlier plain-MSE watch-fraction aux head (ledger:
fm_shared_embedding_watchfraction_aux_head, +0.0001) failed because squared
error penalizes over-prediction on completed plays even though the true
watch intent for those rows is only known to be >= the observed value
(right-censored). Replacing it with a one-sided loss that only penalizes
under-prediction on censored rows should give a cleaner gradient into the
shared item-side embeddings, without changing the main pointwise logloss or
the inference-time scoring function at all (aux head does not touch the
main logits).

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
BATCH = 8192
PATIENCE = 4
LAMBDA_AUX = 0.3  # weight of the auxiliary watch-fraction loss vs main logloss


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
        # auxiliary watch-fraction head: shares self.V, has its own linear readout
        self.aux_vec = rng.normal(0, 0.01, k).astype(np.float32)
        self.b_aux = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.m_aux = np.zeros_like(self.aux_vec)
        self.v_aux = np.zeros_like(self.aux_vec)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y, y_wt=None, censored=None, lam=0.0):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))

        g_aux_vec = np.zeros_like(self.aux_vec)
        g_b_aux = 0.0
        if lam > 0 and y_wt is not None:
            pred_wt = self.b_aux + S @ self.aux_vec
            diff = (y_wt - pred_wt).astype(np.float32)
            # censored rows: only penalize under-prediction (diff > 0).
            # non-censored rows: fully observed, standard two-sided squared error.
            active = np.where(censored, diff > 0, np.ones_like(diff, dtype=bool))
            gp = np.zeros_like(diff)
            gp[active] = -2.0 * diff[active] / B
            gp *= lam
            g_b_aux = float(gp.sum())
            g_aux_vec = (gp[:, None] * S).sum(0)
            contrib = gp[:, None] * self.aux_vec[None, :]
            for f in range(X.shape[1]):
                np.add.at(gV, X[:, f], contrib)

        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW),
            (self.aux_vec, g_aux_vec, self.m_aux, self.v_aux),
        ):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        self.b_aux -= self.lr * g_b_aux

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def gauc_ndcg(users, labels, scores, k=5):
    """Local reimplementation used only for epoch selection; every reported number
    comes from the frozen evaluate.py via the harness."""
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
    Xva, yva = enc["valid"], api.labels("valid")
    uva = api.features("valid")["user_id"]

    # train-only auxiliary watch-fraction target (censored at video completion)
    aux = api.aux_targets("train")
    play_ms = aux["play_time_ms"].astype(np.float64)
    dur_ms = api.features("train")["duration_ms"].astype(np.float64)
    watch_frac = play_ms / np.maximum(dur_ms, 1.0)
    censored_all = watch_frac >= 1.0
    y_wt_all = np.minimum(watch_frac, 1.0).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    if args.frac < 1.0:
        # Sample USERS, not rows, so impression groups stay intact.
        users = api.features("train")["user_id"]
        uniq = np.unique(users)
        keep_users = rng.choice(uniq, size=max(1, int(len(uniq) * args.frac)), replace=False)
        mask = np.isin(users, keep_users)
        Xtr, ytr = Xtr[mask], ytr[mask]
        y_wt_all = y_wt_all[mask]
        censored_all = censored_all[mask]
        epochs, patience = 2, 1
    else:
        epochs, patience = EPOCHS, PATIENCE

    model = FM(dim, seed=args.seed)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), BATCH):
            batch = idx[i:i + BATCH]
            model.step(Xtr[batch], ytr[batch], y_wt_all[batch], censored_all[batch], LAMBDA_AUX)
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
