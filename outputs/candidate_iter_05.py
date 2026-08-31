"""
Baseline FM + a train-only auxiliary "watch fraction" regression head sharing the
same embedding table, to denoise the noisy long_view threshold via play_time_ms
(train-only aux target, never used as an inference feature).

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
LAM = 0.1          # weight of the auxiliary regression loss
GAMMA_INIT = 0.05  # initial scale of shared interaction term feeding the aux head
TARGET_CLIP = 5.0


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
    """Shared-embedding FM with a second, train-only auxiliary regression head.

    main_logit = b   + W [X].sum(1)  + inter
    aux_logit  = bt  + Wt[X].sum(1)  + gamma * inter

    `inter` (the second-order FM interaction) is shared between both heads, so
    gradients from the auxiliary watch-fraction target flow into the same
    embedding table V that the main ranking logit uses. Only main_logit is ever
    used for scoring / prediction.
    """

    def __init__(self, dim, k=K, lr=LR, l2=L2, lam=LAM, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.Wt = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.bt = np.float32(0.0)
        self.gamma = np.float32(GAMMA_INIT)
        self.lr, self.l2, self.lam = lr, l2, lam

        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mWt = np.zeros_like(self.Wt); self.vWt = np.zeros_like(self.Wt)
        self.t = 0

    def _core(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        main = self.b + self.W[X].sum(1) + inter
        return main, inter, E, S

    def logits(self, X):
        main, _, _, _ = self._core(X)
        return main

    def step(self, X, y, t_aux=None):
        B = len(y)
        main, inter, E, S = self._core(X)
        g_main = ((sigmoid(main) - y) / B).astype(np.float32)

        if t_aux is not None:
            aux = self.bt + self.Wt[X].sum(1) + self.gamma * inter
            g_aux = (self.lam * (aux - t_aux) / B).astype(np.float32)
        else:
            g_aux = np.zeros(B, dtype=np.float32)

        g_inter = g_main + self.gamma * g_aux

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        gWt = np.zeros_like(self.Wt)
        np.add.at(gW, X, g_main[:, None])
        np.add.at(gWt, X, g_aux[:, None])
        np.add.at(gV, X, g_inter[:, None, None] * (S[:, None, :] - E))
        ggamma = float(np.sum(g_aux * inter))

        gV += self.l2 * self.V
        gW += self.l2 * self.W
        gWt += self.l2 * self.Wt

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW),
            (self.Wt, gWt, self.mWt, self.vWt),
        ):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

        self.b -= self.lr * g_main.sum()
        self.bt -= self.lr * g_aux.sum()
        self.gamma -= self.lr * np.float32(ggamma)

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs]) for i in range(0, len(X), bs)])


def gauc_ndcg(users, labels, scores, k=5):
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

    # Train-only auxiliary target: clipped log watch-fraction surrogate.
    aux = api.aux_targets("train")
    play_ms = aux["play_time_ms"].astype(np.float64)
    dur_ms = api.features("train")["duration_ms"].astype(np.float64)
    ttr = np.log1p(np.clip(play_ms, 0, None)) - np.log1p(np.clip(dur_ms, 1, None))
    ttr = np.clip(ttr, -TARGET_CLIP, TARGET_CLIP).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    if args.frac < 1.0:
        users = api.features("train")["user_id"]
        keep_users = rng.choice(
            np.unique(users), size=max(1, int(len(np.unique(users)) * args.frac)), replace=False
        )
        mask = np.isin(users, keep_users)
        Xtr, ytr, ttr = Xtr[mask], ytr[mask], ttr[mask]
        epochs, patience = 2, 1
    else:
        epochs, patience = EPOCHS, PATIENCE

    model = FM(dim, seed=args.seed)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), BATCH):
            batch = idx[i:i + BATCH]
            model.step(Xtr[batch], ytr[batch], ttr[batch])
        primary = gauc_ndcg(uva, yva, model.predict(Xva))
        if primary > best + 1e-5:
            best, bad = primary, 0
            best_state = (
                model.V.copy(), model.W.copy(), np.float32(model.b),
            )
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.V, model.W, model.b = best_state

    np.save(args.out, model.predict(enc[args.split]).astype(np.float64))


if __name__ == "__main__":
    main()
