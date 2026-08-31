"""FM baseline + auxiliary is_click head sharing the second-order embeddings.

Hypothesis: is_click (train-only, corr 0.76 with long_view, denser positive rate)
supplies extra gradient to the shared item-side embeddings that participate in the
user x item interaction term -- the only channel that can move within-user ordering.
The main long_view head is what is scored; the aux head is discarded at inference.

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
LAM_AUX = 0.3


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


class MultiTaskFM:
    """Shared second-order embeddings V; two first-order/bias heads: main (long_view)
    and aux (is_click, train-only). Only the main head is used for scoring."""

    def __init__(self, dim, k=K, lr=LR, l2=L2, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.Wm = np.zeros(dim, dtype=np.float32)
        self.Wa = np.zeros(dim, dtype=np.float32)
        self.bm = np.float32(0.0)
        self.ba = np.float32(0.0)
        self.lr, self.l2 = lr, l2

        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mWm = np.zeros_like(self.Wm); self.vWm = np.zeros_like(self.Wm)
        self.mWa = np.zeros_like(self.Wa); self.vWa = np.zeros_like(self.Wa)
        self.t = 0

    def _shared(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return E, S, inter

    def logits_main(self, X):
        E, S, inter = self._shared(X)
        return self.bm + self.Wm[X].sum(1) + inter, E, S

    def step(self, X, y_main, y_aux):
        B = len(y_main)
        E, S, inter = self._shared(X)
        z_m = self.bm + self.Wm[X].sum(1) + inter
        z_a = self.ba + self.Wa[X].sum(1) + inter

        g_m = ((sigmoid(z_m) - y_main) / B).astype(np.float32)
        g_a = (LAM_AUX * (sigmoid(z_a) - y_aux) / B).astype(np.float32)
        g_tot = g_m + g_a

        gV = np.zeros_like(self.V)
        gWm = np.zeros_like(self.Wm)
        gWa = np.zeros_like(self.Wa)
        np.add.at(gWm, X, g_m[:, None])
        np.add.at(gWa, X, g_a[:, None])
        np.add.at(gV, X, g_tot[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gWm += self.l2 * self.Wm
        gWa += self.l2 * self.Wa

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in (
            (self.V, gV, self.mV, self.vV),
            (self.Wm, gWm, self.mWm, self.vWm),
            (self.Wa, gWa, self.mWa, self.vWa),
        ):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.bm -= self.lr * g_m.sum()
        self.ba -= self.lr * g_a.sum()

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits_main(X[i:i + bs])[0] for i in range(0, len(X), bs)])


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
    aux = api.aux_targets("train")
    atr = aux["is_click"].astype(np.float32)
    Xva, yva = enc["valid"], api.labels("valid")
    uva = api.features("valid")["user_id"]

    rng = np.random.default_rng(args.seed)
    if args.frac < 1.0:
        users = api.features("train")["user_id"]
        keep_users = rng.choice(
            np.unique(users), size=max(1, int(len(np.unique(users)) * args.frac)), replace=False
        )
        mask = np.isin(users, keep_users)
        Xtr, ytr, atr = Xtr[mask], ytr[mask], atr[mask]
        epochs, patience = 2, 1
    else:
        epochs, patience = EPOCHS, PATIENCE

    model = MultiTaskFM(dim, seed=args.seed)
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), BATCH):
            batch = idx[i:i + BATCH]
            model.step(Xtr[batch], ytr[batch], atr[batch])
        primary = gauc_ndcg(uva, yva, model.predict(Xva))
        if primary > best + 1e-5:
            best, bad = primary, 0
            best_state = (
                model.V.copy(), model.Wm.copy(), model.Wa.copy(),
                np.float32(model.bm), np.float32(model.ba),
            )
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.V, model.Wm, model.Wa, model.bm, model.ba = best_state

    np.save(args.out, model.predict(enc[args.split]).astype(np.float64))


if __name__ == "__main__":
    main()
