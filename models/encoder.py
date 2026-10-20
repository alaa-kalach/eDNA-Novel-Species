"""
Distance-matching encoder (pure NumPy, no PyTorch needed).

Idea
----
A small network maps each sequence's k-mer profile x to an embedding z.
It is trained on REFERENCE sequences only, so that the distance between two
embeddings matches a target distance between their species:

    loss = mean over pairs (||z_i - z_j|| - T[species_i, species_j])^2

The three methods differ ONLY in the target matrix T:
    tree      -> T = D_tree   (phylogenetic distance, main method)
    taxonomy  -> T = D_tax    (genus / family / order distance, baseline B6)
    labels    -> T = 0 if same species else 1   (labels only, baseline B5)

hidden > 0  -> one-hidden-layer ReLU network;  hidden = 0 -> linear map.

Two-goal training (sep_weight > 0): the loss above (goal 1: respect the tree)
plus  sep_weight * mean over different-species pairs of max(0, margin - d)^2
(goal 2: keep species apart), so close relatives are not blurred together.

Novelty score (as in the proposal): distance from a query to the nearest
known-species centroid, divided by that species' distance to its own nearest
known relative. Higher = more likely a new species.
"""
import numpy as np


def pairwise(Z):
    sq = (Z ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0) + 1e-12)


class DistanceEncoder:
    def __init__(self, in_dim, hidden=128, out_dim=32, lr=1e-3, epochs=400,
                 weight_decay=1e-4, seed=0, sep_weight=0.0, margin=0.0):
        rng = np.random.default_rng(seed)
        # two-goal training: sep_weight > 0 adds a species-separation term that
        # pushes sequences of DIFFERENT species at least `margin` apart
        self.sep_weight, self.margin = sep_weight, margin
        self.linear = hidden == 0
        if self.linear:
            self.W1 = rng.normal(0, np.sqrt(1 / in_dim), (in_dim, out_dim))
            self.b1 = np.zeros(out_dim)
            self.params = ["W1", "b1"]
        else:
            self.W1 = rng.normal(0, np.sqrt(2 / in_dim), (in_dim, hidden))
            self.b1 = np.zeros(hidden)
            self.W2 = rng.normal(0, np.sqrt(1 / hidden), (hidden, out_dim))
            self.b2 = np.zeros(out_dim)
            self.params = ["W1", "b1", "W2", "b2"]
        self.lr, self.epochs, self.wd = lr, epochs, weight_decay
        self.m = {p: 0 * getattr(self, p) for p in self.params}   # Adam state
        self.v = {p: 0 * getattr(self, p) for p in self.params}
        self.t = 0
        self.mu = self.sd = None
        self.history = []

    # ---------- forward ----------
    def _scale(self, X):
        return (X - self.mu) / self.sd

    def _forward(self, Xs):
        A = Xs @ self.W1 + self.b1
        if self.linear:
            return A, None, A
        H = np.maximum(A, 0)
        return A, H, H @ self.W2 + self.b2

    def embed(self, X):
        return self._forward(self._scale(X))[2]

    # ---------- loss + gradients ----------
    def _loss_and_grads(self, Xs, T):
        n = len(Xs)
        A, H, Z = self._forward(Xs)
        d = pairwise(Z)
        off = ~np.eye(n, dtype=bool)
        N = n * (n - 1)
        diff = (d - T) * off
        loss = (diff ** 2).sum() / N
        coef = 4 * diff
        if self.sep_weight > 0:
            # goal 2: different species (target > 0) must be >= margin apart
            short = np.maximum(self.margin - d, 0) * (T > 0) * off
            loss += self.sep_weight * (short ** 2).sum() / N
            coef = coef - 4 * self.sep_weight * short
        W = coef / (N * d)
        G = W.sum(1)[:, None] * Z - W @ Z          # d loss / d Z
        if self.linear:
            return loss, {"W1": Xs.T @ G, "b1": G.sum(0)}
        grads = {"W2": H.T @ G, "b2": G.sum(0)}
        dH = (G @ self.W2.T) * (A > 0)
        grads["W1"] = Xs.T @ dH
        grads["b1"] = dH.sum(0)
        return loss, grads

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p] + (self.wd * getattr(self, p) if p.startswith("W") else 0)
            self.m[p] = b1 * self.m[p] + (1 - b1) * g
            self.v[p] = b2 * self.v[p] + (1 - b2) * g ** 2
            mh = self.m[p] / (1 - b1 ** self.t)
            vh = self.v[p] / (1 - b2 ** self.t)
            setattr(self, p, getattr(self, p) - self.lr * mh / (np.sqrt(vh) + eps))

    def fit(self, X, T, verbose=False):
        """X: (n, in_dim) reference k-mer profiles; T: (n, n) target distances."""
        self.mu = X.mean(0)
        self.sd = X.std() + 1e-6      # one global scale (per-feature scaling blows up rare k-mers)
        Xs = self._scale(X)
        for ep in range(self.epochs):
            loss, grads = self._loss_and_grads(Xs, T)
            self._adam(grads)
            self.history.append(loss)
            if verbose and (ep % 100 == 0 or ep == self.epochs - 1):
                print(f"  epoch {ep:4d}  loss {loss:.4f}")
        return self

    def fit_minibatch(self, X, species_idx, T_species, steps=4000, batch=512,
                      seed=0, X_val=None, val_idx=None, eval_every=500, verbose=True):
        """
        Minibatch training for large data (e.g. global pretraining).
        X: (n, in_dim) k-mer profiles; species_idx: (n,) row of each sequence in
        T_species (species x species target distances). Each step samples `batch`
        sequences and matches all pairwise distances within the batch.
        If X_val/val_idx are given, prints the Spearman correlation between
        embedding distance and target distance on held-out data.
        """
        from scipy.stats import spearmanr
        rng = np.random.default_rng(seed)
        self.mu = X.mean(0)
        self.sd = X.std() + 1e-6
        Xs = self._scale(X)
        species_idx = np.asarray(species_idx)
        n = len(X)
        for step in range(1, steps + 1):
            b = rng.choice(n, size=min(batch, n), replace=False)
            s = species_idx[b]
            loss, grads = self._loss_and_grads(Xs[b], T_species[np.ix_(s, s)])
            self._adam(grads)
            self.history.append(loss)
            if verbose and (step % eval_every == 0 or step == 1):
                msg = f"  step {step:5d}  train loss {loss:.4f}"
                if X_val is not None:
                    Z = self.embed(X_val)
                    iu = np.triu_indices(len(Z), 1)
                    T = T_species[np.ix_(val_idx, val_idx)][iu]
                    keep = T > 0
                    msg += f"  | held-out Spearman(emb, target) {spearmanr(pairwise(Z)[iu][keep], T[keep])[0]:.3f}"
                print(msg, flush=True)
        return self


def target_matrix(species_idx, method, D_tree, D_tax):
    """Pairwise target distances between reference sequences."""
    s = np.asarray(species_idx)
    if method == "tree":
        return D_tree[np.ix_(s, s)]
    if method == "taxonomy":
        return D_tax[np.ix_(s, s)]
    if method == "labels":
        return (s[:, None] != s[None, :]).astype(float)
    raise ValueError(method)


def novelty_scores(Z_ref, ref_species, Z_test, scaled=True):
    """
    Returns (novelty_score, predicted_species) for each test embedding.
    scaled=True : distance to nearest centroid / that species' distance to its
                  nearest other centroid (the proposal's score).
    scaled=False: plain distance to the nearest centroid.
    """
    ref_species = np.asarray(ref_species)
    sp = np.unique(ref_species)
    C = np.stack([Z_ref[ref_species == s].mean(0) for s in sp])
    d2 = (Z_test ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * Z_test @ C.T
    d = np.sqrt(np.maximum(d2, 0))          # memory-efficient Euclidean distances
    j = d.argmin(1)
    score = d[np.arange(len(Z_test)), j]
    if scaled:
        CC = pairwise(C)
        np.fill_diagonal(CC, np.inf)
        score = score / (CC.min(1)[j] + 1e-9)
    return score, sp[j]
