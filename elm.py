"""
Implementation of the three classifiers compared in

    Z. Bai, G.-B. Huang, D. Wang, H. Wang, M. B. Westover,
    "Sparse Extreme Learning Machine for Classification",
    IEEE Transactions on Cybernetics, 44(10):1858-1870, 2014.
    doi:10.1109/TCYB.2014.2298235  (open access: PMC4883115)

* UnifiedELM  - Huang et al. (2012) regularised ELM, closed form, multi-output.
* SparseELM   - the paper's contribution: hinge-loss ELM *without bias*, whose
                dual has only box constraints 0 <= alpha_i <= C and is solved by
                the paper's single-variable iterative algorithm (Algorithm 1).
                Multiclass problems use one-against-one (OAO) voting.
* SVM         - baseline; we use scikit-learn's SVC (LIBSVM, OAO).

Feature mappings ("ELM kernels") supported, exactly as in the paper:
    gaussian   K(u,v) = exp(-||u-v||^2 / (2 sigma^2))
    poly       K(u,v) = (u.v + 1)^m
    sigmoid    h(x)   = 1 / (1 + exp(-(a.x + b)))   (L random hidden nodes)
    sinusoid   h(x)   = sin(a.x + b)                (L random hidden nodes)

Only NumPy is needed for the ELM models.
"""
from itertools import combinations

import numpy as np


# --------------------------------------------------------------------------
# Feature mappings / kernels
# --------------------------------------------------------------------------
def gaussian_kernel(A, B, sigma):
    sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    np.maximum(sq, 0.0, out=sq)
    return np.exp(-sq / (2.0 * sigma ** 2))


def poly_kernel(A, B, m):
    return (A @ B.T + 1.0) ** int(m)


class RandomHiddenLayer:
    """Random additive hidden nodes G(a, b, x).

    The paper only says a_i, b_i are 'randomly generated'. We follow the
    reference ELM MATLAB code of Huang's group: a ~ U[-1, 1], b ~ U[0, 1].
    """

    def __init__(self, n_hidden, activation, rng):
        self.L = int(n_hidden)
        self.activation = activation
        self.rng = rng

    def fit(self, X):
        d = X.shape[1]
        self.a = self.rng.uniform(-1.0, 1.0, size=(d, self.L))
        self.b = self.rng.uniform(0.0, 1.0, size=(1, self.L))
        return self

    def transform(self, X):
        Z = X @ self.a + self.b
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-Z))
        if self.activation == "sinusoid":
            return np.sin(Z)
        raise ValueError(self.activation)


class FeatureMap:
    """Uniform interface: gram(X1, X2) = Omega_ELM(X1, X2) = h(X1) h(X2)^T."""

    def __init__(self, kind, param=None, n_hidden=1000, rng=None):
        self.kind = kind
        self.param = param            # sigma (gaussian) or m (poly)
        self.n_hidden = n_hidden
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(self, X):
        if self.kind in ("sigmoid", "sinusoid"):
            self.hl = RandomHiddenLayer(self.n_hidden, self.kind, self.rng).fit(X)
        return self

    def gram(self, A, B):
        if self.kind == "gaussian":
            return gaussian_kernel(A, B, self.param)
        if self.kind == "poly":
            return poly_kernel(A, B, self.param)
        return self.hl.transform(A) @ self.hl.transform(B).T


# --------------------------------------------------------------------------
# Unified ELM (Huang et al. 2012), Eq. (11) of Bai et al. 2014
#   f(x) = [K(x,x_1) ... K(x,x_N)] (I/C + Omega)^-1 T
# Multiclass is solved directly with one output per class, targets in {-1,+1}.
# --------------------------------------------------------------------------
class UnifiedELM:
    def __init__(self, kind, C, param=None, n_hidden=1000, rng=None):
        self.C = float(C)
        self.fmap = FeatureMap(kind, param, n_hidden, rng)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        T = -np.ones((len(y), len(self.classes_)))
        T[np.arange(len(y)), np.searchsorted(self.classes_, y)] = 1.0
        self.fmap.fit(X)
        self.X_ = X
        Omega = self.fmap.gram(X, X)
        A = Omega + np.eye(len(y)) / self.C
        self.beta_ = np.linalg.solve(A, T)
        self.n_support_ = len(y)          # dense solution: every sample is used
        return self

    def decision_function(self, X):
        return self.fmap.gram(X, self.X_) @ self.beta_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


# --------------------------------------------------------------------------
# Sparse ELM (the paper's method)
# --------------------------------------------------------------------------
def solve_sparse_elm_dual(G, C, eps=1e-3, max_iter=1_000_000):
    """Algorithm 1 of Bai et al. (2014).

    min_a 0.5 a^T G a - 1^T a   s.t. 0 <= a_i <= C,  G_ij = t_i t_j Omega_ij

    One variable is updated per iteration (no equality constraint, so no SMO
    pairs are needed).  Working-set selection (Eqs. 41-42):
        d_i = 1 if a_i = 0;  -1 if a_i = C;  -sign(g_i) otherwise
        J_i = g_i d_i,  c = argmin_i J_i
    Update (Eqs. 39-40): a_c <- clip(a_c - g_c / G_cc, 0, C)
    Stop when min_i J_i > -eps (KKT conditions hold within eps).
    """
    n = G.shape[0]
    alpha = np.zeros(n)
    g = -np.ones(n)                      # gradient  g = G a - 1
    diag = np.diag(G).copy()
    diag[diag <= 1e-12] = 1e-12
    it = 0
    for it in range(max_iter):
        J = np.where(alpha <= 0.0, g, np.where(alpha >= C, -g, -np.abs(g)))
        c = int(np.argmin(J))
        if J[c] > -eps:
            break
        new = min(max(alpha[c] - g[c] / diag[c], 0.0), C)
        delta = new - alpha[c]
        if delta != 0.0:
            alpha[c] = new
            g += delta * G[:, c]
    return alpha, it + 1


class _BinarySparseELM:
    def __init__(self, fmap, C, eps):
        self.fmap, self.C, self.eps = fmap, float(C), eps

    def fit(self, X, t):
        Omega = self.fmap.gram(X, X)
        G = (t[:, None] * t[None, :]) * Omega
        alpha, self.n_iter_ = solve_sparse_elm_dual(G, self.C, self.eps)
        sv = alpha > 0
        self.sv_X = X[sv]
        self.sv_coef = (alpha * t)[sv]
        self.n_support_ = int(sv.sum())
        return self

    def decision_function(self, X):
        if self.n_support_ == 0:
            return np.zeros(len(X))
        return self.fmap.gram(X, self.sv_X) @ self.sv_coef   # Eq. (19), no bias


class SparseELM:
    """Sparse ELM with one-against-one (OAO) voting for multiclass problems.

    n_support_ is summed over the OAO sub-problems (this is how the paper's
    Table VIII can report e.g. 149 SVs on Wine with only 89 training samples).
    """

    def __init__(self, kind, C, param=None, n_hidden=1000, eps=1e-3, rng=None):
        self.C, self.eps = float(C), eps
        self.fmap = FeatureMap(kind, param, n_hidden, rng)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fmap.fit(X)   # one random hidden layer shared by all sub-problems
        self.models_ = []
        for i, j in combinations(range(len(self.classes_)), 2):
            mask = (y == self.classes_[i]) | (y == self.classes_[j])
            t = np.where(y[mask] == self.classes_[i], 1.0, -1.0)
            m = _BinarySparseELM(self.fmap, self.C, self.eps).fit(X[mask], t)
            self.models_.append((i, j, m))
        self.n_support_ = sum(m.n_support_ for _, _, m in self.models_)
        return self

    def predict(self, X):
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        rows = np.arange(len(X))
        for i, j, m in self.models_:
            pos = m.decision_function(X) > 0
            votes[rows[pos], i] += 1
            votes[rows[~pos], j] += 1
        return self.classes_[np.argmax(votes, axis=1)]   # ties -> lower class index


# --------------------------------------------------------------------------
# SVM baseline (LIBSVM through scikit-learn, OAO multiclass like the paper)
# --------------------------------------------------------------------------
def make_svm(kind, C, param):
    from sklearn.svm import SVC
    if kind == "gaussian":
        return SVC(C=C, kernel="rbf", gamma=1.0 / (2.0 * param ** 2), tol=1e-3)
    if kind == "poly":
        return SVC(C=C, kernel="poly", degree=int(param), gamma=1.0, coef0=1.0, tol=1e-3)
    raise ValueError("SVM is only run with gaussian / poly kernels (as in the paper)")
