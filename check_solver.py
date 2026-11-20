"""
Sanity checks for the implementation (run: python check_solver.py).

1. The sparse-ELM dual solver (paper's Algorithm 1) reaches the same optimum
   as a generic box-constrained QP solver (SciPy L-BFGS-B).
2. Its solution satisfies the KKT conditions (Eqs. 35-37) within eps.
3. Unified ELM's closed form equals the ridge-regression solution in feature space.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler

from elm import FeatureMap, UnifiedELM, gaussian_kernel, solve_sparse_elm_dual

X, y = load_wine(return_X_y=True)
X = MinMaxScaler((-1, 1)).fit_transform(X)
mask = y < 2
X, t = X[mask], np.where(y[mask] == 0, 1.0, -1.0)

ok = True
for C in (0.1, 1.0, 10.0):
    G = np.outer(t, t) * gaussian_kernel(X, X, sigma=1.0)
    a, iters = solve_sparse_elm_dual(G, C, eps=1e-6)
    obj = lambda v: 0.5 * v @ G @ v - v.sum()
    ref = minimize(obj, np.zeros(len(t)), jac=lambda v: G @ v - 1, method="L-BFGS-B",
                   bounds=[(0, C)] * len(t), options=dict(ftol=1e-15, gtol=1e-10, maxiter=10000))
    g = G @ a - 1
    kkt = np.max(np.where(a <= 0, np.maximum(-g, 0), np.where(a >= C, np.maximum(g, 0), np.abs(g))))
    good = abs(obj(a) - ref.fun) < 1e-6 * max(1, abs(ref.fun)) and kkt < 1e-5
    ok &= good
    print(f"C={C:<5} Algorithm-1 obj={obj(a):.8f}  L-BFGS-B obj={ref.fun:.8f}  "
          f"max KKT violation={kkt:.1e}  iters={iters}  #SV={int((a > 0).sum())}  "
          f"{'OK' if good else 'MISMATCH'}")

# Unified ELM with random nodes: kernel form == primal ridge form
rng = np.random.default_rng(0)
m = UnifiedELM("sigmoid", C=5.0, n_hidden=50, rng=rng).fit(X, (t > 0).astype(int))
H = m.fmap.hl.transform(X)
T = np.where(np.stack([t < 0, t > 0], 1), 1.0, -1.0)
beta_primal = np.linalg.solve(np.eye(50) / 5.0 + H.T @ H, H.T @ T)
diff = np.abs(H @ beta_primal - m.decision_function(X)).max()
ok &= diff < 1e-8
print(f"Unified ELM: |dual - primal| = {diff:.1e}  {'OK' if diff < 1e-8 else 'MISMATCH'}")
print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
