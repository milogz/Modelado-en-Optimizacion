
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math, HTML

def mat_to_bmatrix(M, fmt="{:g}"):
    import numpy as np
    rows = [" & ".join(fmt.format(v) for v in row) for row in np.array(M)]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"

def vec_to_bmatrix(v, fmt="{:g}"):
    import numpy as np
    v = np.array(v).reshape(-1,1)
    return mat_to_bmatrix(v, fmt=fmt)

def _intersect_lines(a1, b1, a2, b2, tol=1e-9):
    import numpy as np
    A = np.vstack([a1, a2])
    if abs(np.linalg.det(A)) < tol:
        return None
    return np.linalg.solve(A, np.array([b1, b2], dtype=float))

def feasible_polygon(A, b, add_axes=True, tol=1e-9):
    import numpy as np
    m, n = A.shape
    assert n == 2
    lines = [(A[i], b[i]) for i in range(m)]
    if add_axes:
        e1 = np.array([1.0, 0.0]); e2 = np.array([0.0, 1.0])
        lines += [(e1, 0.0), (e2, 0.0)]
    pts = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            p = _intersect_lines(lines[i][0], lines[i][1], lines[j][0], lines[j][1])
            if p is None: 
                continue
            if add_axes and (p[0] < -1e-9 or p[1] < -1e-9):
                continue
            if np.all(A @ p <= b + tol):
                pts.append(p)
    if not pts:
        return np.zeros((0,2))
    P = np.unique(np.round(np.array(pts, dtype=float), 10), axis=0)
    P = P[np.lexsort((P[:,1], P[:,0]))]
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    return hull

def plot_method_graph(A, b, x=None, colors=None, ax=None, title="Región factible", show_fill=True):
    import numpy as np, matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))
    m, n = A.shape
    assert n == 2
    hull = feasible_polygon(A, b, add_axes=True)
    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"][:m]
    if hull.size:
        xmax = max(1.0, np.max(hull[:,0]) + 1.0)
        ymax = max(1.0, np.max(hull[:,1]) + 1.0)
    else:
        xmax, ymax = 10, 10
    xs = np.linspace(0, xmax, 200)
    for i in range(m):
        a = A[i]; bi = b[i]
        if abs(a[1]) > 1e-12:
            ys = (bi - a[0]*xs)/a[1]
            ax.plot(xs, ys, color=colors[i], linewidth=2)
        else:
            xline = bi/a[0] if abs(a[0])>1e-12 else np.nan
            ax.plot([xline, xline], [0, ymax], color=colors[i], linewidth=2)
    if hull.size and show_fill:
        ax.fill(hull[:,0], hull[:,1], alpha=0.15, hatch="///", edgecolor="none")
    if x is not None:
        ax.scatter([x[0]], [x[1]], s=80, color="red", zorder=5)
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_title(title)
    return ax, colors, hull

def slacks(A, b, x):
    return b - A @ x

def bar_slacks(s, colors=None, ax=None, names=None):
    m = len(s)
    y = np.arange(m)
    ax.barh(y, s, color=colors)
    labels = names
    ax.set_yticks(y, labels)
    ax.axvline(0, color="k", linewidth=1)
    ax.set_xlabel("valor")
    ax.set_title("Variables")    
    ax.set_xlim(-max(s), max(s))    
    return ax

def basis_info(A_full, b, c, Bcols):
    import numpy as np
    m, n = A_full.shape
    B = A_full[:, Bcols]
    Ncols = [j for j in range(n) if j not in Bcols]
    N = A_full[:, Ncols]
    x = np.zeros(n); xB = np.linalg.solve(B, b.astype(float))
    x[Bcols] = xB
    cB = c[Bcols]; cN = c[Ncols]
    y = np.linalg.solve(B.T, cB)
    r = cN - N.T @ y
    return {"B":B, "N":N, "Bcols":Bcols, "Ncols":Ncols, "x":x, "xB":xB, "cB":cB, "cN":cN, "y":y, "r":r}

def step_info(A_full, x, Bcols, enter_j):
    import numpy as np
    B = A_full[:, Bcols]; a_j = A_full[:, enter_j]
    d = np.zeros_like(x); dB = -np.linalg.solve(B, a_j)
    d[enter_j] = 1.0
    for k, j in enumerate(Bcols):
        d[j] = dB[k]
    mask = dB < 0
    if not np.any(mask):
        theta = np.inf; leave = None
    else:
        xB = x[Bcols]; ratios = xB[mask] / (-dB[mask])
        idx = int(np.argmin(ratios)); theta = float(np.min(ratios))
        leave = Bcols[np.where(mask)[0][idx]]
    x_new = x + theta * d if np.isfinite(theta) else x.copy()
    return {"d":d, "dB":dB, "theta":theta, "leave":leave, "x_new":x_new}
