import numpy as np
import matplotlib.pyplot as plt

def plot_1d_paths(func, paths_dict, xlim=(-5, 5)):
    xs_curve = np.linspace(*xlim, 400)
    plt.plot(xs_curve, func(xs_curve), lw=2, label="$f(x)$")
    markers = ["o", "x", "^", "s", "D", "v"]
    for i, (lbl, path) in enumerate(paths_dict.items()):
        path = np.asarray(path).ravel()
        plt.plot(path, func(path), lw=1.2, marker=markers[i % len(markers)], label=lbl)
    plt.xlabel("x"); plt.ylabel("f(x)")
    plt.title("Trayectorias 1-D"); plt.legend(); plt.grid(alpha=.3); plt.show()

def plot_convergence(hist_dict, logy=True, ylabel="f"):
    for lbl, fvals in hist_dict.items():
        fvals = np.asarray(fvals)
        if logy:
            fvals = np.maximum(fvals, 1e-15)
        plt.plot(fvals, label=lbl)
    if logy:
        plt.yscale("log")
    plt.xlabel("Iter"); plt.ylabel(ylabel)
    plt.title("Convergencia"); plt.legend(); plt.grid(alpha=.3); plt.show()

def plot_2d_contour(func, paths_dict, span=((-2,2), (-2,2)), levels=25):
    gx = np.linspace(*span[0], 200)
    gy = np.linspace(*span[1], 200)
    X, Y = np.meshgrid(gx, gy)
    Z = func((X, Y))
    plt.contour(X, Y, Z, levels=levels, alpha=.6)
    markers = ["o", "x", "^", "s", "D", "v"]
    for i, (lbl, path) in enumerate(paths_dict.items()):
        P = np.asarray(path)
        plt.plot(P[:, 0], P[:, 1], marker=markers[i % len(markers)], ms=4, label=lbl)
    plt.legend(); plt.gca().set_aspect("equal")
    plt.title("Trayectorias en plano x–y"); plt.grid(alpha=.2); plt.show()

def lift_1d(f, df):
    """
    Adapta funciones 1D (escalares) a interfaz vectorial (n=1).
    Retorna: f_vec(X), grad_vec(X) con X shape (1,)
    """
    def f_vec(X):
        x = float(np.asarray(X).ravel()[0])
        return f(x)
    def grad_vec(X):
        x = float(np.asarray(X).ravel()[0])
        return np.array([df(x)], dtype=float)
    return f_vec, grad_vec