import numpy as np
import pandas as pd
import ipywidgets as W
import matplotlib.pyplot as plt


def ejemplo_ilustrativo():
	np.random.seed(42)

	# Datos sintéticos
	a_true, b_true = 8, 4
	n = 80
	x = np.random.uniform(0, 20, n)
	noise = np.random.normal(0, 3, n)

	# Creación X y Y
	y = (a_true + b_true * x) + noise
	X = np.vstack([np.ones(n), x]).T

	DF = pd.DataFrame({"Años de experiencia (x)":[round(i,1) for i in x], "Salario normalizado (y)":[round(i,1) for i in y]})
	true_coefs = (a_true, b_true)

	return DF, true_coefs

def ajuste_manual(x, y, m_init=1.0, b_init=0.0,
                  show_residuals=True, punto_alpha=0.6, resid_alpha=0.35):
    """
    Interfaz simple para 'ajuste manual' con sliders (pendiente/intercepto)
    y visualización de residuos verticales punteados.

    Parámetros:
      x, y: arrays 1D (mismo largo)
      m_init, b_init: valores iniciales
      show_residuals: si True, dibuja las líneas punteadas (y - y_hat) por punto
    """
    

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    b_slider = W.FloatSlider(value=b_init, min=-20, max=20, step=0.5, description="a (intercepto)")
    m_slider = W.FloatSlider(value=m_init, min=-10, max=10, step=0.1, description="b (pendiente)")

    out = W.Output()

    def _plot(m, b):
        with out:
            out.clear_output(wait=True)

            # Predicción y residuos
            y_hat = m * x + b
            resid = y - y_hat
            #rmse = np.sqrt(np.mean(resid**2))
            sse = sum(resid**2)

            # Ordenar para la línea (más bonito)
            order = np.argsort(x)
            x_line = x[order]
            y_line = (m * x_line + b)

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.scatter(x, y, alpha=punto_alpha, label="Datos")

            # Línea de regresión manual
            ax.plot(x_line, y_line, label=f"Recta: y = {m:.2f}x + {b:.2f}", linewidth=2)

            # Líneas punteadas de residuo
            if show_residuals:
                for xi, yi, yhi in zip(x, y, m*x + b):
                    ax.plot([xi, xi], [yi, yhi], linestyle=":", alpha=resid_alpha)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Ajuste manual • (Suma Errores = {sse:.3f})")
            ax.legend(loc="best")
            ax.grid(True, linewidth=0.3, alpha=0.5)
            plt.show()

    W.interact(_plot, m=m_slider, b=b_slider)
    display(out)


def fit_ols(x, y):
    """
    Ajusta OLS univariado: y = m x + b, vía mínimos cuadrados (lstsq).
    Devuelve: m, b, y una función predictora f(X).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    X1 = np.c_[np.ones(len(x)), x]
    theta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    b, m = float(theta[0]), float(theta[1])
    return m, b, (lambda X: m*np.asarray(X).ravel() + b)

def metrics_ols(y_true, y_pred):
    """
    Métricas básicas: R2, RMSE, MAE.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res/ss_tot
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def plot_ols(x, y, title="", xlabel="x", ylabel="y",
                            sample_for_scatter=150, resid_alpha=0.25, point_alpha=0.5):
    """
    Scatter de (x,y), recta y=m x + b, y líneas punteadas de residuo (verticales).
    sample_for_scatter: para datasets grandes, submuestrea puntos (solo para el scatter).
    """
    m, b, f = fit_ols(x, y)      # m: pendiente; b: intercepto; f: función/modelo de estimación (b+m*x)

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Submuestreo visual (no afecta ajuste)
    if (sample_for_scatter is not None) and (len(x) > sample_for_scatter):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), size=sample_for_scatter, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y

    order = np.argsort(x)
    x_line = x[order]
    y_line = m*x_line + b

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(xs, ys, alpha=point_alpha, label="Datos")

    # Recta
    ax.plot(x_line, y_line, linewidth=2, label=f"OLS: y = {m:.3f} x + {b:.3f}")

    # Residuos (sobre el subconjunto del scatter para legibilidad)
    for xi, yi in zip(xs, ys):
        yhat_i = m*xi + b
        ax.plot([xi, xi], [yi, yhat_i], linestyle=":", alpha=resid_alpha)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# Actualización 2026: funciones para el módulo de Optimización → ML
# ─────────────────────────────────────────────────────────────────────

def plot_loss_surface(x, y, a_range=(-10, 25), b_range=(-2, 10),
                      path=None, title="Superficie de pérdida (MSE)"):
    """
    Visualiza la superficie de pérdida MSE L(a,b) = (1/n) sum (y - (a + b*x))^2
    como gráfico 3D + contorno, con trayectoria GD opcional.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    A = np.linspace(*a_range, 100)
    B = np.linspace(*b_range, 100)
    AA, BB = np.meshgrid(A, B)

    ZZ = np.zeros_like(AA)
    for i in range(len(A)):
        for j in range(len(B)):
            pred = AA[j, i] + BB[j, i] * x
            ZZ[j, i] = np.mean((y - pred) ** 2)

    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(AA, BB, ZZ, alpha=0.6, cmap='viridis')
    if path is not None:
        path = np.asarray(path)
        L_path = [np.mean((y - (a + b * x)) ** 2) for a, b in path]
        ax1.plot(path[:, 0], path[:, 1], L_path, 'r.-', linewidth=2, markersize=5, label='GD')
        ax1.legend()
    ax1.set_xlabel('a (intercepto)')
    ax1.set_ylabel('b (pendiente)')
    ax1.set_zlabel('MSE')
    ax1.set_title(title)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contour(AA, BB, ZZ, levels=30, alpha=0.7, cmap='viridis')
    if path is not None:
        ax2.plot(path[:, 0], path[:, 1], 'r.-', linewidth=1.5, markersize=5, label='GD')
        ax2.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Inicio')
        ax2.plot(path[-1, 0], path[-1, 1], 'rs', markersize=10, label='Final')
        ax2.legend(fontsize=8)
    ax2.set_xlabel('a (intercepto)')
    ax2.set_ylabel('b (pendiente)')
    ax2.set_title('Curvas de nivel')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def sigmoid(z):
    """Función sigmoide: sigma(z) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_loss(y, y_hat, eps=1e-15):
    """Pérdida logística (cross-entropy binaria)."""
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def gd_logistic(X, y, lr=0.1, epochs=200):
    """
    Descenso de gradiente para regresión logística.
    X : (n, d) con columna de 1s; y : (n,) etiquetas {0,1}.
    """
    n, d = X.shape
    theta = np.zeros(d)
    losses = []
    thetas = [theta.copy()]
    for _ in range(epochs):
        z = X @ theta
        p = sigmoid(z)
        grad = (1 / n) * X.T @ (p - y)
        theta = theta - lr * grad
        losses.append(logistic_loss(y, sigmoid(X @ theta)))
        thetas.append(theta.copy())
    return theta, losses, thetas


def plot_decision_boundary(X, y, theta, title="Frontera de decision"):
    """Dibuja scatter 2D con frontera de decision lineal. X: [1, x1, x2]."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[y == 0, 1], X[y == 0, 2], label='Clase 0', alpha=0.6, edgecolors='k')
    ax.scatter(X[y == 1, 1], X[y == 1, 2], label='Clase 1', alpha=0.6, edgecolors='k')
    if abs(theta[2]) > 1e-10:
        x1_range = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)
        x2_boundary = -(theta[0] + theta[1] * x1_range) / theta[2]
        ax.plot(x1_range, x2_boundary, 'k--', linewidth=2, label='Frontera')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
