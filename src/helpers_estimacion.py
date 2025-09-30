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

'''
def ajuste_manual(x, y):    
    m_slider = widgets.FloatSlider(value=1, min=-10, max=10, step=0.1, description="Pendiente")
    b_slider = widgets.FloatSlider(value=0, min=-10, max=10, step=0.5, description="Intercepto")

    def update(m, b):
        plt.figure(figsize=(6,4))
        plt.scatter(x, y, alpha=0.6)
        plt.plot(x, m*x + b, color="red")
        plt.xlabel("x"); plt.ylabel("y")
        plt.show()

    widgets.interact(update, m=m_slider, b=b_slider)
'''

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

    m_slider = W.FloatSlider(value=m_init, min=-10, max=10, step=0.1, description="Pendiente")
    b_slider = W.FloatSlider(value=b_init, min=-20, max=20, step=0.5, description="Intercepto")

    out = W.Output()

    def _plot(m, b):
        with out:
            out.clear_output(wait=True)

            # Predicción y residuos
            y_hat = m * x + b
            resid = y - y_hat
            rmse = np.sqrt(np.mean(resid**2))

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
            ax.set_title(f"Ajuste manual • RMSE = {rmse:.3f}")
            ax.legend(loc="best")
            ax.grid(True, linewidth=0.3, alpha=0.5)
            plt.show()

    W.interact(_plot, m=m_slider, b=b_slider)
    display(out)


def fit_ols_univar(x, y):
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


def plot_ols_with_residuals(x, y, m, b, title="", xlabel="x", ylabel="y",
                            sample_for_scatter=150, resid_alpha=0.25, point_alpha=0.5):
    """
    Scatter de (x,y), recta y=m x + b, y líneas punteadas de residuo (verticales).
    sample_for_scatter: para datasets grandes, submuestrea puntos (solo para el scatter).
    """
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


