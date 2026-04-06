import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display

def init_busqueda():
  import numpy as np
  import matplotlib.pyplot as plt
  plt.rcParams["figure.figsize"] = (6, 4)
  np.set_printoptions(precision=4, suppress=True)
  return np, plt

from IPython.display import Image, display
from pathlib import Path
import os

def show_fig(nombre, subruta="assets/figs"):
    """
    Busca y muestra una imagen de forma robusta para Jupyter y Colab.
    - nombre: nombre del archivo
    - subruta: ruta relativa esperada dentro del proyecto
    """
    # 1️⃣ posibles ubicaciones
    posibles_rutas = [
        Path(nombre),  # mismo directorio
        Path.cwd() / nombre,  # cwd directo
        Path.cwd() / subruta / nombre,  # estructura local tipo repo
        Path("/content") / nombre,  # raíz de colab
        Path("/content") / subruta / nombre,  # colab con estructura repo
    ]

    # 2️⃣ buscar primera ruta existente
    ruta_final = next((p for p in posibles_rutas if p.exists()), None)

    # 3️⃣ mostrar resultado
    if ruta_final:
        #print(f"✅ Imagen encontrada en: {ruta_final}")
        display(Image(filename=str(ruta_final)))
    else:
        print(f"⚠️ No se encontró {nombre} en las rutas esperadas: {posibles_rutas}")


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

def tabla_comp():    

    html_table = """
    <table style="border-collapse: collapse; width: 100%; text-align: center;">
      <tr style="background-color: #f0f0f0;">
        <th></th>
        <th>No lineal sin restricciones</th>
        <th>Lineal con restricciones (LP)</th>
        <th>No lineal con restricciones (NLP)</th>
        <th>Entero lineal (MIP)</th>
      </tr>
      <tr>
        <td style="font-weight: bold;">Forma general</td>
        <td>\\(\\min f_{\\text{nonlin}}(x), \\; x \\in \\mathbb{R}^n\\)</td>
        <td>\\(\\min f(x) = c^T x, \\; A x \\le b, \\; x \\in \\mathbb{R}^n\\)</td>
        <td>\\(\\min f_{\\text{nl con restr}}(x), \\; g(x) \\le b, \\; x \\in \\mathbb{R}^n\\)</td>
        <td>\\(\\min f(x) = c^T x, \\; A x \\le b, \\; x \\in \\mathbb{Z}^n\\)</td>
      </tr>
      <tr>
        <td style="font-weight: bold;">Ejemplos de aplicación</td>
        <td>Ajuste de parámetros sin restricciones (estadística, <i>machine learning</i>, métodos numéricos)</td>
        <td>Mezclas, asignación, planeación de producción, manejo de inventarios</td>
        <td>Diseño de ingeniería con procesos físicos, portafolios con riesgo no lineal</td>
        <td><i>Scheduling</i>, ruteo, asignación entera</td>
      </tr>
      <tr>
        <td style="font-weight: bold;">Métodos típicos de solución</td>
        <td>Gradiente y sus variantes, heurísticas y aproximaciones numéricas</td>
        <td>Simplex, Punto Interior</td>
        <td>Programación cuadrática o convexa (<i>via solvers</i>), métodos numéricos</td>
        <td>Branch & Bound (<i>via solvers</i>)</td>
      </tr>
    </table>
    """
    display(HTML(html_table))


# =============================================================
# Algoritmos centrales de busqueda (movidos desde notebook 3-1)
# =============================================================

def vecindario_circular(x0, d=.2, k=8):
    """
    Genera 'k' puntos vecinos distribuidos en un circulo
    de radio 'd' alrededor de 'x0'.
    Retorna: Lista de puntos vecinos.
    """
    angulos = np.linspace(0, 2*np.pi, k, endpoint=False)
    vecinos = [x0 + d * np.array([np.cos(theta), np.sin(theta)]) for theta in angulos]
    return vecinos


def vecindario_lineal(x, d=0.5):
    """Version simplificada para ejemplo 1D: 'vecinos' en una linea."""
    x = float(np.asarray(x).ravel()[0])
    return np.array([[x - d], [x + d]], dtype=float)


def busqueda_local(x0, f, generador_vecinos, iters=50):
    """
    Evalua vecinos de x0 y elige movimiento que mejore f(x)
    hasta que no se halle mejora o se alcanzan 'iters' iteraciones.
    """
    x = x0.copy()
    hist = [x.copy()]
    for _ in range(iters):
        vecinos = generador_vecinos(x)
        objs = np.array([f(v) for v in vecinos])
        idx = np.argmin(objs)
        x_new = vecinos[idx]
        if (f(x_new) - f(x)) >= 0:
            break
        x = x_new
        hist.append(x.copy())
    return np.array(hist)


def descenso_gradiente(x0, grad, alpha=0.05, iters=50, tol=1e-4):
    """
    Aplica movimientos de longitud 'alpha' en direccion del gradiente
    desde x0, con parada por norma del gradiente (tol) o max iteraciones (iters).
    """
    x = x0.copy()
    hist = [x.copy()]
    for _ in range(iters):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - alpha * g
        hist.append(x.copy())
    return np.array(hist)


# =============================================================
# Catálogo de funciones de prueba 2-D  (para el dashboard)
# =============================================================

def _cuadratica(xy):
    """f(x,y) = x^2 + 2y^2  –  convexa, mínimo en (0,0)."""
    x, y = np.asarray(xy[0]), np.asarray(xy[1])
    return x**2 + 2*y**2

def _grad_cuadratica(xy):
    xy = np.asarray(xy, dtype=float).ravel()
    return np.array([2*xy[0], 4*xy[1]])


def _rosenbrock(xy):
    """f(x,y) = (1-x)^2 + 100(y-x^2)^2  –  difícil, mínimo en (1,1)."""
    x, y = np.asarray(xy[0]), np.asarray(xy[1])
    return (1 - x)**2 + 100*(y - x**2)**2

def _grad_rosenbrock(xy):
    xy = np.asarray(xy, dtype=float).ravel()
    x, y = xy[0], xy[1]
    return np.array([-2*(1 - x) - 400*x*(y - x**2),
                      200*(y - x**2)])


def _rastrigin(xy):
    """f(x,y) = 20 + x^2 + y^2 - 10(cos 2πx + cos 2πy)  –  multimodal."""
    x, y = np.asarray(xy[0]), np.asarray(xy[1])
    return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

def _grad_rastrigin(xy):
    xy = np.asarray(xy, dtype=float).ravel()
    x, y = xy[0], xy[1]
    return np.array([2*x + 20*np.pi*np.sin(2*np.pi*x),
                     2*y + 20*np.pi*np.sin(2*np.pi*y)])


def _saddle(xy):
    """f(x,y) = x^2 - y^2  –  punto silla en (0,0)."""
    x, y = np.asarray(xy[0]), np.asarray(xy[1])
    return x**2 - y**2

def _grad_saddle(xy):
    xy = np.asarray(xy, dtype=float).ravel()
    return np.array([2*xy[0], -2*xy[1]])


FUNCIONES_PRUEBA = {
    "Cuadrática  f = x² + 2y²": {
        "f": _cuadratica, "grad": _grad_cuadratica,
        "span": ((-4, 4), (-4, 4)), "levels": 30,
        "x0_default": [3.0, 3.0],
        "desc": "Convexa simple. El mínimo global está en (0, 0). "
                "Útil para comparar convergencia: gradiente llega rápido, "
                "búsqueda local depende de d y k."
    },
    "Rosenbrock  f = (1−x)² + 100(y−x²)²": {
        "f": _rosenbrock, "grad": _grad_rosenbrock,
        "span": ((-2, 2), (-1, 3)), "levels": np.logspace(-1, 3.5, 35),
        "x0_default": [-1.0, 2.0],
        "desc": "El 'banana': mínimo en (1, 1) dentro de un valle estrecho y curvado. "
                "Gradiente requiere α muy pequeño; búsqueda local puede "
                "quedar atrapada si d es grande."
    },
    "Rastrigin  f = 20 + Σ(xᵢ² − 10 cos 2πxᵢ)": {
        "f": _rastrigin, "grad": _grad_rastrigin,
        "span": ((-3, 3), (-3, 3)), "levels": 30,
        "x0_default": [2.5, 2.5],
        "desc": "Multimodal: muchos mínimos locales. "
                "Gradiente queda fácilmente en un mínimo local; "
                "búsqueda local puede explorar más si d es grande."
    },
    "Punto silla  f = x² − y²": {
        "f": _saddle, "grad": _grad_saddle,
        "span": ((-4, 4), (-4, 4)), "levels": np.linspace(-12, 12, 25),
        "x0_default": [0.5, 0.5],
        "desc": "Punto silla (saddle point) en (0, 0): la función crece en x "
                "y decrece en y. No existe un mínimo global. El gradiente "
                "empuja x→0 (bajando) pero y→∞ (bajando también). "
                "Útil para observar: ¿qué hacen los métodos cerca de (0,0)? "
                "¿Se quedan, escapan, divergen?"
    },
}


# =============================================================
# Dashboard interactivo: Búsqueda Local vs Gradiente
# =============================================================

def dashboard_busqueda():
    """
    Despliega un dashboard interactivo que compara búsqueda local
    y descenso por gradiente en 2D.

    * Compatible con Jupyter local y Google Colab.
    * Usa ipywidgets básicos (FloatSlider, IntSlider, Dropdown, Button, Output).
    """
    import ipywidgets as W

    # ---------- widgets ----------
    func_dd = W.Dropdown(
        options=list(FUNCIONES_PRUEBA.keys()),
        value=list(FUNCIONES_PRUEBA.keys())[0],
        description="Función:",
        style={"description_width": "70px"},
        layout=W.Layout(width="400px"),
    )
    desc_html = W.HTML(layout=W.Layout(width="100%"))

    x0_slider = W.FloatSlider(value=3.0, min=-4, max=4, step=0.1,
                               description="x₀", layout=W.Layout(width="260px"))
    y0_slider = W.FloatSlider(value=3.0, min=-4, max=4, step=0.1,
                               description="y₀", layout=W.Layout(width="260px"))

    # --- Búsqueda local ---
    bl_header = W.HTML(value="<b style='color:#1a73e8'>Búsqueda Local</b>")
    d_slider  = W.FloatSlider(value=0.5, min=0.05, max=2.0, step=0.05,
                               description="radio d", layout=W.Layout(width="260px"))
    k_slider  = W.IntSlider(value=8, min=3, max=24, step=1,
                             description="vecinos k", layout=W.Layout(width="260px"))
    bl_iters  = W.IntSlider(value=80, min=10, max=300, step=10,
                             description="iter máx", layout=W.Layout(width="260px"))

    # --- Gradiente ---
    gd_header = W.HTML(value="<b style='color:#d93025'>Gradiente</b>")
    alpha_slider = W.FloatSlider(value=0.05, min=0.001, max=0.5, step=0.001,
                                  description="α (lr)",
                                  readout_format=".3f",
                                  layout=W.Layout(width="260px"))
    gd_iters  = W.IntSlider(value=80, min=10, max=300, step=10,
                             description="iter máx", layout=W.Layout(width="260px"))

    btn_run = W.Button(description="▶  Ejecutar", button_style="primary",
                       layout=W.Layout(width="160px", height="36px"))
    out_plots = W.Output()
    summary_html = W.HTML()

    # ---------- update description on func change ----------
    def _on_func_change(change):
        key = change["new"]
        info = FUNCIONES_PRUEBA[key]
        desc_html.value = f"<p style='color:#555; font-size:13px'>{info['desc']}</p>"
        # adjust sliders to recommended x0
        x0d = info["x0_default"]
        sp = info["span"]
        x0_slider.min, x0_slider.max = sp[0]
        y0_slider.min, y0_slider.max = sp[1]
        x0_slider.value, y0_slider.value = x0d[0], x0d[1]

    func_dd.observe(_on_func_change, names="value")
    _on_func_change({"new": func_dd.value})   # trigger once

    # ---------- main callback ----------
    def _run(_):
        out_plots.clear_output(wait=True)
        summary_html.value = ""
        key = func_dd.value
        info = FUNCIONES_PRUEBA[key]
        f_2d = info["f"]
        grad_2d = info["grad"]
        span = info["span"]
        levels = info["levels"]

        x0 = np.array([x0_slider.value, y0_slider.value], dtype=float)

        # wrapper for f that accepts np.array  → scalar
        def f_scalar(v):
            v = np.asarray(v, dtype=float).ravel()
            return float(f_2d((v[0], v[1])))

        # run algorithms
        gen_vec = lambda x: vecindario_circular(x, d=d_slider.value, k=k_slider.value)
        path_bl = busqueda_local(x0, f_scalar, gen_vec, iters=bl_iters.value)
        path_gd = descenso_gradiente(x0, grad_2d, alpha=alpha_slider.value,
                                      iters=gd_iters.value, tol=1e-8)

        fvals_bl = [f_scalar(p) for p in path_bl]
        fvals_gd = [f_scalar(p) for p in path_gd]

        with out_plots:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            # --- panel 1: contour + trajectories ---
            ax = axes[0]
            gx = np.linspace(*span[0], 250)
            gy = np.linspace(*span[1], 250)
            X, Y = np.meshgrid(gx, gy)
            Z = f_2d((X, Y))
            ax.contour(X, Y, Z, levels=levels, alpha=0.55, linewidths=0.8)
            ax.contourf(X, Y, Z, levels=levels, alpha=0.15, cmap="viridis")
            ax.plot(path_bl[:, 0], path_bl[:, 1], "o-", ms=4, lw=1.2, color="#1a73e8",
                    label=f"Búsqueda local ({len(path_bl)-1} pasos)")
            ax.plot(path_gd[:, 0], path_gd[:, 1], "s-", ms=3, lw=1.2, color="#d93025",
                    label=f"Gradiente ({len(path_gd)-1} pasos)")
            ax.plot(*x0, "k*", ms=14, zorder=5, label="Inicio")
            ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
            ax.set_title("Trayectorias en el espacio de decisión")
            ax.legend(fontsize=8, loc="best"); ax.set_aspect("equal"); ax.grid(alpha=0.2)

            # --- panel 2: convergence ---
            ax2 = axes[1]
            ax2.plot(fvals_bl, "o-", ms=3, color="#1a73e8", label="Búsq. local")
            ax2.plot(fvals_gd, "s-", ms=3, color="#d93025", label="Gradiente")
            ax2.set_xlabel("Iteración"); ax2.set_ylabel("$f(x)$")
            best_f = min(min(fvals_bl), min(fvals_gd))
            if best_f > 0:
                ax2.set_yscale("log")
            ax2.set_title("Convergencia de $f$")
            ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

            # --- panel 3: neighborhood snapshot ---
            ax3 = axes[2]
            ax3.contour(X, Y, Z, levels=levels, alpha=0.4, linewidths=0.6)
            ax3.contourf(X, Y, Z, levels=levels, alpha=0.10, cmap="viridis")
            # show neighborhood at starting point
            vecinos = vecindario_circular(x0, d=d_slider.value, k=k_slider.value)
            vx = [v[0] for v in vecinos]; vy = [v[1] for v in vecinos]
            circle = plt.Circle(x0, d_slider.value, fill=False, color="#1a73e8",
                                 linestyle="--", linewidth=1.5, label=f"radio d={d_slider.value}")
            ax3.add_patch(circle)
            ax3.scatter(vx, vy, s=40, color="#1a73e8", zorder=4, label=f"k={k_slider.value} vecinos")
            # gradient arrow
            g0 = grad_2d(x0)
            g_norm = g0 / (np.linalg.norm(g0) + 1e-15)
            arrow_len = d_slider.value * 1.2
            ax3.annotate("", xy=x0 - arrow_len*g_norm, xytext=x0,
                         arrowprops=dict(arrowstyle="->", lw=2, color="#d93025"))
            ax3.plot(*x0, "k*", ms=14, zorder=5)
            ax3.set_xlim(x0[0] - 2*d_slider.value, x0[0] + 2*d_slider.value)
            ax3.set_ylim(x0[1] - 2*d_slider.value, x0[1] + 2*d_slider.value)
            ax3.set_title("Vecindario vs dirección gradiente")
            ax3.set_xlabel("$x_1$"); ax3.set_ylabel("$x_2$")
            ax3.legend(fontsize=7, loc="best"); ax3.set_aspect("equal"); ax3.grid(alpha=0.2)

            try:
                plt.tight_layout()
            except (np.linalg.LinAlgError, ValueError):
                pass  # degenerate axes transform when aspect="equal"
            plt.show()

        # summary text
        summary_html.value = (
            f"<div style='margin-top:6px; font-size:13px;'>"
            f"<b>Búsqueda local</b>: {len(path_bl)-1} pasos → "
            f"f* = {fvals_bl[-1]:.6g} en ({path_bl[-1,0]:.4f}, {path_bl[-1,1]:.4f})<br>"
            f"<b>Gradiente</b>: {len(path_gd)-1} pasos → "
            f"f* = {fvals_gd[-1]:.6g} en ({path_gd[-1,0]:.4f}, {path_gd[-1,1]:.4f})"
            f"</div>"
        )

    btn_run.on_click(_run)

    # ---------- layout ----------
    controls_bl = W.VBox([bl_header, d_slider, k_slider, bl_iters])
    controls_gd = W.VBox([gd_header, alpha_slider, gd_iters])
    controls_row = W.HBox([
        W.VBox([func_dd, desc_html, W.HBox([x0_slider, y0_slider])]),
    ])
    param_row = W.HBox([controls_bl, controls_gd, W.VBox([btn_run])],
                        layout=W.Layout(gap="20px"))
    ui = W.VBox([controls_row, param_row, out_plots, summary_html])
    display(ui)
    _run(None)  # initial render
