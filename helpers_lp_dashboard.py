"""
helpers_lp_dashboard.py
=======================
Dashboard interactivo unificado para exploración de LP (bases, holguras,
restricciones activas) y el método Simplex paso a paso.

Compatible con Jupyter local y Google Colab (ipywidgets básicos).
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math
import ipywidgets as W

# Reutilizamos funciones ya existentes (sin duplicar)
from helpers_bases import (
    feasible_polygon, plot_method_graph, slacks, bar_slacks,
    basis_info, step_info, mat_to_bmatrix, vec_to_bmatrix,
)

# =============================================================
# Tab 1:  Exploración de factibilidad, holguras y bases
# =============================================================

def _tab_factibilidad(A, b, c, var_names_orig):
    """
    Panel interactivo: mover (x1, x2) con sliders y ver en tiempo real
    holguras, restricciones activas, punto en la región factible, y valor
    de la FO.  Incluye selector de base para ver el punto extremo
    correspondiente.
    """
    m = A.shape[0]
    n_orig = A.shape[1]   # should be 2 for geometric plot
    hull = feasible_polygon(A, b, add_axes=True)

    # Axis limits from the hull
    if hull.size:
        xmax = float(np.max(hull[:, 0]) + 1.5)
        ymax = float(np.max(hull[:, 1]) + 1.5)
    else:
        xmax, ymax = 10.0, 10.0

    # ----- Widgets -----
    section_label = W.HTML(value="<h3 style='color:#1a73e8; margin:0'>🔍 Exploración de punto y holguras</h3>")
    x1_sl = W.FloatSlider(value=0.0, min=0, max=xmax, step=0.1,
                          description="x₁", layout=W.Layout(width="300px"))
    x2_sl = W.FloatSlider(value=0.0, min=0, max=ymax, step=0.1,
                          description="x₂", layout=W.Layout(width="300px"))
    info_html = W.HTML()
    out_plot = W.Output()

    def _update(change=None):
        out_plot.clear_output(wait=True)
        x = np.array([x1_sl.value, x2_sl.value], dtype=float)
        s = slacks(A, b, x)
        feas = bool(np.all(s >= -1e-9) and np.all(x >= -1e-9))
        z = float(c[:n_orig].T @ x)

        with out_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # --- left: feasible region + point ---
            plot_method_graph(A, b, x=x, ax=ax1, title="Región factible", show_fill=True)
            # add objective contour
            gx = np.linspace(0, xmax, 200)
            gy = np.linspace(0, ymax, 200)
            X, Y = np.meshgrid(gx, gy)
            Z = c[0]*X + c[1]*Y
            ax1.contour(X, Y, Z, levels=12, colors="#999", alpha=0.35, linewidths=0.6, linestyles="--")
            ax1.set_title("Región factible y punto evaluado")

            # --- right: variable values as bars ---
            all_vals = np.hstack([x, s])
            names = list(var_names_orig) + [f"s{i+1}" for i in range(m)]
            n_all = len(all_vals)
            colors = []
            for i, v in enumerate(all_vals):
                if i < n_orig:
                    colors.append("#555555")
                elif v < -1e-9:
                    colors.append("#d93025")  # violated
                elif abs(v) < 1e-9:
                    colors.append("#f9ab00")  # active (zero slack)
                else:
                    colors.append("#34a853")  # inactive (positive slack)

            y_pos = np.arange(n_all)
            ax2.barh(y_pos, all_vals, color=colors)
            ax2.set_yticks(y_pos, names)
            ax2.axvline(0, linewidth=1, color="black")
            ax2.set_xlabel("Valor")
            ax2.set_title("Variables: originales y de holgura")

            # legend patches
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#555555", label="Variable original"),
                Patch(facecolor="#34a853", label="Holgura > 0 (inactiva)"),
                Patch(facecolor="#f9ab00", label="Holgura = 0 (activa)"),
                Patch(facecolor="#d93025", label="Holgura < 0 (violada)"),
            ]
            ax2.legend(handles=legend_elements, fontsize=7, loc="lower right")

            plt.tight_layout()
            plt.show()

        # info text
        activas = [f"R{i+1}" for i in range(m) if abs(s[i]) < 1e-9]
        info_html.value = (
            f"<div style='font-size:13px; margin-top:4px'>"
            f"<b>¿Factible?</b>: {'✅ Sí' if feas else '❌ No'} &nbsp; | &nbsp; "
            f"<b>z = c<sup>T</sup>x</b> = {z:.2f} &nbsp; | &nbsp; "
            f"<b>Holguras</b>: s = {np.array2string(s, precision=2)} &nbsp; | &nbsp; "
            f"<b>Restricciones activas</b>: {activas if activas else 'ninguna'}"
            f"</div>"
        )

    x1_sl.observe(_update, names="value")
    x2_sl.observe(_update, names="value")

    panel = W.VBox([
        section_label,
        W.HBox([x1_sl, x2_sl]),
        info_html,
        out_plot,
    ])
    _update()  # initial render
    return panel


# =============================================================
# Tab 2:  Simplex paso a paso
# =============================================================

def _tab_simplex(A_full, b, c_full, var_names):
    """
    Visualizador paso a paso del método Simplex con gráfico geométrico,
    dirección/longitud, y transición entre bases.
    """
    m, n = A_full.shape

    # ----- State -----
    STATE = {
        "Bcols": list(range(n - m, n)),   # initial: slack columns
        "info": None,
        "step_count": 0,
        "optimal": False,
        "history": [],
    }

    # ----- Widgets -----
    section_label = W.HTML(value="<h3 style='color:#b91c1c; margin:0'>⚡ Simplex paso a paso</h3>")
    base_html = W.HTML()
    step_html = W.HTML()
    out_plot = W.Output()

    btn_step = W.Button(description="▶ Siguiente paso", button_style="primary",
                        layout=W.Layout(width="180px", height="34px"))
    btn_reset = W.Button(description="↺ Reiniciar", button_style="warning",
                         layout=W.Layout(width="130px", height="34px"))

    # Manual override: let user pick entry variable
    enter_dd = W.Dropdown(description="Entrante:", options=[], layout=W.Layout(width="200px"))
    auto_cb = W.Checkbox(value=True, description="Auto (mayor r>0)", indent=False,
                         layout=W.Layout(width="200px"))

    def _render():
        """Compute and display current base info + plot."""
        out_plot.clear_output(wait=True)
        Bcols = STATE["Bcols"]

        # Build boolean mask for basis_info
        Bcols_bool = [i in Bcols for i in range(n)]
        info = basis_info(A_full, b, c_full, Bcols)
        STATE["info"] = info

        x = info["x"]
        z = float(info["cB"].T @ info["xB"])
        feas = bool(np.all(info["xB"] >= -1e-9))
        r = info["r"]
        Ncols = info["Ncols"]

        # Check optimality (max: all reduced costs <= 0)
        STATE["optimal"] = bool(np.all(r <= 1e-9))

        # Update entry dropdown
        if not STATE["optimal"]:
            options = [(f"{var_names[j]} (r={r[k]:.3f})", j)
                       for k, j in enumerate(Ncols) if r[k] > 1e-9]
            enter_dd.options = options
            if options:
                # default: largest reduced cost
                best_k = int(np.argmax(r))
                enter_dd.value = Ncols[best_k]
        else:
            enter_dd.options = [("— óptimo —", -1)]

        # Save to history
        STATE["history"].append(x[:2].copy())

        # --- Base info HTML ---
        IB = [var_names[j] for j in Bcols]
        IN = [var_names[j] for j in Ncols]
        status_str = "🏆 ÓPTIMO" if STATE["optimal"] else ("❌ Infactible" if not feas else "▶ Factible")
        r_strs = [f"{var_names[Ncols[k]]}:{r[k]:+.3f}" for k in range(len(Ncols))]

        base_html.value = (
            f"<div style='font-size:13px; border-left:3px solid #1a73e8; padding-left:8px; margin:4px 0'>"
            f"<b>Iteración {STATE['step_count']}</b> &nbsp; {status_str}<br>"
            f"<b>Base</b>: {{{', '.join(IB)}}} &nbsp; | &nbsp; <b>No base</b>: {{{', '.join(IN)}}}<br>"
            f"<b>x</b> = [{', '.join(f'{v:.3f}' for v in x)}] &nbsp; → &nbsp; <b>z = {z:.3f}</b><br>"
            f"<b>Costos reducidos (r)</b>: [{', '.join(r_strs)}]"
            f"</div>"
        )

        with out_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

            # --- Left: geometric view ---
            A_orig = A_full[:, :2]
            plot_method_graph(A_orig, b, x=None, ax=ax1,
                              title="Simplex: movimiento entre vértices", show_fill=True)

            # plot trajectory
            hist = STATE["history"]
            if len(hist) > 1:
                H = np.array(hist)
                ax1.plot(H[:, 0], H[:, 1], "o--", color="#888", ms=5, lw=1.0, zorder=4, alpha=0.6)
                for t, pt in enumerate(hist[:-1]):
                    ax1.annotate(str(t), xy=pt, fontsize=7, color="#888",
                                 ha="center", va="bottom")
            # current point
            ax1.scatter([x[0]], [x[1]], s=100, color="#1a73e8" if feas else "#d93025",
                        edgecolors="black", linewidths=1.2, zorder=6)
            ax1.annotate(f"iter {STATE['step_count']}", xy=(x[0], x[1]),
                         fontsize=8, fontweight="bold", color="#1a73e8",
                         ha="left", va="bottom", xytext=(5, 5), textcoords="offset points")

            # objective level curve through current point
            gx = np.linspace(0, float(ax1.get_xlim()[1]), 200)
            gy = np.linspace(0, float(ax1.get_ylim()[1]), 200)
            X, Y = np.meshgrid(gx, gy)
            Z = c_full[0]*X + c_full[1]*Y
            ax1.contour(X, Y, Z, levels=[z], colors="#e8710a", linewidths=1.5, linestyles="-")

            # --- Right: variable values bar chart ---
            all_names = var_names
            colors = []
            for i in range(n):
                if i in Bcols:
                    if x[i] < -1e-9:
                        colors.append("#d93025")
                    else:
                        colors.append("#1a73e8")
                else:
                    colors.append("#cccccc")
            y_pos = np.arange(n)
            ax2.barh(y_pos, x, color=colors)
            ax2.set_yticks(y_pos, all_names)
            ax2.axvline(0, linewidth=1, color="black")
            ax2.set_xlabel("Valor")
            ax2.set_title("Valores de todas las variables")

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#1a73e8", label="Básica (≥0)"),
                Patch(facecolor="#d93025", label="Básica (<0, infactible)"),
                Patch(facecolor="#cccccc", label="No básica (=0)"),
            ]
            ax2.legend(handles=legend_elements, fontsize=7, loc="lower right")

            plt.tight_layout()
            plt.show()

    def _on_step(_):
        if STATE["optimal"]:
            step_html.value = "<div style='color:#34a853; font-weight:bold'>✅ Solución óptima alcanzada.</div>"
            return

        info = STATE["info"]
        if info is None:
            return

        # Determine entering variable
        if auto_cb.value:
            r = info["r"]
            Ncols = info["Ncols"]
            best_k = int(np.argmax(r))
            j_in = Ncols[best_k]
        else:
            j_in = enter_dd.value
            if j_in == -1:
                return

        # Compute step
        st = step_info(A_full, info["x"], STATE["Bcols"], j_in)

        if not np.isfinite(st["theta"]):
            step_html.value = (
                "<div style='color:#d93025; font-weight:bold'>"
                "⚠️ Problema no acotado (θ = ∞ en la dirección elegida).</div>"
            )
            return

        leave_j = st["leave"]
        step_html.value = (
            f"<div style='font-size:13px; border-left:3px solid #b91c1c; padding-left:8px; margin:4px 0'>"
            f"<b>Movimiento:</b> entra <b>{var_names[j_in]}</b>, "
            f"sale <b>{var_names[leave_j]}</b><br>"
            f"<b>Dirección d</b> = [{', '.join(f'{v:.3f}' for v in st['d'])}]<br>"
            f"<b>Longitud θ</b> = {st['theta']:.4f}<br>"
            f"<b>Nuevo x</b> = [{', '.join(f'{v:.3f}' for v in st['x_new'])}]"
            f"</div>"
        )

        # Update state
        new_Bcols = list(STATE["Bcols"])
        new_Bcols.remove(leave_j)
        new_Bcols.append(j_in)
        new_Bcols.sort()
        STATE["Bcols"] = new_Bcols
        STATE["step_count"] += 1

        _render()

    def _on_reset(_):
        STATE["Bcols"] = list(range(n - m, n))
        STATE["info"] = None
        STATE["step_count"] = 0
        STATE["optimal"] = False
        STATE["history"] = []
        step_html.value = ""
        _render()

    btn_step.on_click(_on_step)
    btn_reset.on_click(_on_reset)

    controls = W.HBox([btn_step, btn_reset, auto_cb, enter_dd])
    panel = W.VBox([
        section_label,
        controls,
        base_html,
        step_html,
        out_plot,
    ])
    _render()  # initial
    return panel


# =============================================================
# Dashboard principal (unifica ambos tabs)
# =============================================================

def lp_dashboard(A=None, b=None, c=None, var_names=None):
    """
    Dashboard interactivo para explorar conceptos de LP:
    - Tab 1: Factibilidad, holguras, restricciones activas
    - Tab 2: Método Simplex paso a paso

    Parámetros:
      A : np.array (m, 2) — coeficientes de las restricciones Ax <= b
      b : np.array (m,)   — lado derecho
      c : np.array (2,)   — costos de la FO (max c^T x)

    Si no se proporcionan, usa el ejemplo por defecto:
      max 3x1 + 2x2  s.a.  x1+x2<=6, x1+2x2<=8, 2x1+x2<=8
    """
    # --- Defaults ---
    if A is None:
        A = np.array([[1., 1.], [1., 2.], [2., 1.]])
        b = np.array([6., 8., 8.])
        c = np.array([3., 2.])

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    m, n_orig = A.shape

    if var_names is None:
        var_names = [f"x{i+1}" for i in range(n_orig)]

    # Build standard form: A_full = [A | I]
    A_full = np.hstack([A, np.eye(m)])
    c_full = np.hstack([c, np.zeros(m)])
    all_names = list(var_names) + [f"s{i+1}" for i in range(m)]

    # --- Title ---
    title = W.HTML(value=(
        "<h2 style='margin:0 0 4px 0'>🧪 Lab LP: Factibilidad, Bases y Simplex</h2>"
        "<p style='color:#555; font-size:13px; margin:0 0 8px 0'>"
        f"Problema: max {' + '.join(f'{c[i]:g}·{var_names[i]}' for i in range(n_orig))} &nbsp; "
        f"con {m} restricciones y {n_orig} variables de decisión."
        "</p>"
    ))

    # --- Tabs ---
    tab1 = _tab_factibilidad(A, b, c, var_names)
    tab2 = _tab_simplex(A_full, b, c_full, all_names)

    tabs = W.Tab(children=[tab1, tab2])
    tabs.set_title(0, "🔍 Factibilidad y Holguras")
    tabs.set_title(1, "⚡ Simplex Paso a Paso")

    display(W.VBox([title, tabs]))
