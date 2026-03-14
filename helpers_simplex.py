import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math, HTML
import ipywidgets as W
from ipywidgets import HTMLMath

# Import shared functions from helpers_bases (eliminating duplication)
from helpers_bases import (
    mat_to_bmatrix, vec_to_bmatrix,
    _intersect_lines, feasible_polygon,
    plot_method_graph, basis_info, step_info,
)


# =============================================================
# HTML formatting helpers
# =============================================================

def _blk(s):
    """Negro: info de la base actual"""
    return f"<div style='color:#111; font-weight:500;'>{s}</div>"

def _blu(s):
    """Azul: x y costos reducidos"""
    return f"<div style='color:#1e3a8a; font-weight:500;'>{s}</div>"

def _red(s):
    """Rojo: movimiento (dirección/longitud/nuevo punto)"""
    return f"<div style='color:#b91c1c; font-weight:600;'>{s}</div>"


# =============================================================
# Funciones pseudo-algoritmo Simplex (HTML para Markdown)
# =============================================================

def resumir_info_base_actual(A, b, c, Bcols):
    """
    Calcula B, N, cB, cN, x (básico) y devuelve (info_dict, html_negro).
    """
    info = basis_info(A, b, c, Bcols)
    B, N = info["B"], info["N"]
    x, xB = info["x"], info["xB"]
    cB, cN = info["cB"], info["cN"]
    I_B = [j+1 for j, v in enumerate(Bcols) if v]   # 1-based
    I_N = [j+1 for j in info["Ncols"]]

    html_blk = fr"""
        <h4 style="color:#111">Base actual: $I_B = {I_B}$, $I_N = {I_N}$</h4>

        <p>$$c_B^T = {mat_to_bmatrix(cB.reshape(1,-1))},\quad
        c_N^T = {mat_to_bmatrix(cN.reshape(1,-1))}$$</p>

        <p>$$B = {mat_to_bmatrix(B)},\quad
        N = {mat_to_bmatrix(N)},\quad
        b = {vec_to_bmatrix(b)}$$</p>
        """
    return info, html_blk


def verificar_costos_reducidos(info):
    """Con info de la base calcula w, z, r y devuelve bloque HTML azul."""
    y = info["y"]
    r = info["r"]
    B_idx = [i for i in range(len(info["Bcols"])) if info["Bcols"][i]]
    z = float(info["cB"].T @ info["x"][B_idx])

    html_blue = fr"""
        <p><b>Solución básica actual:</b>
        <p>$$x = {vec_to_bmatrix(info["x"])},\;\; z = {z:.2f},\;\; w^{{T}} = c_B^{{T}} B^{{-1}} = {vec_to_bmatrix(y)}$$</p>
        <p>Costos reducidos:
        $$r = c_N - w^{{T}} N = {vec_to_bmatrix(info["r"])}$$
        </p>
        """
    return html_blue


def calcular_direccion_y_longitud(A, info, enter_j, var_names, last_enter):
    """
    A partir de la base e índice entrante (0-based), calcula dirección d,
    longitud θ (regla del mínimo cociente), variable saliente y x nuevo.
    Devuelve (step, html_rojo).
    """
    st = step_info(A, info["x"], info["Bcols"], enter_j)

    if np.isfinite(st["theta"]):
        html_red = fr"""
        <h4 style="color:#b91c1c">Movimiento:</h4>
        <p>Entra: {var_names[last_enter]}; esto induce $d^{last_enter+1}= -B^{{-1}}a_{last_enter+1}$</p>
        <p>$$x^{{t+1}} = x^{{t}} + \alpha d^{last_enter+1} $$</p>
        <p>$${vec_to_bmatrix(st["x_new"])} = {vec_to_bmatrix(info["x"])} + {st['theta']}{vec_to_bmatrix(st["d"])}$$</p>
        <p>(sale variable: {st['leave']+1}; i.e., {var_names[st['leave']]})</p>
        """
    else:
        html_red = fr"""
        <h4 style="color:#b91c1c">Movimiento</h4>
        <p>Problema no acotado en la dirección escogida (no hay $\theta$ finito).</p>
        """
    return st, html_red


# =============================================================
# Dashboard interactivo para el Simplex didáctico
# =============================================================

def simplex_interactivo(A, b, c, Bcols_init, var_names=None):
    """
    Crea y despliega un dashboard interactivo que permite explorar
    el método Simplex paso a paso sobre un LP en forma estándar.

    Parámetros:
      A        : np.array (m x n), forma estándar [A_orig | I]
      b        : np.array (m,)
      c        : np.array (n,)  coeficientes de la función objetivo
      Bcols_init: list[bool] de largo n, True = columna básica inicial
      var_names : list[str], nombres de las variables (len = n)
    """
    if var_names is None:
        n = A.shape[1]
        m_ = A.shape[0]
        var_names = [f"x{j+1}" for j in range(n - m_)] + [f"s{j+1}" for j in range(m_)]

    log_base = HTMLMath(value="")
    log_move = HTMLMath(value="")

    # Crear figura
    fig, ax = plt.subplots(figsize=(6, 5))
    ax, line_colors, hull = plot_method_graph(
        A[:, :2], b, x=None, ax=ax,
        title="Método gráfico + BFS", show_fill=True
    )
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    plt.close(fig)

    fig_widget = W.Output()
    with fig_widget:
        display(fig)

    # Estado mutable
    STATE = {
        "A": A, "b": b, "c": c,
        "Bcols": list(Bcols_init),
        "info": None,
        "last_enter": None,
        "x_plot": None,
    }

    def _draw_point_and_vectors(x_now=None, d=None, theta=None):
        ax.clear()
        plot_method_graph(A[:, :2], b, x=None, ax=ax,
                         title="Método gráfico + BFS", show_fill=True)
        if x_now is not None:
            ax.scatter([x_now[0]], [x_now[1]], s=60, color="black", zorder=6)
        if (d is not None) and (theta is not None) and np.isfinite(theta):
            p = x_now
            q = x_now + theta * d[:2]
            ax.annotate("", xy=(q[0], q[1]), xytext=(p[0], p[1]),
                        arrowprops=dict(arrowstyle="->", linewidth=2, color="red"))
            ax.scatter([q[0]], [q[1]], s=60, color="red", zorder=6)
        fig_widget.clear_output(wait=True)
        with fig_widget:
            display(fig)

    # === Callbacks ===
    def _on_info_clicked(_):
        log_base.value = ""
        log_move.value = ""
        try:
            seleccionados = [chk.value for chk in checks.children]
            STATE["Bcols"] = seleccionados

            info, html_blk = resumir_info_base_actual(
                STATE["A"], STATE["b"], STATE["c"], STATE["Bcols"]
            )
            STATE["info"] = info

            nonbasic = [(v, i) for i, v in enumerate(var_names) if not STATE["Bcols"][i]]
            enter_drop.options = nonbasic
            enter_drop.value = nonbasic[0][1]

            log_base.value += _blk(html_blk)
            html_blue = verificar_costos_reducidos(info)
            log_base.value += _blu(html_blue)

            STATE["x_plot"] = info["x"][:2]
            _draw_point_and_vectors(x_now=STATE["x_plot"])
        except Exception as e:
            log_base.value += f"<pre style='color:#b91c1c'>[error info] {e}</pre>"

    def _on_move_clicked(_):
        log_move.value = ""
        try:
            if STATE["info"] is None:
                log_move.value += '<div style="color:#b91c1c">Primero pulsa &quot;1) Info Base Actual&quot;.</div>'
                return
            j_in = enter_drop.value
            STATE["last_enter"] = j_in

            st, html_red = calcular_direccion_y_longitud(
                STATE["A"], STATE["info"], j_in, var_names, STATE["last_enter"]
            )
            log_move.value += _red(html_red)

            if np.isfinite(st["theta"]):
                _draw_point_and_vectors(x_now=STATE["x_plot"],
                                       d=st["d"], theta=st["theta"])
                STATE["x_plot"] = st["x_new"][:2]
                if st["leave"] is not None:
                    STATE["Bcols"][j_in] = True
                    STATE["Bcols"][st["leave"]] = False
        except Exception as e:
            log_move.value += f"<pre style='color:#b91c1c'>[error move] {e}</pre>"

    # === Layout ===
    checks = W.HBox([
        W.Checkbox(value=v, description=nm)
        for nm, v in zip(var_names, Bcols_init)
    ])
    btn_info = W.Button(description="1) Info Base", button_style="primary")
    nonbasic_init = [(v, i) for i, v in enumerate(var_names) if not Bcols_init[i]]
    enter_drop = W.Dropdown(description="Entra:", options=nonbasic_init,
                            value=nonbasic_init[0][1])
    btn_move = W.Button(description="2) Mover", button_style="warning")

    controls = W.HBox([btn_info, enter_drop, btn_move])
    right_panel = W.VBox([log_base, W.HTML("<hr>"), log_move])

    ui = W.HBox([
        W.VBox([checks, fig_widget], layout=W.Layout(width='60%')),
        W.VBox([controls, right_panel], layout=W.Layout(width='40%'))
    ])

    btn_info.on_click(_on_info_clicked, remove=True)
    btn_info.on_click(_on_info_clicked)
    btn_move.on_click(_on_move_clicked, remove=True)
    btn_move.on_click(_on_move_clicked)

    display(ui)
    _draw_point_and_vectors()
