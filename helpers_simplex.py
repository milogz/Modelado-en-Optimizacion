import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math, HTML
import ipywidgets as W
from ipywidgets import HTMLMath


def _blk(s):
    # Negro: info de la base actual
    return f"<div style='color:#111; font-weight:500;'>{s}</div>"

def _blu(s):
    # Azul: x y costos reducidos
    return f"<div style='color:#1e3a8a; font-weight:500;'>{s}</div>"

def _red(s):
    # Rojo: movimiento (dirección/longitud/nuevo punto)
    return f"<div style='color:#b91c1c; font-weight:600;'>{s}</div>"


def init_fig(A, b):
    # Figura única: región factible + BFS
    fig, ax = plt.subplots(figsize=(6,5))
    ax, line_colors, hull = plot_method_graph(A[:, :2], b, x=None, ax=ax,
                                              title="Método gráfico + BFS", show_fill=True)
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    plt.close(fig)

    fig_widget = W.Output()
    with fig_widget:
        display(fig)
    return fig, ax, fig_widget

# =============================================================
# CALLBACKS DE LOS BOTONES
# =============================================================

def _draw_point_and_vectors(A, b, fig, ax, fig_widget, x_now=None, d=None, theta=None):
    # Limpia y repinta: región factible + líneas (se conserva estilo)
    ax.clear()
    plot_method_graph(A[:, :2], b, x=None, ax=ax, title="Método gráfico + BFS", show_fill=True)
    # Punto actual en negro
    if x_now is not None:
        ax.scatter([x_now[0]], [x_now[1]], s=60, color="black", zorder=6)
    # Flecha de dirección y nuevo punto en rojo
    if (d is not None) and (theta is not None) and np.isfinite(theta):
        p = x_now
        q = x_now + theta * d[:2]
        ax.annotate("", xy=(q[0], q[1]), xytext=(p[0], p[1]),
                    arrowprops=dict(arrowstyle="->", linewidth=2, color="red"))
        ax.scatter([q[0]], [q[1]], s=60, color="red", zorder=6)
    fig_widget.clear_output(wait=True)
    with fig_widget:
        display(fig)


# === CALLBACKS ===
def _on_info_clicked(_):
    log_base.value = ""     
    log_move.value = ""  

    try:
        # === Leer checkboxes y fijar Bcols ===
        seleccionados = [chk.value for j, chk in enumerate(checks.children, start=1)]
        STATE["Bcols"] = seleccionados

        # === Resto del flujo ===
        info, html_blk = resumir_info_base_actual(STATE["A"], STATE["b"], STATE["c"], STATE["Bcols"])
        STATE["info"] = info

        var_names = ['x1','x2','s1','s2','s3']
        nonbasic = [(v,i) for i,v in enumerate(var_names) if not STATE["Bcols"][i]]
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
    log_move.value = ""     # opcional: limpiar solo el panel rojo
    try:
        if STATE["info"] is None:
            log_move.value += "<div style='color:#b91c1c'>Primero pulsa “1) Info Base Actual”.</div>"
            return
        j_in = enter_drop.value
        STATE["last_enter"] = j_in

        st, html_red = calcular_direccion_y_longitud(STATE["A"], STATE["info"], j_in)        
        log_move.value += _red(html_red)

        if np.isfinite(st["theta"]):
            _draw_point_and_vectors(x_now=STATE["x_plot"], d=st["d"], theta=st["theta"])
            STATE["x_plot"] = st["x_new"][:2]
            if st["leave"] is not None:
                '''
                B_idx = [i for i,v in enumerate(STATE["Bcols"]) if v]
                print(B_idx)
                Bset = set(B_idx) - {st["leave"]}
                Bset.add(j_in)
                aux = [(i in Bset) for i in range(STATE["A"].shape[1]) ]
                print(aux)
                '''                
                STATE["Bcols"][j_in] = True
                STATE["Bcols"][st["leave"]] = False
    except Exception as e:
        log_move.value += f"<pre style='color:#b91c1c'>[error move] {e}</pre>"



# ============= Formatting helpers =============

def mat_to_bmatrix(M, fmt="{:g}"):
    M = np.array(M)
    rows = [" & ".join(fmt.format(v) for v in row) for row in M]
    body = " \\\\ ".join(rows)     # <-- aquí sí van dos backslashes para el salto de fila en LaTeX
    return r"\begin{bmatrix}" + body + r"\end{bmatrix}"  # <-- begin/end con UN solo backslash

def vec_to_bmatrix(v, fmt="{:g}"):
    v = np.array(v).reshape(-1, 1)
    return mat_to_bmatrix(v, fmt=fmt)


def mat_to_bmatrix_old(M, fmt="{:g}"):
    rows = [" & ".join(fmt.format(v) for v in row) for row in np.array(M)]
    return r"\\begin{bmatrix}" + r" \\ ".join(rows) + r"\\end{bmatrix}"


def vec_to_bmatrix_old(v, fmt="{:g}"):
    v = np.array(v).reshape(-1,1)
    return mat_to_bmatrix(v, fmt=fmt)

# ============= Geometry & plot helpers =============

def _intersect_lines(a1, b1, a2, b2, tol=1e-9):
    A = np.vstack([a1, a2])
    if abs(np.linalg.det(A)) < tol:
        return None
    return np.linalg.solve(A, np.array([b1, b2], dtype=float))


def feasible_polygon(A, b, add_axes=True, tol=1e-9):
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
    import matplotlib.pyplot as plt
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

# ============= Algebra simple para el Simplex didáctico =============

def basis_info(A_full, b, c, Bcols):
    B_idx = [i for i in range(len(Bcols)) if Bcols[i]]
    m, n = A_full.shape
    B = A_full[:, B_idx]
    #B_idx = [i for i,v in enumerate(Bcols) if v]
    Ncols = [j for j in range(n) if j not in B_idx]
    N = A_full[:, Ncols]
    x = np.zeros(n)    
    xB = np.linalg.solve(B, b.astype(float))
    x[B_idx] = xB
    cB = c[B_idx]
    cN = c[Ncols]
    y = np.linalg.solve(B.T, cB)          # precios sombra (B^T y = c_B)
    r = cN - N.T @ y                      # costos reducidos
    return {"B":B, "N":N, "Bcols":Bcols, "Ncols":Ncols,
            "x":x, "xB":xB, "cB":cB, "cN":cN, "y":y, "r":r}


def step_info(A_full, x, Bcols, enter_j):    
    B_idx = [i for i in range(len(Bcols)) if Bcols[i]]
    B = A_full[:, B_idx]
    a_j = A_full[:, enter_j]
    d = np.zeros_like(x)
    dB = -np.linalg.solve(B, a_j)
    d[enter_j] = 1.0
    for k, j in enumerate(B_idx):
        d[j] = dB[k]
    mask = dB < 0
    if not np.any(mask):
        theta = np.inf
        leave = None
    else:
        xB = x[B_idx]
        ratios = xB[mask] / (-dB[mask])
        idx = int(np.argmin(ratios))
        theta = float(np.min(ratios))
        #leave = Bcols[np.where(mask)[0][idx]]
        leave = B_idx[idx]
    x_new = x + (theta * d if np.isfinite(theta) else 0.0)
    return {"d":d, "dB":dB, "theta":theta, "leave":leave, "x_new":x_new}