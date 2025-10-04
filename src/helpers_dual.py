
import numpy as np
import matplotlib.pyplot as plt

def _safe_bound(b, a):
    if a <= 0:
        return np.inf
    return b / a

def plot_primal_1d_continuous(gain_per_liter, alpha, resource_available, resource_label="Recurso"):
    """
    Grafica beneficio vs x para un LP de 1 var con 1 restricción: alpha * x <= B.
    """
    xmax = _safe_bound(resource_available, alpha)
    xs = np.linspace(0, max(6, xmax*1.2), 300)
    z = gain_per_liter * xs

    plt.plot(xs, z, label=f"Beneficio = {gain_per_liter}·x")
    plt.axvline(xmax, linestyle="--", label=f"Límite {resource_label} (x = {xmax:.2f})")
    plt.xlabel("Litros producidos/vendidos (x)")
    plt.ylabel("Beneficio")
    plt.title(f"1 recurso: límite por {resource_label}")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_primal_1var_multiconst(gain_per_liter, alphas, bs, labels=None):
    """
    Grafica beneficio vs x y dibuja líneas verticales por cada restricción alpha_i x <= b_i.
    Marca el mínimo de los límites como el 'óptimo' factible.
    """
    bounds = [_safe_bound(b, a) for a, b in zip(alphas, bs)]
    xopt = np.min(bounds)
    xmax = np.max(bounds)
    xs = np.linspace(0, max(6, xmax*1.2), 400)
    z = gain_per_liter * xs

    plt.plot(xs, z, label=f"Beneficio = {gain_per_liter}·x")
    for i, xlim in enumerate(bounds):
        lbl = labels[i] if labels and i < len(labels) else f"Restricción {i}"
        plt.axvline(xlim, linestyle="--", label=f"{lbl} (x ≤ {xlim:.2f})")
    plt.axvline(xopt, label=f"Óptimo factible (x* = {xopt:.2f})")
    plt.xlabel("Litros producidos/vendidos (x)")
    plt.ylabel("Beneficio")
    plt.title("1 variable con múltiples restricciones")
    plt.grid(True)
    plt.legend()
    plt.show()

def solve_primal_pulp(gains, A, b, var_bounds=None, names=None):
    """
    Resuelve un LP del tipo:
        max c^T x
        s.a. A_i x <= b_i (para cada fila i de A)
             bounds sobre x (por defecto x >= 0)
    Devuelve dict con objetivo, x, duales (pi) y holguras.
    """
    try:
        import pulp
    except Exception as e:
        return {"error": "PuLP no está instalado. Ejecuta `pip install pulp` y vuelve a intentar.", "exception": str(e)}

    n = len(gains)
    m = len(b)
    if names is None:
        names = [f"x{j}" for j in range(n)]
    if var_bounds is None:
        var_bounds = [(0, None)] * n

    prob = pulp.LpProblem("Limonadita_de_Mango_SAS", pulp.LpMaximize)
    xs = [pulp.LpVariable(names[j], lowBound=var_bounds[j][0], upBound=var_bounds[j][1]) for j in range(n)]

    prob += pulp.lpDot(gains, xs)

    for i in range(m):
        prob += (pulp.lpDot(A[i], xs) <= b[i]), f"c{i}"

    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
    except Exception:
        prob.solve()

    obj = pulp.value(prob.objective)
    xvals = {v.name: v.value() for v in xs}
    duals = {name: getattr(c, "pi", None) for name, c in prob.constraints.items()}
    slacks = {name: getattr(c, "slack", None) for name, c in prob.constraints.items()}

    return {"objective": obj, "x": xvals, "duals": duals, "slacks": slacks, "status": pulp.LpStatus[prob.status]}

def pretty_dual_generic(A, b, c, constraint_labels=None):
    """
    Imprime el dual de un primal estándar:
        max c^T x
        s.a. A x <= b, x >= 0
    Dual:
        min b^T y
        s.a. A^T y >= c, y >= 0
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    c = np.array(c, dtype=float).flatten()

    m, n = A.shape
    labels = constraint_labels if constraint_labels and len(constraint_labels)==m else [f"c{i}" for i in range(m)]

    print("Dual (genérico):")
    # Objetivo
    terms = [f"{b[i]}·y_{labels[i]}" for i in range(m)]
    print("  min  w =", " + ".join(terms))

    # Restricciones A^T y >= c
    print("  s.a.")
    AT = A.T
    for j in range(n):
        lhs_terms = []
        for i in range(m):
            coeff = AT[j, i]
            if abs(coeff) > 1e-12:
                lhs_terms.append(f"{coeff:g}·y_{labels[i]}")
        lhs = " + ".join(lhs_terms) if lhs_terms else "0"
        print(f"       {lhs} >= {c[j]}  (para x_{j})")

    print("       y >= 0")

def check_complementary_slackness(result_dict, tol=1e-6):
    """
    Verifica y_i * slack_i ≈ 0 para todas las restricciones del LP resuelto.
    """
    duals = result_dict.get("duals", {})
    slacks = result_dict.get("slacks", {})
    if not duals or not slacks:
        print("No hay información de duales/holguras (¿solver no soporta pi/slack?).")
        return

    print("Chequeo de Holgura Complementaria:")
    ok = True
    for k in duals.keys():
        y = duals[k]
        s = slacks[k]
        print(f"  {k}: y={y}, slack={s}")
        if y is None or s is None:
            ok = False
        else:
            if abs(y * s) > tol:
                ok = False

    if ok:
        print("  ✔ Cumple (y_i * slack_i ≈ 0).")
    else:
        print("  ⚠ Puede no cumplir estrictamente (revisar tolerancia/solver).")
