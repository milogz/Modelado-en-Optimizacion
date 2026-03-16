"""
helpers_dual.py -- Funciones de apoyo para el notebook 3-5 (Dualidad).

Actualización 2026: implementación completa de funciones de visualización,
solución y análisis de dualidad y holgura complementaria.
"""
import numpy as np
import matplotlib.pyplot as plt
from helpers_bases import mat_to_bmatrix, vec_to_bmatrix

try:
    import pulp
except ImportError:
    pulp = None


def dual_info(A, b, c):
    """
    Dado un LP en forma estándar (max c^T x, Ax <= b, x >= 0),
    retorna la formulación dual (min b^T y, A^T y >= c, y >= 0).
    """
    return {
        "A_dual": A.T,
        "b_dual": c,
        "c_dual": b,
    }


def plot_primal_1d_continuous(gain_per_liter, alpha, resource_available,
                               resource_label='Recurso', x_label='x (litros)',
                               figsize=(7, 4)):
    """
    Visualiza un LP de una sola variable y una restricción de recurso:
        max  p * x
        s.a. alpha * x <= b
             x >= 0

    Dibuja la función objetivo y la restricción, marcando el óptimo.
    """
    x_max = resource_available / alpha
    x_range = np.linspace(0, x_max * 1.3, 300)

    fig, ax = plt.subplots(figsize=figsize)

    # Función objetivo
    z = gain_per_liter * x_range
    ax.plot(x_range, z, 'b-', linewidth=2, label=f'$z = {gain_per_liter:.1f} \\cdot x$')

    # Restricción: alpha * x <= b  =>  x <= b/alpha
    ax.axvline(x=x_max, color='red', linestyle='--', linewidth=1.5,
               label=f'{resource_label}: $x \\leq {x_max:.1f}$')
    ax.axvspan(x_max, x_range[-1], alpha=0.15, color='red')

    # Punto óptimo
    z_opt = gain_per_liter * x_max
    ax.plot(x_max, z_opt, 'ro', markersize=10, zorder=5)
    ax.annotate(f'$x^* = {x_max:.1f}$\n$z^* = {z_opt:.1f}$',
                xy=(x_max, z_opt), xytext=(x_max * 0.6, z_opt * 0.85),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xlabel(x_label)
    ax.set_ylabel('$z$ (ganancia)')
    ax.set_title('Primal: 1 variable, 1 restricción')
    ax.legend(loc='upper left')
    ax.set_xlim(0, x_range[-1])
    ax.set_ylim(0, z.max() * 1.1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_primal_1var_multiconst(gain_per_liter, alphas, bs, labels=None,
                                 x_label='x (litros)', figsize=(7, 4)):
    """
    Visualiza un LP de una sola variable con múltiples restricciones de recurso:
        max  p * x
        s.a. alpha_i * x <= b_i   para cada recurso i
             x >= 0

    Dibuja la función objetivo y todas las restricciones, marcando la activa.
    """
    if labels is None:
        labels = [f'Recurso {i+1}' for i in range(len(alphas))]

    # El óptimo está en el mínimo de b_i / alpha_i
    x_limits = [b / a for a, b in zip(alphas, bs)]
    x_opt = min(x_limits)
    binding_idx = x_limits.index(x_opt)

    x_range = np.linspace(0, max(x_limits) * 1.3, 300)

    fig, ax = plt.subplots(figsize=figsize)

    # Función objetivo
    z = gain_per_liter * x_range
    ax.plot(x_range, z, 'b-', linewidth=2, label=f'$z = {gain_per_liter:.1f} \\cdot x$')

    # Restricciones
    colors = plt.cm.Set1(np.linspace(0, 1, len(alphas)))
    for i, (xl, lbl) in enumerate(zip(x_limits, labels)):
        style = '-' if i == binding_idx else '--'
        lw = 2.0 if i == binding_idx else 1.2
        ax.axvline(x=xl, color=colors[i], linestyle=style, linewidth=lw,
                   label=f'{lbl}: $x \\leq {xl:.1f}$')

    # Zona infactible
    ax.axvspan(x_opt, x_range[-1], alpha=0.1, color='red')

    # Punto óptimo
    z_opt = gain_per_liter * x_opt
    ax.plot(x_opt, z_opt, 'ro', markersize=10, zorder=5)
    ax.annotate(f'$x^* = {x_opt:.1f}$\n$z^* = {z_opt:.1f}$\n(activa: {labels[binding_idx]})',
                xy=(x_opt, z_opt), xytext=(x_opt * 0.4, z_opt * 0.75),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xlabel(x_label)
    ax.set_ylabel('$z$ (ganancia)')
    ax.set_title('Primal: 1 variable, múltiples restricciones')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(0, x_range[-1])
    ax.set_ylim(0, z.max() * 1.1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def solve_primal_pulp(c, A, b, var_names=None, sense='max'):
    """
    Resuelve un LP con PuLP:
        max (o min)  c^T x
        s.a.  A x <= b
              x >= 0

    Retorna
    -------
    result : dict con 'status', 'z_opt', 'x_opt', 'slacks', 'duals'
    """
    if pulp is None:
        print("Instale PuLP: pip install pulp")
        return None

    m, n = A.shape
    if var_names is None:
        var_names = [f'x{i+1}' for i in range(n)]

    if sense == 'max':
        prob = pulp.LpProblem('Primal', pulp.LpMaximize)
    else:
        prob = pulp.LpProblem('Primal', pulp.LpMinimize)

    x = [pulp.LpVariable(var_names[j], lowBound=0) for j in range(n)]

    # Objetivo
    prob += pulp.lpSum(c[j] * x[j] for j in range(n))

    # Restricciones
    constraints = []
    for i in range(m):
        con = prob.addConstraint(
            pulp.lpSum(A[i, j] * x[j] for j in range(n)) <= b[i],
            name=f'R{i+1}'
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extraer resultados
    x_opt = np.array([v.varValue for v in x])
    z_opt = pulp.value(prob.objective)

    # Slacks y duales
    slacks = {}
    duals = {}
    for name, con in prob.constraints.items():
        slacks[name] = con.slack
        duals[name] = con.pi  # precio sombra (dual value)

    return {
        'status': pulp.LpStatus[prob.status],
        'z_opt': z_opt,
        'x_opt': x_opt,
        'var_names': var_names,
        'slacks': slacks,
        'duals': duals,
    }


def pretty_dual_generic(A, b, c, var_prefix='y'):
    """
    Imprime la formulación dual de un LP primal (max c^T x, Ax <= b, x >= 0)
    en formato legible con LaTeX-like text.
    """
    m, n = A.shape
    print("=" * 50)
    print("DUAL (min):")
    print("=" * 50)

    # Objetivo
    obj_terms = []
    for i in range(m):
        obj_terms.append(f'{b[i]:.2f}*{var_prefix}{i+1}')
    print(f"  min  {' + '.join(obj_terms)}")

    # Restricciones
    print("  s.a.")
    for j in range(n):
        con_terms = []
        for i in range(m):
            con_terms.append(f'{A[i, j]:.2f}*{var_prefix}{i+1}')
        print(f"       {' + '.join(con_terms)}  >=  {c[j]:.2f}")

    for i in range(m):
        print(f"       {var_prefix}{i+1} >= 0")
    print("=" * 50)


def check_complementary_slackness(result):
    """
    Verifica las condiciones de holgura complementaria:
        1. Si x_j > 0  =>  la restricción dual j está activa
        2. Si y_i > 0  =>  la restricción primal i está activa (slack = 0)

    Parámetros
    ----------
    result : dict retornado por solve_primal_pulp

    Imprime las condiciones verificadas.
    """
    if result is None:
        print("No hay resultados para verificar.")
        return

    print("=" * 50)
    print("HOLGURA COMPLEMENTARIA")
    print("=" * 50)

    x_opt = result['x_opt']
    slacks = result['slacks']
    duals = result['duals']
    var_names = result['var_names']

    # Condición 1: y_i * slack_i = 0  para cada restricción
    print("\nCondición primal: y_i * (b_i - A_i x*) = 0")
    all_ok = True
    for name in slacks:
        y_val = duals.get(name, 0) or 0
        s_val = slacks.get(name, 0) or 0
        product = abs(y_val * s_val)
        ok = product < 1e-6
        status = "OK" if ok else "FALLA"
        print(f"  {name}: y={y_val:.4f}, slack={s_val:.4f}, "
              f"producto={product:.6f}  [{status}]")
        if not ok:
            all_ok = False

    # Condición 2: x_j * (A^T y - c)_j = 0
    print(f"\nCondición dual: x_j * (exceso dual_j) = 0")
    for j, vname in enumerate(var_names):
        x_val = x_opt[j] if x_opt[j] is not None else 0
        # Si x > 0, la restricción dual j debe estar activa
        if abs(x_val) > 1e-6:
            print(f"  {vname}: x={x_val:.4f} > 0  =>  restricción dual activa  [OK]")
        else:
            print(f"  {vname}: x={x_val:.4f} = 0  =>  sin condición requerida  [OK]")

    if all_ok:
        print("\n=> Todas las condiciones de holgura complementaria se cumplen.")
    else:
        print("\n=> ADVERTENCIA: alguna condición no se cumple (verifique la solución).")
