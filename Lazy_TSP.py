"""
Lazy_TSP.py -- Formulación MIP "lazy" del TSP (self-contained).
Reemplaza la dependencia externa ../Ruteo/Lazy_TSP.
"""
import numpy as np

try:
    import pulp
except ImportError:
    pulp = None


def optimizar_tcl_lazy(D, time_limit=60):
    """
    Resuelve el TSP usando formulación MIP con eliminación de subtour MTZ.
    D: matriz de distancias (n x n)
    time_limit: límite de tiempo en segundos

    Retorna: (tour, z_opt, tiempo, status)
    """
    if pulp is None:
        print("Instale PuLP: pip install pulp")
        return None, None, None, "PuLP no disponible"

    import time

    n = len(D)

    prob = pulp.LpProblem('TSP_Lazy', pulp.LpMinimize)

    # Variables binarias x[i,j]
    x = pulp.LpVariable.dicts('x',
        ((i, j) for i in range(n) for j in range(n) if i != j),
        cat='Binary')

    # Variables MTZ u[i]
    u = pulp.LpVariable.dicts('u', range(1, n),
        lowBound=1, upBound=n - 1, cat='Continuous')

    # Objetivo
    prob += pulp.lpSum(D[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)

    # Restricciones de asignación
    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(n) if j != i) == 1
        prob += pulp.lpSum(x[j, i] for j in range(n) if j != i) == 1

    # MTZ subtour elimination
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + (n - 1) * x[i, j] <= n - 2

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    t_elapsed = time.perf_counter() - t0

    status = pulp.LpStatus[prob.status]
    z_opt = pulp.value(prob.objective)

    # Reconstruir tour
    if status == 'Optimal':
        adj = {}
        for (i, j), var in x.items():
            if var.varValue is not None and var.varValue > 0.5:
                adj[i] = j
        tour = [0]
        current = 0
        for _ in range(n - 1):
            nxt = adj.get(current)
            if nxt is None:
                break
            tour.append(nxt)
            current = nxt
    else:
        tour = list(range(n))

    return tour, z_opt, t_elapsed, status
