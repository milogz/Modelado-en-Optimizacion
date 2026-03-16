"""
Patch script for MO_3-7 MILP.ipynb
Actualización 2026: Inyecta celdas de trazabilidad, comparación solver-heurística,
preguntas de reflexión y citación IA.
"""
import json, copy, sys

NB_PATH = "MO_3-7 MILP.ipynb"

def md(source_lines):
    """Helper: create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines if isinstance(source_lines, list) else [source_lines],
    }

def code(source_lines, outputs=None):
    """Helper: create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source_lines if isinstance(source_lines, list) else [source_lines],
    }

# ── Cells to inject ──────────────────────────────────────────────────────────

traceability_cell = md([
    "> **Actualización 2026:** Se integra enfoque de *Heurísticas vs. Solvers Exactos* \n",
    "> (trade-off calidad–tiempo–recursos computacionales), \n",
    "> manteniendo la base de modelamiento original.\n",
])

reflections_bb = md([
    "---\n",
    "### 🤔 Pregunta de Reflexión\n",
    "\n",
    "El algoritmo *Branch & Bound* garantiza encontrar la solución óptima, pero su complejidad en el peor caso es **exponencial** ($O(2^n)$ para variables binarias).\n",
    "\n",
    "1. ¿Por qué el tiempo de cómputo crece de forma exponencial con el número de variables enteras?\n",
    "2. ¿En qué situaciones prácticas sería **inaceptable** esperar la respuesta exacta?\n",
])

section_tradeoff_intro = md([
    "---\n",
    "## 🔄 Trade-off: Solver Exacto vs. Heurística\n",
    "\n",
    "En problemas combinatorios (como el TSP o el *Knapsack*), los **solvers exactos** basados en *Branch & Bound* garantizan encontrar la solución óptima $z^*$, pero su tiempo de cómputo crece **exponencialmente** con el tamaño de la instancia.\n",
    "\n",
    "Las **heurísticas** sacrifican la garantía de optimalidad a cambio de soluciones *buenas* en tiempos mucho menores. Medimos la calidad con el **gap de optimalidad**:\n",
    "\n",
    "$$\n",
    "\\text{gap}(\\%) = \\frac{z_{\\text{heur}} - z^*}{z^*} \\times 100\\%\n",
    "$$\n",
    "\n",
    "A continuación, compararemos **explícitamente** un solver MIP (PuLP/CBC) contra una heurística simple (2-OPT) en la misma instancia, midiendo tiempo y gap.\n",
])

code_comparison = code([
    "import time\n",
    "import pulp\n",
    "\n",
    "# --- Seleccionar un subconjunto pequeño para que el MIP termine ---\n",
    "N_COMP = 12  # Intente con 15, 18, ... y observe el crecimiento del tiempo\n",
    "np.random.seed(42)\n",
    "subset_idx = np.random.choice(len(coords), size=N_COMP, replace=False)\n",
    "coords_sub = coords[subset_idx]\n",
    "\n",
    "# Matriz de distancias\n",
    "from scipy.spatial.distance import cdist\n",
    "D = cdist(coords_sub, coords_sub)\n",
    "\n",
    "# ── 1. Resolver con MIP (PuLP) ──────────────────────────────────\n",
    "t0 = time.perf_counter()\n",
    "\n",
    "prob = pulp.LpProblem('TSP_exact', pulp.LpMinimize)\n",
    "n = N_COMP\n",
    "x = pulp.LpVariable.dicts('x', ((i, j) for i in range(n) for j in range(n) if i != j),\n",
    "                           cat='Binary')\n",
    "u = pulp.LpVariable.dicts('u', range(1, n), lowBound=1, upBound=n-1, cat='Continuous')\n",
    "\n",
    "# Función objetivo\n",
    "prob += pulp.lpSum(D[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)\n",
    "\n",
    "# Restricciones de asignación\n",
    "for i in range(n):\n",
    "    prob += pulp.lpSum(x[i, j] for j in range(n) if j != i) == 1\n",
    "    prob += pulp.lpSum(x[j, i] for j in range(n) if j != i) == 1\n",
    "\n",
    "# MTZ subtour elimination\n",
    "for i in range(1, n):\n",
    "    for j in range(1, n):\n",
    "        if i != j:\n",
    "            prob += u[i] - u[j] + (n - 1) * x[i, j] <= n - 2\n",
    "\n",
    "prob.solve(pulp.PULP_CBC_CMD(msg=0))\n",
    "t_mip = time.perf_counter() - t0\n",
    "z_mip = pulp.value(prob.objective)\n",
    "\n",
    "print(f'MIP  │ z* = {z_mip:.2f}  │ t = {t_mip:.3f} s')\n",
    "\n",
    "# ── 2. Resolver con Heurística NN + 2-OPT ──────────────────────\n",
    "def nearest_neighbor(D):\n",
    "    n = len(D)\n",
    "    visited = [False]*n\n",
    "    tour = [0]; visited[0] = True\n",
    "    for _ in range(n-1):\n",
    "        last = tour[-1]\n",
    "        nearest = min((j for j in range(n) if not visited[j]), key=lambda j: D[last][j])\n",
    "        tour.append(nearest); visited[nearest] = True\n",
    "    return tour\n",
    "\n",
    "def two_opt(tour, D):\n",
    "    improved = True\n",
    "    while improved:\n",
    "        improved = False\n",
    "        for i in range(1, len(tour)-1):\n",
    "            for j in range(i+1, len(tour)):\n",
    "                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]\n",
    "                old_cost = sum(D[tour[k]][tour[(k+1)%len(tour)]] for k in range(len(tour)))\n",
    "                new_cost = sum(D[new_tour[k]][new_tour[(k+1)%len(new_tour)]] for k in range(len(new_tour)))\n",
    "                if new_cost < old_cost - 1e-10:\n",
    "                    tour = new_tour\n",
    "                    improved = True\n",
    "    return tour\n",
    "\n",
    "t0 = time.perf_counter()\n",
    "tour_heur = nearest_neighbor(D)\n",
    "tour_heur = two_opt(tour_heur, D)\n",
    "t_heur = time.perf_counter() - t0\n",
    "z_heur = sum(D[tour_heur[k]][tour_heur[(k+1)%len(tour_heur)]] for k in range(len(tour_heur)))\n",
    "\n",
    "gap = (z_heur - z_mip) / z_mip * 100 if z_mip > 0 else 0\n",
    "print(f'Heur │ z  = {z_heur:.2f}  │ t = {t_heur:.3f} s │ gap = {gap:.2f}%')\n",
])

code_comparison_plot = code([
    "# ── Visualización comparativa ────────────────────────────────────\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "\n",
    "# Gráfico de barras: Tiempo\n",
    "axes[0].bar(['MIP (CBC)', 'NN + 2-OPT'], [t_mip, t_heur],\n",
    "            color=['steelblue', 'coral'])\n",
    "axes[0].set_ylabel('Tiempo (s)')\n",
    "axes[0].set_title('Comparación de Tiempo de Cómputo')\n",
    "\n",
    "# Gráfico de barras: FO + gap\n",
    "axes[1].bar(['MIP (óptimo)', 'NN + 2-OPT'], [z_mip, z_heur],\n",
    "            color=['steelblue', 'coral'])\n",
    "axes[1].set_ylabel('Distancia total')\n",
    "axes[1].set_title(f'Función Objetivo (gap = {gap:.2f}%)')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
])

reflection_tradeoff = md([
    "---\n",
    "### 🤔 Pregunta de Reflexión\n",
    "\n",
    "1. Cambie `N_COMP` a 15, luego a 18. ¿Cómo escala el tiempo del MIP vs. el de la heurística?\n",
    "2. ¿En qué punto el gap de la heurística se vuelve *aceptable* comparado con la ganancia de tiempo?\n",
    "3. ¿Cuándo es preferible una heurística sobre un solver exacto en la industria?\n",
])

reflection_heur_general = md([
    "---\n",
    "### 🤔 Pregunta de Reflexión\n",
    "\n",
    "1. ¿Por qué la heurística 2-OPT suele mejorar la solución del vecino más cercano (NN)?\n",
    "2. ¿Es posible que 2-OPT quede atrapado en un **óptimo local**? ¿Cómo se relaciona este concepto con la búsqueda local vista en la Lección 3.1?\n",
])

ia_citation_cell = md([
    "---\n",
    "> **Política de uso de IA (artículos 80-82 del syllabus):** \n",
    "> Si utiliza herramientas de IA generativa (ChatGPT, Copilot, Gemini, etc.) para apoyar sus entregas, \n",
    "> debe citar la herramienta utilizada y el *prompt* empleado. \n",
    "> El incumplimiento de esta política será tratado como falta académica.\n",
])


def patch():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # ── 1. Insert traceability cell at position 0 ──
    cells.insert(0, traceability_cell)

    # ── 2. Find "## Heuristicas" section and insert comparison BEFORE it ──
    heur_idx = None
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if "## Heuristicas" in src or "## Heurísticas" in src or "## Heuristicas" in src:
            heur_idx = i
            break

    if heur_idx is not None:
        # Insert comparison section right before the heuristics section
        insert_cells = [
            section_tradeoff_intro,
            code_comparison,
            code_comparison_plot,
            reflection_tradeoff,
        ]
        for offset, cell in enumerate(insert_cells):
            cells.insert(heur_idx + offset, cell)
    else:
        print("WARNING: Could not find '## Heuristicas' section — appending comparison at end")
        cells.extend([section_tradeoff_intro, code_comparison, code_comparison_plot, reflection_tradeoff])

    # ── 3. Insert B&B reflection after "Modelo Clásico" section ──
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if "### Modelo Cl" in src:
            # Insert after the next code cell following this markdown
            for j in range(i+1, min(i+5, len(cells))):
                if cells[j]["cell_type"] == "code":
                    cells.insert(j+1, reflections_bb)
                    break
            break

    # ── 4. Insert general heuristic reflection near the end of heuristics section ──
    # Find "## Resultados" and insert reflection just before it
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if "## Resultados" in src:
            cells.insert(i, reflection_heur_general)
            break

    # ── 5. Append IA citation cell at the very end ──
    cells.append(ia_citation_cell)

    nb["cells"] = cells

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"[OK] {NB_PATH} patched successfully ({len(cells)} cells)")


if __name__ == "__main__":
    patch()
