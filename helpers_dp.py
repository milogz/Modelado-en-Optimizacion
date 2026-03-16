"""
helpers_dp.py – Funciones de apoyo para el notebook MO_3-8 DP.

Actualización 2026: Programación Dinámica como puente hacia
evaluación de políticas y Reinforcement Learning.
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import pulp
except ImportError:
    pulp = None


# ────────────────────────────────────────────────────────────────────
# 1. Crear red de etapas (Stagecoach Problem)
# ────────────────────────────────────────────────────────────────────

def crear_red_etapas():
    """
    Genera un grafo dirigido acíclico (DAG) por etapas,
    clásico problema del diligente (*stagecoach problem*).

    Retorna
    -------
    grafo : dict
        grafo[nodo] = [(vecino, costo), ...]
    etapas : list[list[str]]
        lista de listas con los nodos de cada etapa.
    pos : dict
        posiciones (x,y) para visualización.
    """
    grafo = {
        'A':  [('B1', 2), ('B2', 4), ('B3', 3)],
        'B1': [('C1', 7), ('C2', 4), ('C3', 6)],
        'B2': [('C1', 3), ('C2', 2), ('C3', 4)],
        'B3': [('C1', 4), ('C2', 1), ('C3', 5)],
        'C1': [('D1', 1), ('D2', 4)],
        'C2': [('D1', 6), ('D2', 3)],
        'C3': [('D1', 3), ('D2', 2)],
        'D1': [('E', 1)],
        'D2': [('E', 4)],
        'E':  [],
    }
    etapas = [['A'], ['B1', 'B2', 'B3'], ['C1', 'C2', 'C3'], ['D1', 'D2'], ['E']]
    pos = {}
    for t, nodos in enumerate(etapas):
        n = len(nodos)
        for k, nodo in enumerate(nodos):
            y = (n - 1) / 2 - k
            pos[nodo] = (t, y)
    return grafo, etapas, pos


# ────────────────────────────────────────────────────────────────────
# 2. Inducción hacia atrás
# ────────────────────────────────────────────────────────────────────

def dp_backward(grafo, etapas):
    """
    Resuelve el problema de camino más corto por etapas mediante
    inducción hacia atrás (backward recursion).

    Retorna
    -------
    V : dict   – función de valor V[nodo] = costo-mínimo-al-destino
    pi : dict  – política óptima pi[nodo] = siguiente nodo
    """
    V = {}
    pi = {}

    # Nodo terminal
    destino = etapas[-1][0]
    V[destino] = 0

    # Recorrer etapas de atrás hacia adelante
    for t in range(len(etapas) - 2, -1, -1):
        for s in etapas[t]:
            mejor_costo = float('inf')
            mejor_accion = None
            for (s_next, c) in grafo[s]:
                costo = c + V[s_next]
                if costo < mejor_costo:
                    mejor_costo = costo
                    mejor_accion = s_next
            V[s] = mejor_costo
            pi[s] = mejor_accion

    return V, pi


def reconstruir_camino(pi, origen='A', destino='E'):
    """Recuerda el camino óptimo a partir de la política."""
    camino = [origen]
    nodo = origen
    while nodo != destino:
        nodo = pi[nodo]
        camino.append(nodo)
    return camino


# ────────────────────────────────────────────────────────────────────
# 3. Visualización
# ────────────────────────────────────────────────────────────────────

def visualizar_red(grafo, pos, politica=None, V=None, titulo='Red por etapas'):
    """
    Dibuja el grafo dirigido con costos en los arcos.
    Si se proporciona `politica`, resalta el camino óptimo.
    Si se proporciona `V`, muestra los valores de la función V(s).
    """
    if nx is None:
        print("Instale networkx: pip install networkx")
        return

    G = nx.DiGraph()
    edge_labels = {}
    for s, vecinos in grafo.items():
        for (s2, c) in vecinos:
            G.add_edge(s, s2, weight=c)
            edge_labels[(s, s2)] = str(c)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Dibujar todos los arcos
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray',
                           connectionstyle='arc3,rad=0.05', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                  font_size=9, ax=ax)

    # Resaltar camino óptimo
    if politica:
        camino = reconstruir_camino(politica)
        opt_edges = list(zip(camino[:-1], camino[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=opt_edges, ax=ax,
                               edge_color='red', width=2.5, arrows=True,
                               connectionstyle='arc3,rad=0.05')

    # Etiquetas de nodos
    labels = {}
    for nodo in G.nodes():
        lbl = nodo
        if V is not None and nodo in V:
            lbl += f"\nV={V[nodo]}"
        labels[nodo] = lbl

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightyellow',
                           edgecolors='black', node_size=900)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)

    ax.set_title(titulo, fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────────
# 4. DP → LP ("achatamiento")
# ────────────────────────────────────────────────────────────────────

def dp_to_lp(grafo, origen='A', destino='E'):
    """
    Formula el problema de camino más corto como un LP (flujo de costo mínimo)
    y lo resuelve con PuLP.

    Retorna
    -------
    z_lp : float               – costo óptimo
    flujos : dict[(s,s'), val]  – variables de flujo óptimo
    """
    if pulp is None:
        print("Instale PuLP: pip install pulp")
        return None, None

    # Recopilar todos los arcos
    arcos = []
    for s, vecinos in grafo.items():
        for (s2, c) in vecinos:
            arcos.append((s, s2, c))

    nodos = list(grafo.keys())

    prob = pulp.LpProblem('CaminoCorto_LP', pulp.LpMinimize)

    # Variables: flujo por cada arco (continuas >= 0)
    x = {}
    for (s, s2, c) in arcos:
        x[(s, s2)] = pulp.LpVariable(f'x_{s}_{s2}', lowBound=0, cat='Continuous')

    # Función objetivo: minimizar costo total
    prob += pulp.lpSum(c * x[(s, s2)] for (s, s2, c) in arcos)

    # Restricciones de conservación de flujo
    for nodo in nodos:
        flujo_salida = pulp.lpSum(x[(s, s2)] for (s, s2, _) in arcos if s == nodo)
        flujo_entrada = pulp.lpSum(x[(s, s2)] for (s, s2, _) in arcos if s2 == nodo)

        if nodo == origen:
            prob += flujo_salida - flujo_entrada == 1, f'flujo_{nodo}'
        elif nodo == destino:
            prob += flujo_salida - flujo_entrada == -1, f'flujo_{nodo}'
        else:
            prob += flujo_salida - flujo_entrada == 0, f'flujo_{nodo}'

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    z_lp = pulp.value(prob.objective)
    flujos = {k: v.varValue for k, v in x.items() if v.varValue and v.varValue > 0.5}

    return z_lp, flujos
