"""
Heuristicas.py -- Algoritmos heurísticos para TSP (self-contained).
Reemplaza la dependencia externa ../Ruteo/Heuristicas.
"""
import numpy as np


def tour_cost(tour, D):
    """Calcula el costo total de un tour dado la matriz de distancias D."""
    return sum(D[tour[k]][tour[(k + 1) % len(tour)]] for k in range(len(tour)))


def tsp_nn(D, start=0):
    """
    Heurística del Vecino Más Cercano (Nearest Neighbor).
    Retorna: tour (lista de nodos)
    """
    n = len(D)
    visited = [False] * n
    tour = [start]
    visited[start] = True

    for _ in range(n - 1):
        last = tour[-1]
        nearest = None
        best_dist = float('inf')
        for j in range(n):
            if not visited[j] and D[last][j] < best_dist:
                best_dist = D[last][j]
                nearest = j
        tour.append(nearest)
        visited[nearest] = True

    return tour


def tsp_cheapest_insertion(D):
    """
    Heurística de Inserción Más Barata (Cheapest Insertion).
    Retorna: tour (lista de nodos)
    """
    n = len(D)
    if n <= 2:
        return list(range(n))

    # Start with the two farthest nodes
    max_dist = -1
    i0, j0 = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if D[i][j] > max_dist:
                max_dist = D[i][j]
                i0, j0 = i, j

    tour = [i0, j0]
    in_tour = set(tour)

    while len(tour) < n:
        # Find the node not in tour with cheapest insertion cost
        best_cost = float('inf')
        best_node = None
        best_pos = None

        for node in range(n):
            if node in in_tour:
                continue
            for pos in range(len(tour)):
                i = tour[pos]
                j = tour[(pos + 1) % len(tour)]
                cost = D[i][node] + D[node][j] - D[i][j]
                if cost < best_cost:
                    best_cost = cost
                    best_node = node
                    best_pos = pos + 1

        tour.insert(best_pos, best_node)
        in_tour.add(best_node)

    return tour


def tsp_2opt_atsp(tour, D, max_iter=1000):
    """
    Mejora un tour existente con 2-OPT (también funciona para ATSP asimétrico).
    Retorna: tour mejorado (lista de nodos)
    """
    tour = list(tour)
    n = len(tour)
    improved = True
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Calculate cost change
                old_cost = (D[tour[i - 1]][tour[i]] +
                           D[tour[j]][tour[(j + 1) % n]])
                new_cost = (D[tour[i - 1]][tour[j]] +
                           D[tour[i]][tour[(j + 1) % n]])
                if new_cost < old_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True

    return tour
