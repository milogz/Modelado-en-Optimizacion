"""
InsTSP.py -- Funciones de instancias TSP (self-contained).
Reemplaza la dependencia externa ../Instances/InsTSP.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from pathlib import Path

try:
    from PIL import Image as PILImage
    import glob as _glob
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def generar_datos(n, seed=42, lat_range=(4.5, 5.0), lon_range=(-74.2, -73.9)):
    """
    Genera n coordenadas (lat, lon) aleatorias en un rango dado.
    Retorna DataFrame-like array (n, 2) con columnas [lat, lon].
    """
    rng = np.random.default_rng(seed)
    lats = rng.uniform(*lat_range, size=n)
    lons = rng.uniform(*lon_range, size=n)
    return np.column_stack([lats, lons])


def distancias_tsp(coords):
    """
    Calcula la matriz de distancias euclidianas entre todos los pares.
    coords: array (n, 2)
    Retorna: array (n, n)
    """
    return cdist(coords, coords, metric='euclidean')


def reconstruir_tour(x_vars, n):
    """
    Reconstruye la secuencia del tour a partir de variables x[i,j]
    (diccionario con claves (i,j) y valores 0/1).
    """
    # Build adjacency from active arcs
    adj = {}
    for (i, j), val in x_vars.items():
        if val is not None and val > 0.5:
            adj[i] = j

    tour = [0]
    current = 0
    for _ in range(n - 1):
        nxt = adj.get(current)
        if nxt is None:
            break
        tour.append(nxt)
        current = nxt
    return tour


def visualizar(coords, tour, titulo='Tour TSP', save_path=None):
    """
    Visualiza un tour TSP sobre coordenadas 2D.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 1], coords[:, 0], c='blue', s=40, zorder=5)

    # Draw tour edges
    for k in range(len(tour)):
        i = tour[k]
        j = tour[(k + 1) % len(tour)]
        ax.plot([coords[i, 1], coords[j, 1]],
                [coords[i, 0], coords[j, 0]], 'r-', linewidth=1)

    # Label nodes
    for idx in range(len(coords)):
        ax.annotate(str(idx), (coords[idx, 1], coords[idx, 0]),
                    fontsize=7, ha='center', va='bottom')

    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()


def hacer_gif(carpeta, nombre_gif, duracion=500):
    """
    Crea un GIF a partir de imágenes PNG en una carpeta.
    """
    if not HAS_PIL:
        print("PIL no disponible. Instale: pip install Pillow")
        return

    imagenes = sorted(_glob.glob(str(Path(carpeta) / "*.png")))
    if not imagenes:
        print(f"No se encontraron imágenes en {carpeta}")
        return

    frames = [PILImage.open(img) for img in imagenes]
    frames[0].save(
        str(Path(carpeta) / nombre_gif),
        save_all=True,
        append_images=frames[1:],
        duration=duracion,
        loop=0
    )
    print(f"GIF creado: {Path(carpeta) / nombre_gif}")
