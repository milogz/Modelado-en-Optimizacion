import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display

def init_busqueda():
  import numpy as np
  import matplotlib.pyplot as plt
  plt.rcParams["figure.figsize"] = (6, 4)
  np.set_printoptions(precision=4, suppress=True)
  return np, plt

from IPython.display import Image, display
from pathlib import Path
import os

def show_fig(nombre, subruta="assets/figs"):
    """
    Busca y muestra una imagen de forma robusta para Jupyter y Colab.
    - nombre: nombre del archivo
    - subruta: ruta relativa esperada dentro del proyecto
    """
    # 1️⃣ posibles ubicaciones
    posibles_rutas = [
        Path(nombre),  # mismo directorio
        Path.cwd() / nombre,  # cwd directo
        Path.cwd() / subruta / nombre,  # estructura local tipo repo
        Path("/content") / nombre,  # raíz de colab
        Path("/content") / subruta / nombre,  # colab con estructura repo
    ]

    # 2️⃣ buscar primera ruta existente
    ruta_final = next((p for p in posibles_rutas if p.exists()), None)

    # 3️⃣ mostrar resultado
    if ruta_final:
        #print(f"✅ Imagen encontrada en: {ruta_final}")
        display(Image(filename=str(ruta_final)))
    else:
        print(f"⚠️ No se encontró {nombre} en las rutas esperadas: {posibles_rutas}")


def plot_1d_paths(func, paths_dict, xlim=(-5, 5)):
    xs_curve = np.linspace(*xlim, 400)
    plt.plot(xs_curve, func(xs_curve), lw=2, label="$f(x)$")
    markers = ["o", "x", "^", "s", "D", "v"]
    for i, (lbl, path) in enumerate(paths_dict.items()):
        path = np.asarray(path).ravel()
        plt.plot(path, func(path), lw=1.2, marker=markers[i % len(markers)], label=lbl)
    plt.xlabel("x"); plt.ylabel("f(x)")
    plt.title("Trayectorias 1-D"); plt.legend(); plt.grid(alpha=.3); plt.show()


def plot_convergence(hist_dict, logy=True, ylabel="f"):
    for lbl, fvals in hist_dict.items():
        fvals = np.asarray(fvals)
        if logy:
            fvals = np.maximum(fvals, 1e-15)
        plt.plot(fvals, label=lbl)
    if logy:
        plt.yscale("log")
    plt.xlabel("Iter"); plt.ylabel(ylabel)
    plt.title("Convergencia"); plt.legend(); plt.grid(alpha=.3); plt.show()


def plot_2d_contour(func, paths_dict, span=((-2,2), (-2,2)), levels=25):
    gx = np.linspace(*span[0], 200)
    gy = np.linspace(*span[1], 200)
    X, Y = np.meshgrid(gx, gy)
    Z = func((X, Y))
    plt.contour(X, Y, Z, levels=levels, alpha=.6)
    markers = ["o", "x", "^", "s", "D", "v"]
    for i, (lbl, path) in enumerate(paths_dict.items()):
        P = np.asarray(path)
        plt.plot(P[:, 0], P[:, 1], marker=markers[i % len(markers)], ms=4, label=lbl)
    plt.legend(); plt.gca().set_aspect("equal")
    plt.title("Trayectorias en plano x–y"); plt.grid(alpha=.2); plt.show()

def lift_1d(f, df):
    """
    Adapta funciones 1D (escalares) a interfaz vectorial (n=1).
    Retorna: f_vec(X), grad_vec(X) con X shape (1,)
    """
    def f_vec(X):
        x = float(np.asarray(X).ravel()[0])
        return f(x)

    def grad_vec(X):
        x = float(np.asarray(X).ravel()[0])
        return np.array([df(x)], dtype=float)
    return f_vec, grad_vec

def tabla_comp():    

    html_table = """
    <table style="border-collapse: collapse; width: 100%; text-align: center;">
      <tr style="background-color: #f0f0f0;">
        <th></th>
        <th>No lineal sin restricciones</th>
        <th>Lineal con restricciones (LP)</th>
        <th>No lineal con restricciones (NLP)</th>
        <th>Entero lineal (MIP)</th>
      </tr>
      <tr>
        <td style="font-weight: bold;">Forma general</td>
        <td>\\(\\min f_{\\text{nonlin}}(x), \\; x \\in \\mathbb{R}^n\\)</td>
        <td>\\(\\min f(x) = c^T x, \\; A x \\le b, \\; x \\in \\mathbb{R}^n\\)</td>
        <td>\\(\\min f_{\\text{nl con restr}}(x), \\; g(x) \\le b, \\; x \\in \\mathbb{R}^n\\)</td>
        <td>\\(\\min f(x) = c^T x, \\; A x \\le b, \\; x \\in \\mathbb{Z}^n\\)</td>
      </tr>
      <tr>
        <td style="font-weight: bold;">Ejemplos de aplicación</td>
        <td>Ajuste de parámetros sin restricciones (estadística, <i>machine learning</i>, métodos numéricos)</td>
        <td>Mezclas, asignación, planeación de producción, manejo de inventarios</td>
        <td>Diseño de ingeniería con procesos físicos, portafolios con riesgo no lineal</td>
        <td><i>Scheduling</i>, ruteo, asignación entera</td>
      </tr>
      <tr>
        <td style="font-weight: bold;">Métodos típicos de solución</td>
        <td>Gradiente y sus variantes, heurísticas y aproximaciones numéricas</td>
        <td>Simplex, Punto Interior</td>
        <td>Programación cuadrática o convexa (<i>via solvers</i>), métodos numéricos</td>
        <td>Branch & Bound (<i>via solvers</i>)</td>
      </tr>
    </table>
    """
    display(HTML(html_table))
