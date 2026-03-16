"""
Patch script for MO_3-2 Estimacion.ipynb
Actualización 2026: Optimización como motor de ML, función de pérdida,
clasificación logística, preguntas de reflexión, citación IA.
"""
import json

NB_PATH = "MO_3-2 Estimacion.ipynb"

def md(source_lines):
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines if isinstance(source_lines, list) else [source_lines]}

def code(source_lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines if isinstance(source_lines, list) else [source_lines]}

# ── New cells ──

traceability = md([
    "> **Actualización 2026:** Se integra enfoque de *optimización como motor de ML*, \n",
    "> función de pérdida y clasificación, manteniendo la base de modelamiento original.\n",
])

section_ml_engine = md([
    "---\n",
    "## La Optimización como Motor de Machine Learning\n",
    "\n",
    "En la sección anterior vimos cómo el *descenso de gradiente* minimiza el error cuadrático para ajustar una recta. \n",
    "Ahora generalizamos esta idea: **todo modelo de ML supervisado se entrena minimizando una función de pérdida**.\n",
    "\n",
    "Formalmente, dada una muestra de entrenamiento $\\{(\\mathbf{x}_i, y_i)\\}_{i=1}^n$ y un modelo parametrizado por $\\boldsymbol{\\theta}$, \n",
    "el entrenamiento consiste en resolver:\n",
    "\n",
    "$$\n",
    "\\min_{\\boldsymbol{\\theta}} \\; \\mathcal{L}(\\boldsymbol{\\theta}) \n",
    "= \\frac{1}{n}\\sum_{i=1}^{n}\\ell\\bigl(y_i,\\;\\hat{y}_i(\\boldsymbol{\\theta})\\bigr)\n",
    "$$\n",
    "\n",
    "donde $\\ell$ es la **función de pérdida individual** (p.ej. error cuadrático para regresión, cross-entropy para clasificación).\n",
    "\n",
    "El *descenso de gradiente* es simplemente una **técnica de búsqueda en la superficie** $\\mathcal{L}(\\boldsymbol{\\theta})$ — \n",
    "similar a la búsqueda local de la Lección 3.1, pero usando la información del gradiente para elegir la dirección de movimiento.\n",
])

code_loss_viz = code([
    "# Visualización de la superficie de pérdida MSE con trayectoria GD\n",
    "from helpers_estimacion import plot_loss_surface\n",
    "\n",
    "# Usar los datos del ejemplo ilustrativo\n",
    "x_data = datos['Años de experiencia (x)'].values\n",
    "y_data = datos['Salario normalizado (y)'].values\n",
    "\n",
    "# Ejecutar GD y guardar trayectoria\n",
    "a_gd, b_gd = 0.0, 0.0   # punto inicial\n",
    "alpha_gd = 0.0001         # learning rate (pequeño para estabilidad)\n",
    "path_gd = [(a_gd, b_gd)]\n",
    "\n",
    "n_pts = len(x_data)\n",
    "for step in range(300):\n",
    "    pred = a_gd + b_gd * x_data\n",
    "    error = pred - y_data\n",
    "    grad_a = (2/n_pts) * np.sum(error)\n",
    "    grad_b = (2/n_pts) * np.sum(error * x_data)\n",
    "    a_gd -= alpha_gd * grad_a\n",
    "    b_gd -= alpha_gd * grad_b\n",
    "    path_gd.append((a_gd, b_gd))\n",
    "\n",
    "plot_loss_surface(x_data, y_data, path=path_gd)\n",
])

reflection_convexity = md([
    "---\n",
    "### Pregunta de Reflexión\n",
    "\n",
    "1. La superficie de pérdida MSE es **convexa** (tiene forma de \"tazón\"). ¿Por qué esto garantiza que el descenso de gradiente encuentre el **mínimo global**?\n",
    "2. ¿Qué ocurriría si la función de pérdida tuviera múltiples mínimos locales?\n",
])

section_classification = md([
    "---\n",
    "## Clasificación como Optimización\n",
    "\n",
    "La regresión no es el único problema de estimación. En **clasificación binaria**, queremos predecir una etiqueta $y \\in \\{0, 1\\}$ \n",
    "a partir de características $\\mathbf{x}$. El modelo calcula:\n",
    "\n",
    "$$\\hat{y} = \\sigma(\\boldsymbol{\\theta}^T \\mathbf{x}) = \\frac{1}{1 + e^{-\\boldsymbol{\\theta}^T \\mathbf{x}}}$$\n",
    "\n",
    "y se entrena minimizando la **función de pérdida logística** (*cross-entropy binaria*):\n",
    "\n",
    "$$\\ell(y, \\hat{y}) = -\\bigl[y\\log(\\hat{y}) + (1-y)\\log(1-\\hat{y})\\bigr]$$\n",
    "\n",
    "Esto es, nuevamente, un **problema de optimización no lineal sin restricciones**, resuelto con descenso de gradiente — \n",
    "exactamente el mismo esquema iterativo $\\boldsymbol{\\theta}^{(t+1)} = \\boldsymbol{\\theta}^{(t)} - \\alpha \\nabla \\mathcal{L}$.\n",
])

code_logistic = code([
    "from helpers_estimacion import sigmoid, gd_logistic, plot_decision_boundary\n",
    "\n",
    "# Dataset sintético de 2 clases\n",
    "np.random.seed(7)\n",
    "n_class = 100\n",
    "X0 = np.random.randn(n_class, 2) + np.array([1, 1])\n",
    "X1 = np.random.randn(n_class, 2) + np.array([4, 4])\n",
    "X_raw = np.vstack([X0, X1])\n",
    "y_cls = np.array([0]*n_class + [1]*n_class)\n",
    "\n",
    "# Agregar columna de 1s (intercepto)\n",
    "X_cls = np.c_[np.ones(2*n_class), X_raw]\n",
    "\n",
    "# Entrenamiento con GD\n",
    "theta_opt, losses, _ = gd_logistic(X_cls, y_cls, lr=0.1, epochs=200)\n",
    "\n",
    "# Visualizar convergencia\n",
    "plt.plot(losses); plt.xlabel('Iteración'); plt.ylabel('Cross-entropy')\n",
    "plt.title('Convergencia del descenso de gradiente (logístico)'); plt.grid(alpha=0.3); plt.show()\n",
    "\n",
    "# Visualizar frontera de decisión\n",
    "plot_decision_boundary(X_cls, y_cls, theta_opt)\n",
])

reflection_logistic = md([
    "---\n",
    "### Pregunta de Reflexión\n",
    "\n",
    "1. La función de pérdida logística *también* es convexa. ¿Qué implica esto para la convergencia del descenso de gradiente?\n",
    "2. ¿Qué similitudes y diferencias observa entre el GD para regresión y el GD para clasificación?\n",
    "3. ¿Cómo se conecta este enfoque con modelos más complejos como las *redes neuronales*?\n",
])

ia_citation = md([
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

    # 1. Insert traceability at position 0
    cells.insert(0, traceability)

    # 2. Find section "4 - Ejemplo con datos reales" and insert ML-engine section before it
    target_idx = None
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if "4 - Ejemplo con datos reales" in src or "### 4" in src:
            target_idx = i
            break

    insert_cells = [section_ml_engine, code_loss_viz, reflection_convexity,
                    section_classification, code_logistic, reflection_logistic]

    if target_idx is not None:
        for offset, cell in enumerate(insert_cells):
            cells.insert(target_idx + offset, cell)
    else:
        # Fallback: insert before last cell
        for cell in insert_cells:
            cells.insert(-1, cell)

    # 3. Append IA citation at end
    cells.append(ia_citation)

    nb["cells"] = cells
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"[OK] {NB_PATH} patched successfully ({len(cells)} cells)")

if __name__ == "__main__":
    patch()
