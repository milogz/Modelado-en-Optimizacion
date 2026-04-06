"""
Script to add dashboard references to existing notebooks.
Adds a callout markdown cell right after the title cell in each notebook.
"""
import json, sys

def add_dashboard_ref(nb_path, ref_text):
    """Insert a markdown cell with ref_text after the first cell."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    # Check if reference already exists
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            src = "".join(cell["source"])
            if "Laboratorio interactivo" in src or "Dashboard Lab" in src:
                print(f"  ⏭ Reference already present in {nb_path}, skipping.")
                return
    
    new_cell = {
        "cell_type": "markdown",
        "id": "dashboard-ref",
        "metadata": {},
        "source": [ref_text]
    }
    nb["cells"].insert(1, new_cell)  # after the title cell
    
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  ✅ Added dashboard reference to {nb_path}")


# --- MO_3-1 ---
add_dashboard_ref(
    "MO_3-1 Busqueda.ipynb",
    "> 🧪 **Laboratorio interactivo**: Para experimentar directamente con los conceptos de esta lección "
    "(parámetros de vecindario, tasa de aprendizaje, comparación de convergencia), abre el "
    "**Dashboard Lab 3.1**: `MO_Lab_3-1_Busqueda_Dashboard.ipynb`."
)

# --- MO_3-3 ---
add_dashboard_ref(
    "MO_3-3 LP (Bases).ipynb",
    "> 🧪 **Laboratorio interactivo**: Para experimentar con holguras, restricciones activas, "
    "bases y puntos extremos de forma interactiva, abre el "
    "**Dashboard Lab 3.3**: `MO_Lab_3-3_LP_Dashboard.ipynb` (Tab 1: Factibilidad y Holguras)."
)

# --- MO_3-4 ---
add_dashboard_ref(
    "MO_3-4 LP (Simplex).ipynb",
    "> 🧪 **Laboratorio interactivo**: Para explorar el método Simplex paso a paso "
    "(dirección, longitud, costos reducidos, pivoteo), abre el "
    "**Dashboard Lab 3.3**: `MO_Lab_3-3_LP_Dashboard.ipynb` (Tab 2: Simplex Paso a Paso)."
)

print("\nDone.")
