"""
Patch: Fix MO_3-7 MILP.ipynb import cell to use local modules
instead of non-existent ../Instances and ../Ruteo directories.
"""
import json

NB_PATH = "MO_3-7 MILP.ipynb"

# The new import cell that replaces the broken one
NEW_IMPORT_SOURCE = [
    "## Celda de configuracion\n",
    "\n",
    "# Librerias\n",
    "from IPython.display import Image, display\n",
    "import math\n",
    "import time\n",
    "import sys\n",
    "import os\n",
    "import numpy as np\n",
    "\n",
    "# Instancias y algoritmos (modulos locales)\n",
    "from InsTSP import (\n",
    "    generar_datos,\n",
    "    distancias_tsp,\n",
    "    reconstruir_tour,\n",
    "    visualizar,\n",
    "    hacer_gif\n",
    ")\n",
    "\n",
    "# Algoritmos de ruteo (modulos locales)\n",
    "from Lazy_TSP import optimizar_tcl_lazy\n",
    "from Heuristicas import tsp_nn, tsp_cheapest_insertion, tsp_2opt_atsp, tour_cost\n",
]

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell that has the broken imports
patched = False
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        if "InsTSP" in src and "../Instances" in src:
            c["source"] = NEW_IMPORT_SOURCE
            patched = True
            break

if patched:
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"[OK] {NB_PATH} import cell patched successfully")
else:
    print("[WARN] Could not find the broken import cell")
