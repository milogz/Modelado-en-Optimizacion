"""
Patch script for MO_3-0 Intro Metodos Optimizacion.ipynb
Adds traceability cell and updates module structure to include DP.
"""
import json

NB_PATH = "MO_3-0 Intro Metodos Optimizacion.ipynb"

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src if isinstance(src, list) else [src]}

traceability = md([
    "> **Actualización 2026:** Se actualiza la estructura del módulo para incluir \n",
    "> Programación Dinámica (DP) como puente hacia Reinforcement Learning.\n",
])

def patch():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    # 1. Insert traceability at position 0
    cells.insert(0, traceability)

    # 2. Find "Estructura del módulo" section and update it to include DP
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        if "Estructura del m" in src and "módulo" in src:
            # Replace the content to include DP block
            c["source"] = [
                "### Estructura del módulo\n",
                "\n",
                "Durante este módulo, conoceremos estrategias de solución para problemas de optimización comunes en la práctica: desde encontrar el mínimo de una función, hasta resolver problemas de decisión continuos y discretos, en casos lineales y no lineales. Específicamente, nos concentraremos en los siguientes cuatro bloques.\n",
                "\n",
                "1. **Optimización sin restricciones** (Semana 9): Contrastaremos procedimientos basados en vecindarios (búsqueda local) y basados en derivadas (búsqueda de gradiente) para optimizar funciones no lineales sin restricciones, introduciendo nociones de *óptimo local/global*, *incumbente*, y *convergencia* (**Lección 3.1**). Estudiaremos el uso del *descenso de gradiente* para la optimización en problemas de regresión y clasificación, introduciendo la noción de *función de pérdida* como base para los problemas de estimación en *machine learning* e inteligencia artificial (**Lección 3.2**).\n",
                "   \n",
                "2. **Optimización lineal con restricciones** (Semanas 10 y 11): Retomaremos problemas de decisión como los cubiertos en la primera parte del curso, e introduciremos conceptos de *programación lineal* (y método Simplex) como estrategia vigente y base para análisis económico y métodos avanzados (**Lecciones 3.3, 3.4, y 3.5**). Como complemento (opcional), tendremos una lección introductoria que extiende las nociones al caso de la optimización no lineal (**Lección 3.6**), presentando problemas comunes y herramientas de solución disponibles.\n",
                "   \n",
                "3. **Optimización en problemas de decisiones discretas** (Semana 12): Exploraremos los retos asociados a la solución de problemas con variables enteras o binarias, presentando estrategias aproximadas (heurísticas) como alternativa práctica de solución para problemas difíciles de resolver con métodos exactos, y compararemos explícitamente el trade-off calidad–tiempo entre solvers y heurísticas (**Lecciones 3.7**).\n",
                "   \n",
                "4. **Decisiones en el tiempo: Programación Dinámica** (Semana 13): Cerraremos el módulo con la *Programación Dinámica* (DP), que aborda problemas de optimización **secuencial**. Veremos cómo la *inducción hacia atrás* de Bellman resuelve problemas por etapas, cómo un problema secuencial puede \"achatarse\" a una formulación LP masiva, y cómo DP abre la puerta al *Reinforcement Learning* (**Lección 3.8**).\n",
            ]
            break

    nb["cells"] = cells
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"[OK] {NB_PATH} patched successfully ({len(cells)} cells)")

if __name__ == "__main__":
    patch()
