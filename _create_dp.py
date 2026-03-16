"""
Generator script for MO_3-8 DP.ipynb
Creates the full Dynamic Programming notebook from scratch.
"""
import json

NB_PATH = "MO_3-8 DP.ipynb"

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src if isinstance(src, list) else [src]}
def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src if isinstance(src, list) else [src]}

cells = []

# ── Traceability ──
cells.append(md([
    "> **Actualización 2026:** Se introduce *Programación Dinámica* (DP) \n",
    "> como puente entre optimización secuencial y Reinforcement Learning.\n",
]))

# ── Title ──
cells.append(md([
    "# Modelado en Optimización (IIND-2501)\n",
    "\n",
    "## Lección 3.8: Programación Dinámica\n",
]))

cells.append(md([
    "> **Conexión con las lecciones anteriores:** En las lecciones 3.1-3.7 hemos visto cómo resolver problemas de optimización \n",
    "> con distintas estrategias: búsqueda local, descenso de gradiente, Simplex, Branch & Bound, y heurísticas. \n",
    "> Todos estos métodos asumen que las decisiones se toman **simultáneamente**. \n",
    "> Ahora abordaremos problemas donde las decisiones son **secuenciales** — cada decisión depende del *estado* actual y afecta los estados futuros.\n",
]))

# ── Setup ──
cells.append(code([
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from helpers_dp import (\n",
    "    crear_red_etapas, dp_backward, reconstruir_camino,\n",
    "    visualizar_red, dp_to_lp\n",
    ")\n",
]))

# ── Section 1: Introduction ──
cells.append(md([
    "## 1 — Introducción a la Programación Dinámica\n",
    "\n",
    "La **Programación Dinámica** (DP) es una técnica para resolver problemas de optimización que pueden descomponerse en **etapas** o **períodos**. \n",
    "Su fundamento es el **Principio de Optimalidad de Bellman** (1957):\n",
    "\n",
    "> *Una política óptima tiene la propiedad de que, independientemente de las decisiones iniciales, \n",
    "> las decisiones restantes deben constituir una política óptima con respecto al estado resultante de la primera decisión.*\n",
    "\n",
    "Formalmente, si $V_t(s)$ es el costo mínimo para ir desde el estado $s$ en la etapa $t$ hasta el destino final:\n",
    "\n",
    "$$\n",
    "V_t(s) = \\min_{a \\in A(s)} \\bigl\\{c(s,a) + V_{t+1}\\bigl(f(s,a)\\bigr)\\bigr\\}\n",
    "$$\n",
    "\n",
    "donde:\n",
    "- $A(s)$: conjunto de acciones posibles en el estado $s$\n",
    "- $c(s,a)$: costo inmediato de tomar la acción $a$ en el estado $s$\n",
    "- $f(s,a)$: estado siguiente tras tomar la acción $a$\n",
    "\n",
    "La ecuación se resuelve **de atrás hacia adelante** (*backward induction* o *inducción hacia atrás*).\n",
]))

cells.append(md([
    "---\n",
    "### Pregunta de Reflexión\n",
    "\n",
    "1. ¿En qué se diferencia la programación dinámica de la *búsqueda local* (Lección 3.1)?\n",
    "2. ¿Por qué es importante que el problema tenga **estructura de etapas**?\n",
]))

# ── Section 2: Stagecoach Problem ──
cells.append(md([
    "## 2 — Ejemplo: Camino más corto en una red por etapas (*Stagecoach Problem*)\n",
    "\n",
    "Consideremos el clásico problema del diligente: un viajero debe ir de la ciudad **A** a la ciudad **E**, \n",
    "pasando por varias ciudades intermedias organizadas en etapas. En cada etapa, elige a qué ciudad ir, \n",
    "pagando un costo (distancia, peaje, etc.) por cada arco.\n",
    "\n",
    "**Objetivo:** encontrar el camino de costo mínimo de A a E.\n",
]))

cells.append(code([
    "# Crear la red de etapas\n",
    "grafo, etapas, pos = crear_red_etapas()\n",
    "\n",
    "# Visualizar la red (sin política aún)\n",
    "visualizar_red(grafo, pos, titulo='Red por etapas — Stagecoach Problem')\n",
]))

cells.append(md([
    "### Resolución por Inducción Hacia Atrás\n",
    "\n",
    "Partimos del destino E ($V(E)=0$) y vamos calculando $V_t(s)$ para cada nodo, \n",
    "moviéndonos hacia atrás por las etapas:\n",
]))

cells.append(code([
    "# Resolver con DP (backward induction)\n",
    "V, pi = dp_backward(grafo, etapas)\n",
    "\n",
    "# Mostrar función de valor y política\n",
    "print('Función de Valor V(s):')\n",
    "for s in sorted(V.keys()):\n",
    "    print(f'  V({s}) = {V[s]}')\n",
    "\n",
    "print('\\nPolitica optima pi(s):')\n",
    "for s in sorted(pi.keys()):\n",
    "    print(f'  pi({s}) = {pi[s]}')\n",
    "\n",
    "# Reconstruir y mostrar camino óptimo\n",
    "camino_opt = reconstruir_camino(pi)\n",
    "print(f'\\nCamino optimo: {\" -> \".join(camino_opt)}')\n",
    "print(f'Costo optimo:  {V[\"A\"]}')\n",
]))

cells.append(code([
    "# Visualizar la red con la política óptima resaltada\n",
    "visualizar_red(grafo, pos, politica=pi, V=V,\n",
    "               titulo=f'Camino optimo (costo = {V[\"A\"]})')\n",
]))

cells.append(md([
    "---\n",
    "### Pregunta de Reflexión\n",
    "\n",
    "1. ¿Cuál es la relación entre la función de valor $V_t(s)$ y el concepto de *incumbente* que vimos en búsqueda local?\n",
    "2. ¿Qué ventaja tiene resolver el problema *de atrás hacia adelante* en lugar de enumerar todos los caminos?\n",
    "3. Si hubiera $k$ nodos por etapa y $T$ etapas, ¿cuántas operaciones requiere DP vs. enumeración exhaustiva?\n",
]))

# ── Section 3: DP → LP ──
cells.append(md([
    "## 3 — DP → LP: \"Achatando\" un problema secuencial\n",
    "\n",
    "Cualquier problema de camino más corto en un grafo dirigido se puede formular como un **programa lineal** \n",
    "de flujo de costo mínimo:\n",
    "\n",
    "$$\n",
    "\\min \\sum_{(i,j) \\in \\mathcal{A}} c_{ij} \\, x_{ij}\n",
    "$$\n",
    "$$\n",
    "\\text{s.a.} \\quad \\sum_{j:(i,j)\\in\\mathcal{A}} x_{ij} - \\sum_{j:(j,i)\\in\\mathcal{A}} x_{ji} = \n",
    "\\begin{cases} 1 & \\text{si } i = \\text{origen} \\\\ -1 & \\text{si } i = \\text{destino} \\\\ 0 & \\text{en otro caso} \\end{cases}\n",
    "$$\n",
    "$$x_{ij} \\geq 0 \\quad \\forall (i,j)$$\n",
    "\n",
    "Esto \"achata\" la estructura de etapas en una formulación **masiva** pero lineal. \n",
    "Verifiquemos que ambas formulaciones dan el **mismo resultado**:\n",
]))

cells.append(code([
    "# Resolver el mismo problema como LP con PuLP\n",
    "z_lp, flujos = dp_to_lp(grafo)\n",
    "\n",
    "print(f'Costo optimo (LP):  {z_lp}')\n",
    "print(f'Costo optimo (DP):  {V[\"A\"]}')\n",
    "print(f'Equivalencia:       {abs(z_lp - V[\"A\"]) < 1e-6}')\n",
    "\n",
    "print('\\nFlujos optimos (arcos activos):')\n",
    "for (s, s2), val in sorted(flujos.items()):\n",
    "    print(f'  {s} -> {s2}: x = {val:.1f}')\n",
]))

cells.append(md([
    "---\n",
    "### Pregunta de Reflexión\n",
    "\n",
    "1. ¿Cuáles son las **ventajas computacionales** de DP sobre la formulación LP para problemas con estructura de etapas?\n",
    "2. ¿En qué situaciones podría ser preferible usar el LP? (*Pista: piense en la flexibilidad para agregar restricciones adicionales.*)\n",
    "3. Note que las variables $x_{ij}$ del LP son **continuas**, pero la solución óptima es **entera** ($0$ o $1$). \n",
    "   ¿Por qué ocurre esto? (*Pista: propiedad de total unimodularidad.*)\n",
]))

# ── Section 4: Bridge to RL ──
cells.append(md([
    "## 4 — Puente hacia Reinforcement Learning\n",
    "\n",
    "La **ecuación de Bellman** que usamos para resolver el *stagecoach problem* es, de hecho, \n",
    "el fundamento teórico del **Reinforcement Learning** (RL).\n",
    "\n",
    "La relación clave es:\n",
    "\n",
    "| Concepto DP | Concepto RL |\n",
    "|:--:|:--:|\n",
    "| Estado $s$ | Estado del agente |\n",
    "| Acción $a$ | Acción del agente |\n",
    "| Costo $c(s,a)$ | Recompensa $r(s,a)$ (negativa) |\n",
    "| Función de valor $V(s)$ | Value function |\n",
    "| Política $\\pi(s)$ | Policy |\n",
    "| Inducción hacia atrás | Value Iteration |\n",
    "\n",
    "**DP asume que conocemos el modelo** — es decir, sabemos exactamente los costos $c(s,a)$ y las transiciones $f(s,a)$. \n",
    "En RL, el agente **no conoce el modelo** y debe aprenderlo interactuando con el entorno.\n",
    "\n",
    "```\n",
    "   DP (modelo conocido)        RL (modelo desconocido)\n",
    "   ────────────────────        ──────────────────────\n",
    "   Value Iteration        →   Q-Learning / SARSA\n",
    "   Policy Iteration       →   Policy Gradient\n",
    "   Bellman Optimality Eq. →   Q-function update\n",
    "```\n",
    "\n",
    "Este puente entre DP y RL muestra cómo los principios de **optimización secuencial** \n",
    "se extienden naturalmente al mundo del aprendizaje por refuerzo — un tema avanzado \n",
    "que podrán explorar en cursos posteriores.\n",
]))

cells.append(md([
    "---\n",
    "### Pregunta de Reflexión\n",
    "\n",
    "1. ¿Qué ocurre cuando **no conocemos** el modelo de transición $f(s,a)$? ¿Cómo podría un agente aprender la política óptima sin este conocimiento?\n",
    "2. ¿Ve alguna conexión entre el concepto de *incumbente* (Lección 3.1) y la *función de valor* $V(s)$?\n",
    "3. ¿Cómo se relaciona el *learning rate* del descenso de gradiente (Lección 3.2) con el *factor de descuento* $\\gamma$ en RL?\n",
]))

# ── IA Citation ──
cells.append(md([
    "---\n",
    "> **Política de uso de IA (artículos 80-82 del syllabus):** \n",
    "> Si utiliza herramientas de IA generativa (ChatGPT, Copilot, Gemini, etc.) para apoyar sus entregas, \n",
    "> debe citar la herramienta utilizada y el *prompt* empleado. \n",
    "> El incumplimiento de esta política será tratado como falta académica.\n",
]))


# ── Build notebook ──
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.7"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"[OK] {NB_PATH} created ({len(cells)} cells)")
