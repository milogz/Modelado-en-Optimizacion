# Modelado en Optimización 📊

Repositorio en desarrollo para el curso **Modelado en Optimización** (Departamento de Ingeniería Industrial, Universidad de los Andes, Colombia).

El material combina **notebooks interactivos**, **código auxiliar en Python**, y **figuras** para guiar a estudiantes en el aprendizaje de métodos fundamentales de optimización.

---

## 📚 Contenidos

Los notebooks siguen una progresión didáctica:

- **MO_3-0 Intro MetodosOpti.ipynb**  
  Panorama general de métodos de optimización y su relación con problemas reales.

- **MO_3-1 Busqueda.ipynb**  
  Búsqueda directa y métodos iterativos simples.

- **MO_3-2 Estimacion.ipynb**  
  Estimación de gradientes y nociones de ajuste.

- **MO_3-3 LP (Bases).ipynb**  
  Fundamentos de la programación lineal: holgura, factibilidad, bases.

- **MO_3-4 LP (Simplex).ipynb**  
  Algoritmo Simplex, pasos iterativos y visualización de soluciones.

- **MO_3-5 LP (Dualidad).ipynb**  
  Interpretación económica (precios sombra), holgura complementaria

- **MO_3-6 NLP.ipynb**  
  Introducción a problemas de optimización con restricciones y no linealidades

- **MO_3-7 MILP.ipynb**  
  Introducción a la programación lineal entera mixta (MILP).

- **MO_3-8 DP.ipynb**  
  Programación Dinámica: inducción hacia atrás, equivalencia DP→LP, puente hacia Reinforcement Learning.

### 🧪 Laboratorios Interactivos (Dashboards)

Los dashboards son notebooks complementarios enfocados en **experimentación** con los conceptos clave, separando la exploración interactiva del desarrollo paso a paso de las lecciones.

- **MO_Lab_3-1_Busqueda_Dashboard.ipynb**  
  Comparación interactiva de búsqueda local vs descenso por gradiente. Controles para parámetros de vecindario (radio $d$, vecinos $k$) y tasa de aprendizaje ($\alpha$) sobre distintas superficies (Cuadrática, Rosenbrock, Rastrigin).  
  *Complementa*: `MO_3-1 Busqueda.ipynb`

- **MO_Lab_3-3_LP_Dashboard.ipynb**  
  Dashboard unificado de LP con dos tabs:  
  • **Tab 1** – Factibilidad y Holguras: explorar puntos, ver holguras y restricciones activas en tiempo real.  
  • **Tab 2** – Simplex Paso a Paso: ejecutar iteraciones del Simplex visualmente (dirección, longitud, pivoteo).  
  *Complementa*: `MO_3-3 LP (Bases).ipynb` y `MO_3-4 LP (Simplex).ipynb`

### 🧭 Hilo Narrativo (Semestre 2026-10)

El módulo sigue un flujo progresivo:

1. **Más allá de lo Lineal** → De Simplex a heurísticas; trade-off calidad–tiempo.
2. **La Optimización como Motor de ML** → Descenso de Gradiente; función de pérdida; regresión y clasificación.
3. **Decisiones en el Tiempo** → Programación Dinámica → puerta abierta al Reinforcement Learning.

Los notebooks están apoyados por scripts auxiliares (`helpers_*.py`) y figuras en la carpeta `assets/figs`.