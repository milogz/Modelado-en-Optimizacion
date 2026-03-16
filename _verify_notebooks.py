"""
Verification script v2: checks structural requirements for all modified/new notebooks.
"""
import json, sys, importlib

RESULTS = []

def check(name, condition, msg=""):
    status = "PASS" if condition else "FAIL"
    RESULTS.append((name, status, msg))
    print(f"  [{status}] {name}" + (f" -- {msg}" if msg and not condition else ""))

def verify_notebook(path, expect_traceability=True, expect_ia_citation=True, min_reflections=2):
    print(f"\n=== {path} ===")
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except FileNotFoundError:
        check(f"{path} exists", False, "File not found")
        return
    except json.JSONDecodeError as e:
        check(f"{path} valid JSON", False, str(e))
        return

    check(f"{path} valid JSON", True)
    cells = nb.get("cells", [])
    check(f"{path} has cells", len(cells) > 0, f"{len(cells)} cells")

    # Check traceability - look for "2026" in first cell
    if expect_traceability:
        first_src = "".join(cells[0].get("source", []))
        has_trace = "2026" in first_src
        check(f"{path} traceability cell", has_trace,
              f"first cell: {first_src[:80]!r}..." if not has_trace else "")

    # Check IA citation
    if expect_ia_citation:
        last_src = "".join(cells[-1].get("source", []))
        has_ia = "80-82" in last_src or "IA generativa" in last_src
        check(f"{path} IA citation cell", has_ia)

    # Count reflection questions — search for "Pregunta de Reflexi" (handles both ó and o)
    reflection_count = 0
    for c in cells:
        src = "".join(c.get("source", []))
        if "Pregunta de Reflexi" in src:
            reflection_count += 1
    check(f"{path} reflection questions >= {min_reflections}", reflection_count >= min_reflections,
          f"found {reflection_count}")

def verify_module(module_name):
    print(f"\n=== Module: {module_name} ===")
    try:
        mod = importlib.import_module(module_name)
        check(f"{module_name} importable", True)
        return mod
    except Exception as e:
        check(f"{module_name} importable", False, str(e))
        return None

def main():
    verify_notebook("MO_3-0 Intro Metodos Optimizacion.ipynb",
                    expect_traceability=True, expect_ia_citation=False, min_reflections=0)
    verify_notebook("MO_3-2 Estimacion.ipynb",
                    expect_traceability=True, expect_ia_citation=True, min_reflections=2)
    verify_notebook("MO_3-7 MILP.ipynb",
                    expect_traceability=True, expect_ia_citation=True, min_reflections=2)
    verify_notebook("MO_3-8 DP.ipynb",
                    expect_traceability=True, expect_ia_citation=True, min_reflections=2)

    verify_module("helpers_dp")

    mod_est = verify_module("helpers_estimacion")
    if mod_est:
        for fn in ["plot_loss_surface", "sigmoid", "logistic_loss", "gd_logistic", "plot_decision_boundary"]:
            check(f"helpers_estimacion.{fn} exists", hasattr(mod_est, fn))

    print("\n=== readme.md ===")
    with open("readme.md", "r", encoding="utf-8") as f:
        readme = f.read()
    check("readme mentions MO_3-8 DP", "MO_3-8 DP" in readme)
    check("readme mentions narrative arc", "Hilo Narrativo" in readme)

    print("\n" + "="*50)
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"TOTAL: {passed} PASS, {failed} FAIL")
    if failed:
        print("\nFailed checks:")
        for name, status, msg in RESULTS:
            if status == "FAIL":
                print(f"  - {name}: {msg}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
