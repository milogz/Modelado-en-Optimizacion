"""Quick script to inspect notebook cell structure."""
import json, sys

def inspect(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n=== {path} ===")
    for i, c in enumerate(nb["cells"]):
        first_line = c["source"][0][:140].replace("\n", " ") if c["source"] else "(empty)"
        print(f"  {i:>3}: [{c['cell_type']:>8}] {first_line}")

for p in sys.argv[1:]:
    inspect(p)
