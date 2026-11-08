"""cuRobo + mesh validation of one fixed-order plan of order_then_layout.py (lattice_joint.validate on it).

    python -X utf8 -m experiments.baselines.validate_plan <index in the orders list> [--sga] [--model triangular_frame8]
Output: experiments/baselines/_out/validate_<model>_<tag>_<index>/<model>_validated.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from curobo_bridge.runtime import prepare

prepare()                  # torch before numpy / Panda3D on Windows (c10.dll)

OUT = Path(__file__).resolve().parent / "_out"


def main():
    i = int(sys.argv[1])
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "triangular_frame6"
    tag = "sga" if "--sga" in sys.argv else "all"
    src = OUT / f"order_then_layout_{model}{'_sga' if tag == 'sga' else ''}.json"
    r = json.loads(src.read_text(encoding="utf-8"))["orders"][i]
    d = OUT / f"validate_{model}_{tag}_{i}"
    d.mkdir(exist_ok=True)
    plan = dict(U=r["U"], layout=[[q, r["layout"][q]["pose"], r["layout"][q]["key"]] for q in r["order"]])
    (d / f"{model}.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    import lattice_joint.validate as val
    val.OUT = d
    return val.main(["--model", model])


if __name__ == "__main__":
    sys.exit(main())
