"""CLI: python -m fidelitymem.eval.cli --memories 1000 --queries 100 --k 5 10"""
from __future__ import annotations

import argparse
import json

from .harness import format_table, run_suite


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="FidelityMem benchmark harness")
    ap.add_argument("--memories", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=0.95)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--scale", type=int, nargs="*", help="run the scale study at these record counts")
    ap.add_argument("--out", type=str, default=None, help="write JSON report here")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="seeds for the scale study (mean ± 95% CI)")
    ap.add_argument("--embedder", type=str, default="best", help="best | spacy | st | hashing")
    a = ap.parse_args(argv)
    from ..stores import set_default_embedder
    from ..stores.embedders import best_available, embedder_name

    emb = best_available(None if a.embedder == "best" else a.embedder)
    set_default_embedder(emb)
    print(f"embedder={embedder_name(emb)}", flush=True)
    if a.scale is not None:
        from .harness import run_scale

        rep = run_scale(tuple(a.scale) or (300, 1000, 3000, 10000), a.queries, a.k[0], a.seed, a.rho, tuple(a.seeds) if a.seeds else None)
        rep = {"embedder": embedder_name(emb), "seeds": a.seeds or [a.seed], "levels": rep}
        if a.out:
            with open(a.out, "w") as f:
                json.dump(rep, f, indent=1)
        return
    res = run_suite(a.memories, a.queries, tuple(a.k), a.seed, a.rho, a.quick)
    if a.json:
        print(json.dumps([r.__dict__ for r in res], indent=2))
    else:
        print(f"memories={a.memories} queries={a.queries} seed={a.seed} rho={a.rho}")
        print(format_table(res))
        print("\nper-type Recall@K for the controller run:")
        for r in res:
            if r.system.startswith("fidelitymem_ctrl"):
                print(f"  K={r.k}: " + ", ".join(f"{t}={v:.3f}" for t, v in sorted(r.per_type.items())))


if __name__ == "__main__":
    main()
