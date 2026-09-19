"""One command from a finished run to the README: confluence → metrics → charts → docs/ + README results.

python -m calm_coder.bench.report runs/<dir>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from calm_coder.bench import charts, confluence, metrics

ROOT = Path(__file__).parents[2]
START, END = "<!-- RESULTS -->", "<!-- /RESULTS -->"


def posthoc_md(d: Path) -> str:
    p = d / "posthoc" / "results.jsonl"
    if not p.exists():
        return ""
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ce = [r for r in rs if r["analysis"] == "ceiling"]
    rc = [r for r in rs if r["analysis"] == "recombine"]
    lost = [r["task_id"] for r in ce if r["verified"] and not r["calm_solved"]]
    open_ = [r["task_id"] for r in ce if not r["verified"] and not r["exhausted"]]
    gain = [r["task_id"] for r in rc if not r["c8_solved"] and r["exhaustive"]["verified"]]
    loss = [r["task_id"] for r in rc if r["c8_solved"] and not r["exhaustive"]["verified"]]
    c8 = sum(r["c8_solved"] for r in rc)
    comb = sum(bool(r["exhaustive"]["verified"]) for r in rc)
    return (
        "\n\n### Post-hoc, zero-token analyses (not pre-registered)\n\n"
        f"- **Search ceiling.** Exhaustive search over CALM's own N=8 fills (cap 3000 compositions) on {len(ce)} tasks: "
        f"tasks where the policy missed a working combination: {lost or 'none'}. Unsolved and not exhausted at the cap: "
        f"{open_ or 'none'}. CALM's losses are fill quality/coupling, not search.\n"
        f"- **Recombining C's samples.** Split each of C's 8 whole-class samples into per-method fills in a fresh store and "
        f"search (same tokens as C@8): {comb}/{len(rc)} solved vs C@8 {c8}/{len(rc)}. "
        f"Solved only by recombination: {gain or 'none'}; lost: {loss or 'none'}.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    d = Path(ap.parse_args().run_dir)
    cf = confluence.check_run(d)
    (d / "confluence.json").write_text(json.dumps(cf, indent=1))
    m = metrics.compute(d)
    (d / "metrics.json").write_text(json.dumps(m, indent=1, default=str))
    summary = metrics.summary_md(m)
    (d / "summary.md").write_text(summary)
    m = json.loads((d / "metrics.json").read_text())
    out = d / "charts"
    out.mkdir(exist_ok=True)
    for f in (charts.solve_vs_n, charts.independence, charts.ttfv_cdf, charts.dedup, charts.mismatch, charts.cost_frontier):
        f(m, out)
    docs = ROOT / "docs"
    (docs / "charts").mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.png"):
        shutil.copy(p, docs / "charts" / p.name)
    for name in ("summary.md", "metrics.json", "confluence.json"):
        shutil.copy(d / name, docs / name)
    body = summary.split("\n", 1)[1].replace("\n## ", "\n### ") + posthoc_md(d)
    imgs = "\n".join(f"![{p.stem}](docs/charts/{p.name})" for p in sorted((docs / "charts").glob("*.png")))
    section = (f"{START}\n## Results\n\nRun `{d.name}`; full numbers in [docs/summary.md](docs/summary.md) and "
               f"[docs/metrics.json](docs/metrics.json).\n{body}\n\n{imgs}\n{END}")
    readme = (ROOT / "README.md").read_text()
    if END in readme:
        readme = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: section, readme, flags=re.S)
    else:
        readme = readme.replace(START, section)
    (ROOT / "README.md").write_text(readme)
    print(summary)


if __name__ == "__main__":
    main()
