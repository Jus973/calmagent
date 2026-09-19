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
    body = summary.split("\n", 1)[1].replace("\n## ", "\n### ")
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
