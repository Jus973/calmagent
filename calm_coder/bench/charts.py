"""Charts (§3.15), PNG + SVG. Every caption states temperature, model, subset size and seeds.

python -m calm_coder.bench.charts runs/<dir>     # needs metrics.json (run bench.metrics first)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (dataviz skill), categorical slots in fixed order; text never wears series color.
CALM, C_N, C_TOK = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "text.color": INK,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False, "legend.labelcolor": INK2,
})


def caption(m: dict) -> str:
    c = m["config"]
    return (f"Model {c.get('model')} · T={c['sampling']['temperature']} · {m['n_tasks']} ClassEval tasks · "
            f"seeds {m['seeds']} · oracle-verified (coverage)")


def _save(fig, out: Path, name: str, m: dict) -> list[Path]:
    fig.text(0.01, 0.01, caption(m), fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    paths = [out / f"{name}.png", out / f"{name}.svg"]
    for p in paths:
        fig.savefig(p, dpi=160)
    plt.close(fig)
    return paths


def _label_end(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)


def _label_ends(ax, items, min_gap=0.055):
    """Direct end labels, dodged vertically (axis fraction) so coinciding series stay readable."""
    items = sorted(items, key=lambda t: t[1])
    placed = []
    for x, y, text in items:
        yy = y if not placed or y - placed[-1] >= min_gap else placed[-1] + min_gap
        placed.append(yy)
        ax.text(x * 1.06, yy, text, va="center", fontsize=9, color=INK2)


def solve_vs_n(m, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ns = sorted(int(n) for n in m["solve"]["calm"])
    ends = []
    for key, color, label, ls in (("calm", CALM, "CALM@N", "-"), ("c@N", C_N, "C@N", "-"), ("c@tokens", C_TOK, "C@tokens", (0, (4, 2)))):
        s = m["solve"][key]
        y = [s[str(n)]["rate"] if str(n) in s else s[n]["rate"] for n in Ns]
        lo = [(s.get(str(n)) or s[n])["wilson"][0] for n in Ns]
        hi = [(s.get(str(n)) or s[n])["wilson"][1] for n in Ns]
        ax.fill_between(Ns, lo, hi, color=color, alpha=0.10, linewidth=0)
        ax.plot(Ns, y, color=color, marker="o", label=label, linestyle=ls, markeredgecolor=SURFACE, markeredgewidth=2)
        ends.append((Ns[-1], y[-1], label))
    _label_ends(ax, ends)
    a = m["solve"]["a_pass1"]
    a1 = (a.get("1") or a[1])["rate"]
    ax.axhline(a1, color=INK2, linestyle="--", linewidth=1.2, label="holistic pass@1 (A)")
    if "a_greedy" in m["solve"]:
        g = (m["solve"]["a_greedy"].get("1") or m["solve"]["a_greedy"][1])["rate"]
        ax.axhline(g, color=MUTED, linestyle=":", linewidth=1.2, label="A greedy")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.set_xlim(Ns[0] * 0.85, Ns[-1] * 1.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("samples per slot (CALM) / per class (C)")
    ax.set_ylabel("class-level solve rate (95% Wilson CI)")
    ax.set_title("Solve rate vs sampling budget", loc="left", color=INK)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out, "i_solve_vs_n", m)


def independence(m, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ns = sorted(int(n) for n in m["independence"])
    iv = lambda n: m["independence"].get(str(n)) or m["independence"][n]
    pred, obs = [iv(n)["predicted"] for n in Ns], [iv(n)["observed"] for n in Ns]
    ax.plot(Ns, pred, color=CALM, linestyle="--", marker="o", markerfacecolor=SURFACE, label="predicted if slots were independent")
    ax.plot(Ns, obs, color=CALM, marker="o", markeredgecolor=SURFACE, markeredgewidth=2, label="observed CALM")
    ax.fill_between(Ns, obs, pred, color=CALM, alpha=0.08, linewidth=0)
    _label_ends(ax, [(Ns[-1], pred[-1], "predicted"), (Ns[-1], obs[-1], "observed")])
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.set_xlim(Ns[0] * 0.85, Ns[-1] * 1.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("samples per slot")
    ax.set_ylabel("solve rate")
    ax.set_title("Independence prediction vs observed — the gap is coupling", loc="left", color=INK)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out, "ii_independence", m)


def ttfv_cdf(m, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    n_tasks = m["n_tasks"]
    for key, color, label in (("calm", CALM, "CALM"), ("c", C_N, "C (whole class)")):
        xs = sorted(v / 1000 for v in m["ttfv"][key]["all_ms"])
        if not xs:
            continue
        ys = np.arange(1, len(xs) + 1) / n_tasks
        ax.step([0] + xs, [0] + list(ys), where="post", color=color, label=label)
        _label_end(ax, xs[-1], ys[-1], label)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time to first verified class (s): generation + stub tests + search")
    ax.set_ylabel("fraction of tasks solved")
    ax.set_title("Time to first verified class (CDF over all tasks)", loc="left", color=INK)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out, "iii_ttfv_cdf", m)


def dedup(m, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ns = sorted(int(n) for n in m["dedup"])
    dv = lambda n: m["dedup"].get(str(n)) or m["dedup"][n]
    x = np.arange(len(Ns))
    w = 0.36
    a = [100 * (dv(n)["alpha"] or 0) for n in Ns]
    e = [100 * (dv(n)["exact"] or 0) for n in Ns]
    b1 = ax.bar(x - w / 2 - 0.01, e, w, color=C_TOK, label="exact-unparse")
    b2 = ax.bar(x + w / 2 + 0.01, a, w, color=CALM, label="alpha-normalized")
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.0f}%", (r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"N={n}" for n in Ns])
    ax.set_ylabel("emitted fills that collapsed (%)")
    ax.set_title("Duplicate fills collapse in the store (saves test runs, not tokens)", loc="left", color=INK)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out, "iv_dedup", m)


def mismatch(m, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    mm, ho = m["mismatch"], m["heldout"]
    items = [
        ("interface mismatch\n(stub-passing comps that fail)", mm["rate"], f"{mm['failed']}/{mm['comps_all_stub_pass']}"),
        ("systematic mismatch (F1)\n(tasks)", mm["systematic_rate_of_all_tasks"], f"{mm['systematic']}/{m['n_tasks']}"),
        ("held-out class-level fail\n(select on method tests)", None if ho["rate"] is None else 1 - ho["rate"],
         f"{ho['selected'] - ho['class_level_pass']}/{ho['selected']}"),
    ]
    y = np.arange(len(items))[::-1]
    vals = [100 * (v or 0) for _, v, _ in items]
    ax.barh(y, vals, height=0.5, color=CALM)
    for yi, v, (_, raw, frac) in zip(y, vals, items):
        ax.annotate(("n/a" if raw is None else f"{v:.0f}%") + f"  ({frac})", (v, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.axvline(30, color=MUTED, linestyle=":", linewidth=1)
    ax.annotate("F1 threshold 30%", (30, y[0] + 0.4), fontsize=8, color=MUTED, xytext=(3, 0), textcoords="offset points")
    ax.set_yticks(y, [n for n, _, _ in items], fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("%")
    ax.set_title("Where coupling shows up", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "v_mismatch", m)


def cost_frontier(m, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    Ns = sorted(int(n) for n in m["solve"]["calm"])
    n_tasks = m["n_tasks"]
    for arm, key, color, label in (("calm", "calm", CALM, "CALM"), ("c", "c@N", C_N, "C (whole class)")):
        xs_tot, xs_comp, ys = [], [], []
        for n in Ns:
            c = m["cost"][f"{arm}@{n}"]
            xs_tot.append((c["prompt"] + c["completion"]) / n_tasks)
            xs_comp.append(c["completion"] / n_tasks)
            s = m["solve"][key]
            ys.append((s.get(str(n)) or s[n])["rate"])
        for xc, xt, yy in zip(xs_comp, xs_tot, ys):
            ax.plot([xc, xt], [yy, yy], color=color, alpha=0.35, linewidth=5, solid_capstyle="butt")
        ax.plot(xs_tot, ys, color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        for n, x, yy in zip(Ns, xs_tot, ys):
            ax.annotate(f"N={n}", (x, yy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.plot([], [], color=INK2, alpha=0.35, linewidth=5, label="prompt share (completion → total)")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean tokens per task (prompt + completion, log scale)")
    ax.set_ylabel("solve rate")
    ax.set_title("Cost frontier: solve rate vs tokens spent", loc="left", color=INK)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, out, "vi_cost_frontier", m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    d = Path(ap.parse_args().run_dir)
    m = json.loads((d / "metrics.json").read_text())
    out = d / "charts"
    out.mkdir(exist_ok=True)
    for f in (solve_vs_n, independence, ttfv_cdf, dedup, mismatch, cost_frontier):
        print(*f(m, out))


if __name__ == "__main__":
    main()
