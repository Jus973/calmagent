"""Re-package ClassEval tasks as tiny standalone repos an off-the-shelf agent can work in.

    python -m bench_agent.make_tasks

Each task becomes `bench_agent/tasks/<ClassEval_id>/`:

    solution.py             the skeleton (docstrings + signatures, bodies replaced by `pass`)
    tests/test_solution.py  the hidden tests, importable, importing from solution
    TASK.md                 the instruction the agent is handed

The agent is never told the tests' contents by us; they sit in the repo because a real developer's
repo has its tests in it, and because the agent loop needs something to run. The bench's own
grading re-runs them from a pristine copy, so an agent that edits the tests cannot fool it.

Selection (seed 0): 8 tasks the 7B model solved in `runs/20260919T163449Z_main` under any arm, and
2 it never solved, so the solve rate has signal in both directions.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import shutil

HF_CACHE = pathlib.Path.home() / ".cache/huggingface/datasets/FudanSELab___class_eval"


def load_rows() -> list[dict]:
    """ClassEval's 100 rows.

    `calm_coder.bench.classeval.load_rows` needs `datasets`, which is not installed in this
    interpreter, and `calm_coder/` is frozen tonight so it cannot grow a fallback. The dataset
    itself is already in the HuggingFace cache as an Arrow file, so read that directly and keep
    `datasets` as the preferred path for anyone who has it.
    """
    try:
        from calm_coder.bench.classeval import load_rows as _upstream
        return _upstream()
    except ImportError:
        pass
    import pyarrow as pa
    arrows = sorted(HF_CACHE.rglob("class_eval-test.arrow"))
    if not arrows:
        raise SystemExit(f"no ClassEval arrow file under {HF_CACHE}; pip install datasets")
    with pa.memory_map(str(arrows[0])) as src:
        table = pa.ipc.open_stream(src).read_all()
    return table.to_pylist()


ROOT = pathlib.Path(__file__).resolve().parent
TASKS = ROOT / "tasks"
SOURCE_RUN = pathlib.Path("runs/20260919T163449Z_main/results.jsonl")

TASK_MD = """# Task

Implement the class in `solution.py` so that the test suite passes.

Run the tests with:

    python -m pytest -q

Rules:

- Edit `solution.py` only. Do not edit anything under `tests/`.
- Keep the class name, the constructor, and every method signature exactly as given.
- Use the standard library only.
- You are done when `python -m pytest -q` reports no failures.

## Environment

- Every command runs in a fresh non-interactive subshell with no terminal attached.
- Never run an interactive tool. `nano`, `vi`, `vim`, `emacs`, `less` and `more` will hang until
  they are killed and you will lose the turn. Write files with `cat <<'EOF' > solution.py`, and
  read them with `cat`, `nl -ba` or `sed -n`.
- Rewriting the whole of `solution.py` in one `cat <<'EOF'` heredoc is usually faster and more
  reliable than patching it with `sed`.
"""


def solved_and_unsolved(run: pathlib.Path) -> tuple[list[str], list[str]]:
    solved: dict[str, bool] = collections.defaultdict(bool)
    for line in run.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        solved[row["task_id"]] |= bool(row.get("solved"))
    yes = sorted([t for t, v in solved.items() if v])
    no = sorted([t for t, v in solved.items() if not v])
    return yes, no


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-solved", type=int, default=8)
    ap.add_argument("--n-unsolved", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", type=pathlib.Path, default=SOURCE_RUN)
    args = ap.parse_args()

    yes, no = solved_and_unsolved(args.run)
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(yes, args.n_solved)) + sorted(rng.sample(no, args.n_unsolved))

    rows = {r["task_id"]: r for r in load_rows()}
    if TASKS.exists():
        shutil.rmtree(TASKS)
    TASKS.mkdir(parents=True)

    manifest = []
    for task_id in chosen:
        row = rows[task_id]
        d = TASKS / task_id
        (d / "tests").mkdir(parents=True)

        # The skeleton already carries the imports it needs as `import_statement`.
        imports = "\n".join(row["import_statement"])
        skeleton = row["skeleton"]
        solution = (skeleton if imports and imports in skeleton else f"{imports}\n\n{skeleton}").rstrip() + "\n"
        (d / "solution.py").write_text(solution)

        # ClassEval's test module refers to the class by bare name; import it from solution.
        test_src = row["test"]
        header = f"from solution import {row['class_name']}\n"
        (d / "tests" / "test_solution.py").write_text(header + "\n" + test_src.rstrip() + "\n")
        (d / "tests" / "__init__.py").write_text("")
        # so `from solution import X` resolves when pytest is run from the task root
        (d / "conftest.py").write_text(
            "import pathlib, sys\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n"
        )
        (d / "TASK.md").write_text(TASK_MD)

        manifest.append({
            "task_id": task_id,
            "class_name": row["class_name"],
            "solved_by_7b_in_source_run": task_id in yes,
            "test_classes": row["test_classes"],
        })

    (TASKS / "TASKS.json").write_text(json.dumps({
        "seed": args.seed,
        "source_run": str(args.run),
        "n_solved": args.n_solved,
        "n_unsolved": args.n_unsolved,
        "tasks": manifest,
    }, indent=2) + "\n")

    for m in manifest:
        print(f"{m['task_id']:<14} {m['class_name']:<28} prior_solved={m['solved_by_7b_in_source_run']}")
    print(f"\nwrote {len(manifest)} tasks to {TASKS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
