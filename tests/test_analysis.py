import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import final_report  # noqa: E402


def row(task, arm, seed, solved, **kw):
    return {"task_id": task, "arm": arm, "seed": seed, "N": 8, "solved": solved,
            "completion_tokens": 1000, "prompt_tokens": 500, "requests": 8, "ttfv_ms": 1000, **kw}


@pytest.fixture
def run_dir(tmp_path):
    rows = []
    for i in range(10):
        t = f"ClassEval_{i}"
        rows.append(row(t, "c", 0, i < 4))
        rows.append(row(t, "v2", 0, i < 7, over_budget=False, feedback="F1", seed_comp_solved=i < 4,
                        dead_slots=[] if i < 7 else ["b"],
                        rounds=[{"round": 0, "producer": "whole_class", "dead_after": [] if i < 4 else ["b"]},
                                {"round": 1, "producer": "repair", "targets": ["b"], "dead_after": []}]))
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def test_final_report_pairs_arms_and_writes_tables(run_dir, tmp_path):
    m = final_report.compute(final_report.load([run_dir]), ["c"])
    assert m["solve"]["v2"]["rate"] == 0.7 and m["solve"]["c"]["rate"] == 0.4
    p = m["paired"]["v2-c"]
    assert p["arm_only"] == 3 and p["base_only"] == 0 and p["diff"] == pytest.approx(0.3)
    assert p["mcnemar_p"] < 0.3 and p["bootstrap_ci"][0] >= 0
    assert m["v2"]["v2"]["repair_rounds_mean"] == 1.0
    assert m["v2"]["v2"]["solved_by_seed_samples"] == 4
    md = final_report.to_md(m)
    assert "| v2 | 8 | 0.70 |" in md and "McNemar" in md
