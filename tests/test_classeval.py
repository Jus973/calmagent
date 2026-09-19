from calm_coder.bench.classeval import check_task, select_subset


def test_check_task_accepts_head(rows):
    task, why = check_task(rows[0], runs=1)
    assert why is None and task.task_id == "ClassEval_0"


def test_check_task_rejects_non_stdlib(rows):
    row = dict(rows[0], import_statement=["import numpy as np"])
    task, why = check_task(row, runs=1)
    assert task is None and why.startswith("non_stdlib_imports")


def test_check_task_rejects_module_level_code(rows):
    row = dict(rows[0], skeleton="X = 1\n" + rows[0]["skeleton"])
    assert check_task(row, runs=1)[1].startswith("module_level_code")


def test_select_is_deterministic(rows):
    a, b = select_subset(3, 0, rows), select_subset(3, 0, rows)
    assert a["task_ids"] == b["task_ids"] and len(a["task_ids"]) == min(3, a["eligible"])
