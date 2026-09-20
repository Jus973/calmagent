import json, pathlib, shutil, subprocess, sys, tempfile
sys.path.insert(0, ".")
from bench_agent.make_tasks import load_rows
rows = {r["task_id"]: r for r in load_rows()}
man = json.loads(pathlib.Path("bench_agent/tasks/TASKS.json").read_text())
ok = True
for m in man["tasks"]:
    tid = m["task_id"]
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td) / tid
        shutil.copytree(f"bench_agent/tasks/{tid}", d)
        row = rows[tid]
        imports = "\n".join(row["import_statement"])
        (d / "solution.py").write_text(imports + "\n\n" + row["solution_code"])
        p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=d, capture_output=True, text=True, timeout=300)
        tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-200:]
        status = "PASS" if p.returncode == 0 else "FAIL"
        ok &= p.returncode == 0
        print(f"{tid:<14} reference -> {status:<4} {tail}")
print("\nALL REFERENCES PASS" if ok else "\nSOME REFERENCES FAIL")
