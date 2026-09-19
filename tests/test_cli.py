import json
import re
from pathlib import Path

import httpx
import pytest

import calm_coder.cli as cli
from calm_coder.runner.sandbox import run_tests
from calm_coder.serve.client import Client
from calm_coder.task import task_from_files

DEMO = Path(cli.__file__).parent / "demo"

REF = {
    "set": "def set(self, key, value):\n    self.data[self._normalize_key(key)] = value\n\ndef _normalize_key(self, key):\n    return key.strip().lower()",
    "get": "def get(self, key, default=None):\n    return self.data.get(self._norm(key), default)\n\ndef _norm(self, k):\n    return k.strip().lower()",
    "delete": "def delete(self, key):\n    k = key.strip().lower()\n    if k in self.data:\n        del self.data[k]\n        return True\n    return False",
    "keys_with_prefix": "def keys_with_prefix(self, prefix):\n    p = prefix.strip().lower()\n    return sorted(k for k in self.data if k.startswith(p))",
}


def test_demo_slot_tests_inferred_structurally():
    t = task_from_files(DEMO / "task.py", DEMO / "test_task.py")
    assert t.slot_tests == {"set": "KVStoreTestSet", "get": "KVStoreTestGet", "delete": "KVStoreTestDelete",
                            "keys_with_prefix": "KVStoreTestKeysWithPrefix"}


@pytest.mark.parametrize("flags", [[], ["--sequential"], ["--width", "3"]],
                         ids=["pipelined", "sequential", "wide"])
def test_cli_solve_demo_with_fake_model(tmp_path, monkeypatch, flags):
    def h(req):
        body = json.loads(req.content)
        name = re.search(r"Implement `(\w+)` now\.$", body["messages"][-1]["content"]).group(1)
        text = f"```python\n{REF[name]}\n```"
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"content": text}}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    monkeypatch.setattr(cli, "Client", lambda: Client("http://fake/v1", "fake", no_n=True, transport=httpx.MockTransport(h)))
    out = tmp_path / "kv.py"
    rc = cli.main(["solve", str(DEMO / "task.py"), "--tests", str(DEMO / "test_task.py"), "--N", "2",
                   "--out", str(out), "--events", str(tmp_path / "ev.jsonl"), *flags])
    assert rc == 0
    t = task_from_files(DEMO / "task.py", DEMO / "test_task.py")
    res = run_tests(out.read_text(), t.test_src, list(t.test_classes))
    assert all(v["result"] == "pass" for v in res.values()), res
    assert "CalmStubHit" not in out.read_text()
