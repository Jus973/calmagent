from calm_coder.runner.sandbox import run_tests

TESTS = '''
import unittest
class TA(unittest.TestCase):
    def test_a(self): self.assertEqual(f(), 1)
class TB(unittest.TestCase):
    def test_b(self): g()
class TC(unittest.TestCase):
    def test_c(self): self.assertEqual(f(), 2)
'''


def test_pass_fail_inconclusive():
    src = "class CalmStubHit(BaseException): pass\ndef f(): return 1\ndef g(): raise CalmStubHit('g')\n"
    r = run_tests(src, TESTS, ["TA", "TB", "TC", "TMissing"])
    assert {k: v["result"] for k, v in r.items()} == {"TA": "pass", "TB": "inconclusive", "TC": "fail", "TMissing": "error"}


def test_infinite_loop_times_out():
    src = "def f():\n    while True: pass\ndef g(): pass\n"
    r = run_tests(src, TESTS, ["TA", "TB"], per_class_timeout_s=1, wall_timeout_s=10)
    assert r["TA"]["result"] == "timeout" and r["TB"]["result"] == "pass"


def test_import_error_marks_all():
    r = run_tests("def f(:\n", TESTS, ["TA", "TB"])
    assert {v["result"] for v in r.values()} == {"error"}


def test_writes_stay_in_tempdir(tmp_path):
    src = "def f():\n    open('x.txt','w').write('1'); return 1\ndef g(): pass\n"
    assert run_tests(src, TESTS, ["TA"])["TA"]["result"] == "pass"
