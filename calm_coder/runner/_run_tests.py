"""Subprocess entry point: python -I _run_tests.py mod.py per_class_timeout_s Class1 Class2 ...

Writes result.json in cwd: {test_class: {"result", "detail", "wall_ms", "tests_run"}}.
"""
import importlib.util
import io
import json
import os
import resource
import signal
import sys
import time
import traceback
import unittest

DETAIL_MAX = 2048


class CalmTimeout(BaseException):
    pass


def _limits() -> None:
    for lim, val in ((resource.RLIMIT_AS, 1 << 30), (resource.RLIMIT_CPU, 15), (resource.RLIMIT_NOFILE, 256)):
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass  # RLIMIT_AS is unsupported on macOS; the wall timeout still applies


def _write(out: dict) -> None:
    with open("result.json", "w") as f:
        json.dump(out, f)


def main() -> None:
    path, per_class = sys.argv[1], float(sys.argv[2])
    classes = sys.argv[3:]
    _limits()
    sys.path.insert(0, os.getcwd())
    devnull = open(os.devnull, "w")
    real_out, real_err = sys.stdout, sys.stderr
    out: dict = {}
    try:
        sys.stdout = sys.stderr = devnull
        spec = importlib.util.spec_from_file_location("mod", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mod"] = mod
        spec.loader.exec_module(mod)
    except BaseException:
        sys.stdout, sys.stderr = real_out, real_err
        tb = traceback.format_exc()[-DETAIL_MAX:]
        kind = "inconclusive" if "CalmStubHit" in tb else "error"
        _write({c: {"result": kind, "detail": "import: " + tb, "wall_ms": 0, "tests_run": 0} for c in classes})
        return

    for name in classes:
        t0 = time.monotonic()
        cls = getattr(mod, name, None)
        if not (isinstance(cls, type) and issubclass(cls, unittest.TestCase)):
            out[name] = {"result": "error", "detail": f"no test class {name}", "wall_ms": 0, "tests_run": 0}
            continue
        suite = unittest.TestLoader().loadTestsFromTestCase(cls)
        result = unittest.TestResult()

        def on_alarm(signum, frame, result=result):
            result.stop()
            raise CalmTimeout(name)

        signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, per_class)
        try:
            suite.run(result)
        except CalmTimeout:
            pass
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        tbs = [tb for _, tb in result.errors + result.failures]
        joined = "\n".join(tbs)
        if "CalmTimeout" in joined or result.shouldStop:
            kind = "timeout"
        elif "CalmStubHit" in joined:
            kind = "inconclusive"
        elif result.errors:
            kind = "error"
        elif result.failures:
            kind = "fail"
        elif result.testsRun == 0:
            kind = "error"
        else:
            kind = "pass"
        detail = tbs[0][-DETAIL_MAX:] if tbs else ""
        if kind == "inconclusive":
            detail = next((ln.strip() for ln in joined.splitlines() if ln.startswith("CalmStubHit")), detail)
        out[name] = {"result": kind, "detail": detail,
                     "wall_ms": int((time.monotonic() - t0) * 1000), "tests_run": result.testsRun}
    sys.stdout, sys.stderr = real_out, real_err
    _write(out)


if __name__ == "__main__":
    main()
