import ast
import json
from pathlib import Path

import pytest

from calm_coder.bench.classeval import task_from_row
from calm_coder.task import Task

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def rows():
    return json.loads((FIX / "classeval_head5.json").read_text())


def fn(src: str) -> ast.FunctionDef:
    import textwrap
    return ast.parse(textwrap.dedent(src)).body[0]


TOY_SKELETON = '''
import re

class Toy:
    """A toy."""

    def __init__(self):
        self.items = {}

    def put(self, key, value):
        """Store value."""

    def get(self, key):
        """Fetch value."""

    @staticmethod
    def norm(key):
        """Normalize."""
'''

TOY_TESTS = '''
import unittest

class ToyTestPut(unittest.TestCase):
    def test_put(self):
        t = Toy(); t.put(" A ", 1); self.assertEqual(t.items, {"a": 1})

class ToyTestGet(unittest.TestCase):
    def test_get(self):
        t = Toy(); t.put("a", 1); self.assertEqual(t.get(key=" A"), 1)

class ToyTestNorm(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(Toy.norm(" Ab "), "ab")
'''


@pytest.fixture
def toy() -> Task:
    return Task.from_skeleton(
        task_id="toy", skeleton=TOY_SKELETON, test_src=TOY_TESTS,
        slot_tests={"put": "ToyTestPut", "get": "ToyTestGet", "norm": "ToyTestNorm"},
    )


@pytest.fixture(scope="session")
def classeval_task0(rows):
    return task_from_row(rows[0])
