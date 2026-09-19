import ast

from calm_coder.serve.extract import ExtractError, extract_class, extract_functions


def names(r):
    assert not isinstance(r, ExtractError), r
    return [f.name for f in r]


def test_fenced():
    t = "Here:\n```python\ndef get(self, key):\n    return self._n(key)\n\ndef _n(self, k):\n    return k\n```\nDone."
    assert names(extract_functions(t, class_name="Toy")) == ["get", "_n"]


def test_unfenced():
    assert names(extract_functions("def get(self, key):\n    return 1\n", class_name="Toy")) == ["get"]


def test_indented_method():
    t = "```python\n    def get(self, key):\n        return 1\n\n    def _n(self):\n        pass\n```"
    assert names(extract_functions(t, class_name="Toy")) == ["get", "_n"]


def test_indented_method_with_decorator_unfenced():
    t = "    @staticmethod\n    def norm(key):\n        return key\n"
    (f,) = extract_functions(t, class_name="Toy")
    assert f.name == "norm" and f.decorator_list


def test_whole_class_emitted():
    t = "```python\nimport re\nclass Toy:\n    def __init__(self):\n        pass\n    def get(self, key):\n        return 1\n```"
    assert names(extract_functions(t, class_name="Toy")) == ["__init__", "get"]


def test_module_level_helper_plus_class():
    t = "```python\ndef clean(k):\n    return k\n\nclass Toy:\n    def put(self, key, value):\n        pass\n```"
    assert names(extract_functions(t, class_name="Toy")) == ["clean", "put"]


def test_unterminated_fence_and_think_tags():
    t = "<think>I think def x(: is bad</think>```python\ndef get(self, key):\n    return 1\n"
    assert names(extract_functions(t, class_name="Toy")) == ["get"]


def test_split_across_blocks():
    t = "```python\ndef put(self, key, value):\n    self.items[self._n(key)] = value\n```\nand\n```python\ndef _n(self, k):\n    return k\n```"
    assert names(extract_functions(t, class_name="Toy")) == ["put", "_n"]


def test_garbage():
    assert isinstance(extract_functions("I cannot help with that.", class_name="Toy"), ExtractError)
    assert isinstance(extract_functions("```python\ndef f(:\n```", class_name="Toy"), ExtractError)


def test_extract_class_adds_static_and_imports(toy):
    t = "```python\nclass Toy:\n    def __init__(self):\n        self.items = {}\n    def norm(key):\n        return key.strip().lower()\n```"
    src = extract_class(t, toy)
    assert src.startswith("import re")
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef))
    norm = next(n for n in cls.body if getattr(n, "name", "") == "norm")
    assert ast.unparse(norm.decorator_list[0]) == "staticmethod"


def test_extract_class_from_bare_methods(toy):
    src = extract_class("```python\n    def get(self, key):\n        return 1\n```", toy)
    assert "class Toy" in src


def test_extract_class_garbage(toy):
    assert isinstance(extract_class("nope", toy), ExtractError)
