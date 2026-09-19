import ast

from calm_coder.store.normalize import canonicalize
from conftest import fn


def c(src, rename=None, drop=False, cls=None):
    return canonicalize(fn(src), rename_helper=rename or {}, drop_name=drop, class_name=cls)


def test_alpha_comments_docstring_quotes_invariant():
    a = '''
    def f(self, xs):
        """Doc."""
        total = 0  # running sum
        for x in xs:
            total += len('a' + x)
        return [y for y in [total]]
    '''
    b = '''
    def f(self, xs):
        acc = 0
        for item in xs:
            acc += len("a" + item)
        return [z for z in [acc]]
    '''
    assert c(a) == c(b)


def test_attribute_rename_changes_output():
    assert c("def f(self):\n    return self.items") != c("def f(self):\n    return self.data")


def test_params_kept_for_keyword_calls():
    # Deviation from §3.1 (see CLAUDE.md §9): parameters are interface, not locals.
    assert c("def f(self, key):\n    return key") != c("def f(self, k):\n    return k")


def test_global_nonlocal_not_renamed():
    out = c('''
    def f(self):
        global COUNTER
        COUNTER = 1
        def g():
            nonlocal_x = 2
            return nonlocal_x
        return COUNTER
    ''')
    assert "COUNTER" in out
    out2 = c('''
    def f(self):
        x = 0
        def g():
            nonlocal x
            x = 1
        g()
        return x
    ''')
    assert "nonlocal x" in out2 and "v0" not in out2.split("nonlocal")[1].split("\n")[0]


def test_roundtrip_parses():
    src = '''
    def f(self, a, *args, k=1, **kw):
        try:
            with open(a) as fh:
                data = fh.read()
        except OSError as e:
            data = str(e)
        if (n := len(data)) > 3:
            return {q: n for q in data}
        return lambda t: t + n
    '''
    ast.parse(c(src))


def test_helper_call_rewriting_all_receivers():
    src = '''
    def f(self, x):
        return self.h(x) + cls.h(x) + Cls.h(x) + h(x) + self.other(x)
    '''
    out = c(src, rename={"h": "_h_abc"}, cls="Cls")
    assert out.count("_h_abc") == 4 and "self.other" in out


def test_drop_name_for_helpers():
    assert c("def _norm(k):\n    return k", drop=True).startswith("def _h(")


def test_helper_params_renamed_fill_params_kept():
    assert c("def _a(self, k):\n    return k", drop=True) == c("def _b(self, s):\n    return s", drop=True)
    assert c("def _a(k):\n    return k", drop=True) == "def _h(v0):\n    return v0"
