"""Canned slot completions for the latency benchmark (the KVStore demo task).

Four samples per slot, in the order a run draws them: the first is usually wrong, so the benchmark
exercises recombination and the search rather than measuring a lucky first composition. The bodies are
the ones the recorded demo agents produced; the point of the fixtures is that every configuration is
timed against *the same* model output, so a difference in wall clock is the harness, not sampling.

A completion is the code block a slot request asks for. `TAILS` is what a model tends to append after
it: a closing fence, a sentence of explanation, sometimes a method the prompt told it not to write.
The benchmark appends one and reports how many decoded tokens sit after the method ends, so the size
of that tail is a visible parameter and never a hidden assumption.
"""
from __future__ import annotations

SAMPLES: dict[str, list[str]] = {
    "set": [
        # forgets normalization: fails its own test
        "```python\ndef set(self, key, value):\n    self.data[key] = value\n```",
        "```python\ndef set(self, key, value):\n    self.data[key.strip().lower()] = value\n```",
        # same definition, different local names: one hash, so the store collapses it
        "```python\ndef set(self, key, value):\n    k = key.strip().lower()\n    self.data[k] = value\n```",
        "```python\ndef set(self, key, value):\n    self.data[self._norm(key)] = value\n\n"
        "def _norm(self, key):\n    return key.strip().lower()\n```",
    ],
    "get": [
        "```python\ndef get(self, key, default=None):\n    return self.data[key]\n```",
        "```python\ndef get(self, key, default=None):\n"
        "    return self.data.get(key.strip().lower(), default)\n```",
        "```python\ndef get(self, key, default=None):\n    k = key.strip().lower()\n"
        "    if k in self.data:\n        return self.data[k]\n    return default\n```",
        "```python\ndef get(self, key, default=None):\n    return self.data.get(key.lower(), default)\n```",
    ],
    "delete": [
        "```python\ndef delete(self, key):\n    if key in self.data:\n        del self.data[key]\n"
        "        return True\n    return False\n```",
        "```python\ndef delete(self, key):\n    self.data.pop(key.strip().lower(), None)\n```",
        "```python\ndef delete(self, key):\n    k = key.strip().lower()\n    if k in self.data:\n"
        "        del self.data[k]\n        return True\n    return False\n```",
        "```python\ndef delete(self, key):\n    return self.data.pop(self._norm(key), None) is not None\n\n"
        "def _norm(self, key):\n    return key.strip().lower()\n```",
    ],
    "keys_with_prefix": [
        "```python\ndef keys_with_prefix(self, prefix):\n"
        "    return [k for k in self.data if k.startswith(prefix)]\n```",
        "```python\ndef keys_with_prefix(self, prefix):\n    p = prefix.strip().lower()\n"
        "    return sorted(k for k in self.data if k.startswith(p))\n```",
        "```python\ndef keys_with_prefix(self, prefix):\n"
        "    return sorted([k for k in self.data.keys() if k.startswith(prefix.strip().lower())])\n```",
        "```python\ndef keys_with_prefix(self, prefix):\n"
        "    return sorted(k for k in self.data if prefix.strip().lower() in k)\n```",
    ],
}

TAILS: dict[str, str] = {
    "none": "",
    # the fence plus one sentence: the least a chatty server adds
    "short": "\n\nThis implementation normalizes the key before using it, as the docstring requires.",
    # the fence, an explanation, and a declared method the prompt asked it not to write
    "long": "\n\nExplanation:\n\n- The key is normalized (stripped and lowercased) before use, so that "
            "\" Foo \" and \"foo\" address the same entry.\n- Dictionary access is O(1) on average.\n"
            "- Returning early keeps the branch structure flat and easy to read.\n\n"
            "For completeness, here is how the rest of the class fits together:\n\n```python\n"
            "def get(self, key, default=None):\n    return self.data.get(key.strip().lower(), default)\n\n"
            "def delete(self, key):\n    k = key.strip().lower()\n    if k in self.data:\n"
            "        del self.data[k]\n        return True\n    return False\n```\n\n"
            "Let me know if you would like docstring examples added as doctests.",
}
