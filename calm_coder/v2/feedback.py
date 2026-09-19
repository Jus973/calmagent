"""Feedback levels: how much of a failing test's output a repair prompt may carry.

The leak guard (tests never enter a prompt) has exactly one controlled exception, and this is it.

F0  which method's tests fail, nothing else.
F1  exception type + message for up to MAX_ITEMS failing tests, each truncated to MAX_CHARS.
F2  F1 plus the single source line that raised.

Only F2 can quote test source, and only the one raising line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

Level = Literal["F0", "F1", "F2"]
LEVELS: tuple[Level, ...] = ("F0", "F1", "F2")

MAX_ITEMS = 3
MAX_CHARS = 300
LEAK_WINDOW = 24       # no run of this many characters of test source may reach an F0/F1 prompt
REDACTED = "<...>"

_FRAME = re.compile(r'^  File "[^"]*", line \d+, in (\S+)$')
_EXC = re.compile(r"^(\w+(?:\.\w+)*Error|\w*Exception|AssertionError|\w+):(.*)$")


@dataclass(frozen=True)
class Failure:
    """One failing test class, reduced to what a feedback level may show."""
    test_class: str
    exc_type: str
    message: str
    raising_line: str      # source line that raised; only rendered at F2

    @property
    def interface_error(self) -> bool:
        """Interface / shared-state convention errors, as opposed to value errors."""
        return self.exc_type in ("AttributeError", "KeyError", "TypeError", "NameError", "IndexError")


def scrub(text: str, test_src: str, *, window: int = LEAK_WINDOW) -> str:
    """Remove any run of >= `window` characters that also occurs in the test source.

    Assertion messages quote the test expression that produced them, so a message can smuggle in
    test code that no feedback level is allowed to show. Redacting here means every consumer of a
    Failure is safe by construction; only `raising_line`, which F2 may show, is exempt.
    """
    if not text or not test_src:
        return text
    grams = {test_src[i:i + window] for i in range(max(0, len(test_src) - window + 1))}
    out, i, n = [], 0, len(text)
    while i < n:
        if i + window <= n and text[i:i + window] in grams:
            j = i + window
            while j < n and text[i:j + 1][-window:] in grams:
                j += 1
            out.append(REDACTED)
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def parse_failure(test_class: str, detail: str, test_src: str = "") -> Failure | None:
    """Reduce a traceback to (exception type, message, raising line)."""
    lines = [ln.rstrip() for ln in (detail or "").splitlines() if ln.strip()]
    if not lines:
        return None
    exc_type, message = "", ""
    for ln in reversed(lines):
        if ln.startswith(" "):
            continue
        m = _EXC.match(ln)
        if m:
            exc_type, message = m.group(1), m.group(2).strip()
            break
    if not exc_type:
        exc_type, message = "Unknown", lines[-1][:MAX_CHARS]
    raising = ""
    for i, ln in enumerate(lines):
        if _FRAME.match(ln) and i + 1 < len(lines) and lines[i + 1].startswith("    "):
            raising = lines[i + 1].strip()
    return Failure(test_class=test_class, exc_type=exc_type, message=scrub(message, test_src),
                   raising_line=raising)


def render(failures: Iterable[Failure], level: Level, *, slot_id: str | None = None,
           max_items: int = MAX_ITEMS, max_chars: int = MAX_CHARS) -> str:
    """The feedback block of a repair prompt. Identical shape for every arm that uses feedback."""
    fs = list(failures)[:max_items]
    if level == "F0":
        names = ", ".join(sorted({f.test_class for f in fs})) or "the class"
        target = f"`{slot_id}`" if slot_id else "this class"
        return f"The tests for {target} fail ({names}). No further detail is available."
    out = []
    for f in fs:
        item = f"{f.test_class}: {f.exc_type}: {f.message}"[:max_chars]
        if level == "F2" and f.raising_line:
            item += f"\n  raised at: {f.raising_line[:max_chars]}"
        out.append(item)
    return "The current implementation fails these tests:\n" + "\n".join(out)
