# CALM Coder — build spec (HackMIT 2026, ~24h, Python, local models)

> **One line for judges.** A shared file is a last-writer-wins register, so every merge is an ordering decision. A grow-only, hash-keyed set of definitions has no ordering decision to make. CALM tells you that is exactly the line between "needs coordination" and "doesn't."

> **One line for reviewers.** Parsel-style per-function sample-and-verify, rebuilt on a coordination-free store, with a CALM analysis of where the single barrier actually is — evaluated on ClassEval against equal-budget whole-class sampling.

This file is the repo's `CLAUDE.md`. Claude Code reads it every session. §0–§1 are rules; §2–§7 are the spec; §8 is the schedule; §9 is the running decision log; §11 maps sponsor tracks onto work the spec already does. If context gets tight, move §2–§7 to `docs/SPEC.md` and leave a pointer.

---

## 0. How to work in this repo (read every session)

- Python 3.11+. Allowed deps: `httpx`, `rich`, `datasets`, `numpy`, `matplotlib`, `pytest`, `hypothesis` (optional). Ask before adding anything else. No frameworks, no ORMs, no plugin systems. Dataclasses, sets, `subprocess`, `asyncio`.
- §8 is a **priority order**, not a wish list. When behind, cut from the bottom. Never polish an upstream module while the downstream one it feeds doesn't exist yet.
- `calm_coder/store/*` and `calm_coder/runner/*` get unit tests **before** anything depends on them. `pytest -q` passes before every commit. Small commits, imperative messages.
- The invariants in §1 are non-negotiable. If a task seems to need a deletion, a mutation, or a "pick the best" step in the store or derivation layer, stop and log the conflict in §9 instead of working around it. Working around it silently is the one way to make the demo a lie.
- Experiments write **append-only JSONL** under `runs/<UTC-timestamp>_<name>/`. Never overwrite a run.
- When a ClassEval task doesn't fit this spec (odd skeleton, missing field, non-stdlib import, flaky reference), **exclude the task and log why** in `runs/.../excluded.jsonl`. Do not special-case tasks in library code.
- Agents never see test code. Tests are the verifier, not the prompt (§6.1).
- Determinism: every sample carries `(task_id, arm, slot, seed, sample_idx)`. Seeds are logged. Temperature is a run-level constant.
- **Sponsor tracks (§11) are packaging, not scope.** A sponsor item may add a metric computed from logs we already write, a backend selected by env vars, a thin CLI over existing modules, or a paragraph in the writeup. It may not change I1–I3, the benchmark, the subset, the baselines, the pre-registered hypotheses, the headline model choice, or the demo. If a sponsor requirement would, drop the sponsor and log it in §9. Sponsor work is always the first thing cut (§8).

---

## 1. Invariants — the CALM part, stated precisely

**I1. Grow-only.** The store has exactly two write operations: `add_def` and `add_outcome`. Both are idempotent set inserts. There is no update, delete, retract, or "replace." A `grep -nE "\b(del|remove|pop|discard|clear|update)\(" calm_coder/store` must return nothing (add this as a test).

**I2. Content-addressed, two-level references.** A definition is keyed by the hash of its canonical form. Inside a definition there are two kinds of references, and they are treated differently:
- **Slot references** (`self.parse_arguments(...)`, where `parse_arguments` is a method declared in the skeleton) are **late-bound**: they refer to the interface, and a *composition* decides which fill satisfies them. This is what makes N fills per slot combinatorially composable.
- **Helper references** (calls to private helpers the *same emission* defined, e.g. `self._split_kv(...)`) are **frozen to the helper's hash at emit time** and rewritten to `_h_<hash12>` in the canonical form. This is the Unison move: helper *names* are metadata; identity is the body.

Freezing slot refs to fill hashes (as in pure Unison) would make each fill composable with only one specific fill of its dependencies — that kills the combinatorial search. Don't do it.

**I3. Derivations are monotone and pure.** Everything derived from the store (`complete`, `slot_evidence`, `verified`, `done`) is a function of the *sets* of defs and outcomes, uses only ∃ / ∧ / transitive closure over immutable data, and contains no negation, no max/argmax, no "latest," no count-based threshold that can flip back. Given the same set of base events in any order, derived facts are identical (confluence). This is tested (§3.4, §6.4).

**What we claim.** The *harness layer* (store + derivations + exit) is monotone; therefore N agents and M test workers can add to it concurrently with no locks, no merge step, and no ordering decisions, and the success exit (`∃ verified composition`) is coordination-free: any worker that observes it may announce completion.

**What we do not claim.** The *agent layer* (an LLM) is not confluent — the set of definitions emitted depends on sampling and on which feedback an agent saw. That is fine: CALM says confluence is required of the *program's output*, and our output is "any verified composition," which is monotone. The one genuine barrier is the **failure exit**: declaring "no solution within budget" is a ∀ over agents ("everyone finished, nothing verified"), so it needs a barrier. Stopping agents early after success is a courtesy, not a correctness requirement — late emissions are inert.

**Non-monotone temptations to refuse:** pruning failed fills; keeping only "the best" fill per slot; reading "the latest" fill; treating dedup as deletion (dedup is a *derived count*, the store just doesn't grow); early-stopping a slot because "enough passed." Scheduling *policies* may use any of this information to decide what to test next — that's fine, because scheduling affects *when* facts appear, not *which* facts are derivable.

---

## 2. Concepts and data model

```python
# calm_coder/store/defs.py
from dataclasses import dataclass, field
from typing import Literal, Mapping, FrozenSet

@dataclass(frozen=True)
class Slot:
    id: str                 # method name, e.g. "parse_arguments"
    signature: str          # "def parse_arguments(self, command_string):"
    docstring: str          # from the skeleton (includes the >>> examples)
    is_static: bool         # @staticmethod in skeleton
    order: int              # declaration order in skeleton (for materialization)

@dataclass(frozen=True)
class Definition:
    hash: str                        # sha256 hex of canonical form (see §3.2)
    kind: Literal["fill", "helper"]
    slot: str | None                 # fills: slot id; helpers: None
    name: str                        # as emitted (helpers keep this only as metadata)
    canonical_src: str               # normalized, helper calls rewritten to _h_<hash12>
    raw_src: str                     # exactly as emitted (for debugging)
    slot_refs: FrozenSet[str]        # late-bound references to skeleton slots
    helper_refs: FrozenSet[str]      # hashes of helpers this def calls (frozen)
    unresolved_refs: FrozenSet[str]  # self.<name>() where name is neither a slot nor an emitted helper
    is_static: bool
    meta: Mapping[str, object] = field(default_factory=dict)  # agent_id, seed, sample_idx, ts, completion_tokens

@dataclass(frozen=True)
class Composition:
    id: str                          # sha256 of sorted (slot_id, fill_hash) pairs
    bindings: Mapping[str, str]      # slot_id -> fill hash; unbound slots become stubs at materialization

@dataclass(frozen=True)
class Outcome:
    test_id: str                     # unittest class name, e.g. "ArgumentParserTestParseArguments"
    comp_id: str
    result: Literal["pass", "fail", "error", "timeout", "inconclusive"]
    detail: str                      # truncated traceback / stub name
    wall_ms: int
```

`inconclusive` = the test hit a stubbed slot (a `CalmStubHit` exception was raised). It is a *positive fact* ("this test needs slot X filled"), not a failure.

```python
# calm_coder/store/store.py
class Store:
    def add_def(self, d: Definition) -> bool: ...      # True if new. Idempotent. Appends to event log.
    def add_outcome(self, o: Outcome) -> bool: ...     # True if new. Idempotent. Appends to event log.
    def defs(self) -> FrozenSet[Definition]: ...
    def defs_for_slot(self, slot_id: str) -> FrozenSet[Definition]: ...
    def helpers(self) -> Mapping[str, Definition]: ...  # hash -> helper
    def outcomes(self) -> FrozenSet[Outcome]: ...
    def event_log(self) -> list[dict]: ...             # append-only, JSON-serializable, for replay
```

```python
# calm_coder/store/derive.py — pure functions of (store, task). No mutation. No negation.
def complete(store, task, comp) -> bool:
    """Every slot that any bound fill references (transitively through helpers) is bound in comp,
    every helper hash referenced is present in the store, and no bound def has unresolved refs.
    For a fixed comp this can only go False -> True as the store grows."""

def slot_evidence(store, task, d: Definition) -> dict:
    """Counts of {pass, fail, error, timeout, inconclusive} for outcomes whose test_id is d.slot's
    own test class and whose composition binds d. Used by the *scheduler* only."""

def verified(store, task) -> FrozenSet[str]:
    """comp_ids for which, for every test class t in task.test_classes, there exists an
    Outcome(t, comp_id, 'pass'). A universal over a fixed finite set, evaluated per comp:
    monotone because outcomes are never retracted."""

def done(store, task) -> bool:
    return len(verified(store, task)) > 0      # ∃ — the coordination-free exit
```

---

## 3. Module specs with acceptance tests

Repo layout:

```
calm_coder/
  store/     defs.py  normalize.py  store.py  derive.py  materialize.py
  serve/     client.py  prompts.py  extract.py
  agents/    fill.py  scheduler.py
  runner/    sandbox.py  _run_tests.py  tests.py
  bench/     classeval.py  baselines.py  experiment.py  metrics.py  charts.py
  viz/       live.py
  demo/      task.py  git_baseline.py  run_demo.py
  cli.py     (`calm solve`, §3.18)
tests/       (pytest)
runs/        (gitignored)
```

### 3.1 `store/normalize.py` — canonical form

`canonicalize(fn: ast.FunctionDef, *, rename_helper: dict[str, str], drop_name: bool) -> str`

1. Deep-copy the node. Remove a leading docstring `Expr(Constant(str))` from the body.
2. **Alpha-rename locals.** Collect names bound in the function: parameters (all kinds), assignment/aug-assign/ann-assign targets, `for`/`with … as`/`except … as` targets, comprehension targets, walrus targets, nested `def`/`lambda` names and params. Skip anything declared `global`/`nonlocal`. Map each distinct name to `v0, v1, …` in order of first appearance. Rewrite `Name` nodes (Load/Store/Del) with those ids. Never touch `Attribute.attr`, keyword-argument names in calls, or import names. `self`/`cls` first parameter is kept as-is (it's part of the interface).
3. **Rewrite helper calls.** For `Attribute(value=Name('self'|'cls'|<ClassName>), attr=n)` and bare `Name(n)` calls where `n ∈ rename_helper`, set the attr/name to `rename_helper[n]` (`_h_<hash12>`).
4. If `drop_name` (helpers), set `fn.name = "_h"`; fills keep the slot name.
5. Return `ast.unparse(fn)` (Python ≥ 3.9; drops comments and normalizes formatting/quotes).

Acceptance: (a) two functions differing only in local variable names, comments, docstring, and quote style produce identical output; (b) renaming an attribute (`self.items` → `self.data`) changes the output; (c) a global/nonlocal name is not renamed; (d) round-trip `ast.parse(canonicalize(x))` succeeds; (e) helper call rewriting hits `self.h(`, `cls.h(`, `Cls.h(`, and bare `h(`.

Fallback if alpha-renaming isn't done by H3 (see §8): skip step 2, report dedup at the "exact-unparse" level, and say so on the chart.

### 3.2 `store/defs.py` — hashing and emission → definitions

`emission_to_defs(task, slot, emitted_fns: list[ast.FunctionDef], meta) -> list[Definition]`

- Partition emitted functions: the one named `slot.id` is the **fill**; every other function is a **helper** (regardless of leading underscore — models are inconsistent). A second function named like *another* slot is dropped and logged (`stray_slot_def`), never stored as a fill for that slot.
- Build the helper call graph among emitted functions. Hash helpers in dependency order: `sha256("helper|" + canonicalize(h, rename_helper=already_hashed, drop_name=True))`. On a cycle, hash the whole SCC as one unit (sorted canonical sources concatenated) and give every member the same hash prefix with an index suffix; log `cyclic_helpers`.
- Fill hash: `sha256("fill|" + slot.id + "|" + canonicalize(fill, rename_helper=helper_map, drop_name=False))`.
- `slot_refs`: names `n` in `self.n(` / `cls.n(` / `Cls.n(` where `n ∈ task.slot_ids` (excluding `__init__`). `helper_refs`: hashes of emitted helpers actually called (transitively closed at materialization, not here). `unresolved_refs`: `self.n(` where `n` is neither a slot nor an emitted helper nor a field-typed call we can't see (accept the over-approximation).
- Field references (`self.x`) are **not** tracked as dependencies. The constructor is fixed by the skeleton; fields are the shared contract. (Optional metric: record `field_refs` for post-hoc analysis of interface mismatch.)

Acceptance: same body → same hash across two "agents"; identical helper under two different names → one helper hash, and two fills that differ only in the helper's name → same fill hash; a fill calling a helper with a *different body* → different fill hash.

### 3.3 `store/store.py`

In-memory sets + append-only event log (`{"t": ts, "kind": "def"|"outcome", "data": {...}}`). `add_*` returns `False` on duplicates and **still does not log** (idempotent). Provide `Store.from_events(events)` for replay. Thread-safe enough for asyncio (single-threaded event loop) — test workers report back into the loop; don't share the store across processes.

Acceptance: I1 grep test; `add_def` twice → one def, one event; `from_events(shuffle(log))` yields equal `defs()`/`outcomes()`.

### 3.4 `store/derive.py`

As in §2. Acceptance (the confluence test, also run on real logs in §6.4): for a fixed task, take a real event log, replay it in 20 random orders into fresh stores, and assert `frozenset(map(dedupkey, defs))`, `verified(...)`, `done(...)`, and `{comp: complete(...)}` over all seen comps are identical across replays. Also: `verified` never shrinks as events are appended one by one (monotonicity property test).

### 3.5 `store/materialize.py`

`materialize(task, store, comp, *, stub_unbound=True) -> str`

Emits, in order: `"\n".join(task.import_statement)`; the class header + class docstring; the constructor **verbatim from the skeleton** (`class_constructor` body, or nothing if the skeleton has no `__init__`); then for each slot in skeleton order: the bound fill's `raw_src` with helper calls rewritten to `_h_<hash12>` (rewrite on a parsed AST, then `ast.unparse`), with `@staticmethod` added when `slot.is_static` and the def has no `self`/`cls` first param; unbound slots get

```python
    def <name>(<signature args>):
        raise CalmStubHit("<name>")
```

Then all helpers reachable from bound fills (transitive closure over `helper_refs`), each once, named `_h_<hash12>`, as `@staticmethod` when the first param isn't `self`/`cls`. Call-site rewriting keeps the receiver form used at the call site (`self.X(` → `self._h_x(`; `Cls.X(` → `Cls._h_x(`; bare `X(` → `Cls._h_x(` if helper is static else `self._h_x(`). Prepend `class CalmStubHit(Exception): pass` to the module.

Mirror the official ClassEval pipeline's `add_static_statement` behavior: a method at class indentation whose signature lacks `self`/`cls` gets `@staticmethod`.

Acceptance: materializing the reference solution's methods (parsed as fills) reproduces a class that passes the task's full test module; two fills from different emissions that both define a helper named `_norm` materialize with two distinct `_h_*` methods and no name clash; a composition with one unbound slot materializes and `ast.parse`s.

### 3.6 `serve/client.py` — model serving abstraction

Any OpenAI-compatible chat endpoint. `Client.sample(messages, n, temperature, max_tokens, seed) -> list[Sample(text, completion_tokens, prompt_tokens, latency_ms)]`. Use `n` when the server supports it (vLLM does); env `CALM_NO_N=1` falls back to `n` concurrent requests (Ollama, some llama.cpp builds). `asyncio` + `httpx`, global concurrency semaphore (`CALM_MAX_INFLIGHT`, default 32). Token counts come from `usage`; if absent, estimate with `len(text)//4` and flag `tokens_estimated=True` in meta. Also record `usage.prompt_tokens_details.cached_tokens` when the server returns it; otherwise record `cached_tokens=null` (unknown, **never estimated**). For vLLM, snapshot the server's `/metrics` prefix-cache counters at run start and end (metric names vary by vLLM version; log whatever is exposed).

Preferred server: `vllm serve <model> --seed 0 --enable-prefix-caching --max-num-seqs 256` on an NVIDIA GPU. Fallbacks: llama.cpp server (`--parallel 8 -c 8192`), Ollama (`OLLAMA_NUM_PARALLEL=8`), `mlx_lm.server` on Apple Silicon. The whole system must run against any of these with only env vars changed (`CALM_BASE_URL`, `CALM_MODEL`).

Hosted variants (§11), same code path: **rented GPU** (e.g. Runpod, if its challenge provides compute) = the vLLM command above on the pod, only `CALM_BASE_URL` changes. **Hosted API** (optional secondary arm only) = `CALM_BASE_URL=https://api.openai.com/v1`; `n` and `seed` are accepted but determinism there is best-effort, so log `system_fingerprint` and treat reruns as non-reproducible. A hosted frontier model is never the headline model: it is likely at ceiling on ClassEval and likely trained on it (see below).

**Model choice is empirical (§6.2).** Pick the largest model that gives holistic pass@1 in the 30–55% band on a 10-task pilot; if >60%, step down a size (ceiling kills the lift hypothesis). Candidate list to verify against what's actually installable this weekend: Qwen2.5-Coder-{3B,7B,14B}-Instruct, Qwen3-{4B,8B}, Qwen3-Coder-30B-A3B-Instruct (MoE; fits 24 GB at 4-bit), DeepSeek-Coder-V2-Lite-Instruct. Note ClassEval (2023) is likely in recent models' training data; it's why we prefer the smallest model that still functions.

Acceptance: smoke test hits the endpoint with `n=4`, gets 4 texts and a usage block.

### 3.7 `serve/prompts.py`

**Slot fill** (system): *You are implementing exactly one method of a Python class. The method to implement is named in the last line of the user message. Output a single Python code block containing that complete method with its signature exactly as in the skeleton. You may also define private helper methods you need (names starting with `_`). Do not include the class header, `__init__`, imports, or other declared methods. You may call other declared methods via `self.<name>()` and rely on their docstrings. No prose.*
(user): the **full skeleton** (imports + class + constructor + all signatures/docstrings — same context the ClassEval "compositional" strategy gives) + *"Implement `<name>` now."*

Prefix rule: the system prompt and skeleton are byte-identical across all slots of a task, and the slot name comes **last**, so every slot request for a task shares one prompt prefix (vLLM prefix caching; cached-input pricing on hosted APIs). This only reorders the same content ClassEval's compositional prompt gives, so it changes cost, not what the model is told. Never put anything slot-specific before the skeleton. Acceptance: for a task, the prompts for any two slots share a common prefix covering all but the final line.

**Whole class** (baselines A/C): the ClassEval holistic prompt — skeleton + *"Complete the class. Output one Python code block containing the full class."*

**Incremental** (baseline B, optional): skeleton with previously generated methods filled in; implement the next one.

**Repair** (optional arm, §6.3): slot-fill prompt + *"A previous implementation failed with:"* + truncated traceback (≤ 40 lines). Still emits a fresh definition; nothing is edited.

Sampling defaults: `temperature=0.8, top_p=0.95` when `n>1`; `max_tokens=512` for slots, `2048` for whole class. Record in run config.

### 3.8 `serve/extract.py`

`extract_functions(text, *, class_name) -> list[ast.FunctionDef] | ExtractError`

Try, in order: fenced ```` ```python ```` blocks; any fenced block; the raw text. For each candidate: `ast.parse(textwrap.dedent(src))`; if that fails, try `ast.parse("class _W:\n" + indent(src))` (models often emit an indented method); if the parse yields a `ClassDef` (model emitted the whole class), take its body. Collect `FunctionDef`/`AsyncFunctionDef` nodes at the top level of whatever parsed. Return `ExtractError(reason)` if nothing parses; the caller records it as a positive fact (`Outcome(test_id="__extract__", result="error")` keyed by a synthetic comp id) so failures are visible in metrics without any deletion.

Whole-class extraction (baselines) mirrors the official pipeline: fence → class-indent fix → `add_static_statement` → prepend `import_statement`.

Acceptance: fixtures for fenced/unfenced/indented/whole-class/garbage outputs.

### 3.9 `runner/sandbox.py` + `runner/_run_tests.py`

`run_tests(module_src: str, test_src: str, test_classes: list[str], *, per_class_timeout_s=5, wall_timeout_s=20) -> dict[test_id, Outcome-like]`

- Fresh `tempfile.TemporaryDirectory()` as cwd (several ClassEval tasks write files). Write `mod.py = module_src + "\n\n" + test_src`.
- `subprocess.run([sys.executable, "-I", _run_tests.py, "mod.py", *test_classes], timeout=wall_timeout_s, capture_output=True)`.
- `_run_tests.py`: `resource.setrlimit` (AS 1 GiB, CPU 15 s, NOFILE 256); import `mod.py` by path; for each test class: `signal.alarm(per_class_timeout_s)`, `unittest.TestLoader().loadTestsFromTestCase(cls)`, run with a quiet `TextTestRunner`, collect `testsRun/failures/errors`, first traceback (truncated to 2 KB); result = `pass` if all pass, `inconclusive` if any traceback mentions `CalmStubHit`, `timeout` on alarm, else `fail`/`error`. Print one JSON object to stdout. Import failure of `mod.py` → every requested class gets `error` with the import traceback.
- Parallelism: `asyncio.to_thread` or a `ThreadPoolExecutor(max_workers=os.cpu_count())` around `subprocess.run`; each run is its own process anyway.

This is stricter than the official pipeline (which imports generated code in-process with a 5 s `func_timeout`). It's a sandbox against runaway loops and stray file writes, not against adversarial code.

Acceptance: reference solutions for the selected subset pass 3/3 runs; a `while True` method times out and reports `timeout` without hanging the pool; a stubbed slot yields `inconclusive`.

### 3.10 `runner/tests.py`

- `slot_test_class(task, slot) -> str`: from `methods_info[i]["test_class"]` if present, else regex `class (\w+)\(unittest\.TestCase\)` on `methods_info[i]["test_code"]`.
- `class_level_tests(task) -> list[str]`: `task.test_classes` minus all slot test classes. Naming is **not** consistent (`BankAccountTest`, `AvgPartitionTestMain`, `AreaCalculatorTestCalculateMain` all occur) — never infer by suffix.
- `test_slot_deps(task, test_class_src) -> set[str]`: over-approximate: every `.<name>(` attribute call in the test class body where `name ∈ slot ids`. Used by the scheduler to implicate slots and to predict `inconclusive` up front.

### 3.11 `bench/classeval.py`

`load_dataset("FudanSELab/ClassEval")["test"]` (100 rows). Fields used: `task_id, skeleton, test, solution_code, import_statement (list[str]), class_name, test_classes (list[str]), class_constructor, methods_info (list[dict])`.

`Task` construction: parse `skeleton` with `ast`; find the single `ClassDef`; slots = its `FunctionDef`s except `__init__`, in order, with `is_static` from decorators; signature = header line rebuilt via `ast.unparse(node.args)`; docstring via `ast.get_docstring`. Cross-check slot names against `methods_info[*].method_name` — mismatch → exclude + log.

**Subset selection (`select_subset(n=50, seed=0)`)**, applied once and frozen to `data/subset.json`:
1. Skeleton parses; exactly one class; body is docstring + `__init__` (optional) + methods only (no nested classes/constants).
2. All imports are stdlib (`sys.stdlib_module_names`).
3. Materializing the reference methods as fills passes the full test module **3/3** in our sandbox (filters flaky/time/random-dependent tests and environment issues).
4. 2 ≤ #slots ≤ 8.
5. From survivors, sample 50 with the fixed seed; log the exclusion reason for every dropped task.

Reference facts (verified 2026-09-19 against the repo/HF card): 100 classes, 410–412 methods, avg 33.1 tests/class; avg 4.97 methods/class, avg method-dependency depth 1.77 (median 2); constructors are given **with bodies** in the skeleton; method test classes routinely call *other* methods (e.g. `ArgumentParserTestParseArguments` calls `add_argument`); the official pipeline runs every test class against the whole generated class — there is no ground-truth substitution for method-level tests.

### 3.12 `agents/fill.py` + `agents/scheduler.py` — the CALM arm

Per task, budget `N` fills per slot:

**Phase 0 — generate.** For every slot concurrently: one `sample(n=N)` call with the slot prompt. Extract → `emission_to_defs` → `store.add_def` for each def. Record `Emission` rows (raw text, tokens, seed) to JSONL regardless of extraction success.

**Phase 1 — stub-context slot tests** (cheap, fully parallel). For each fill `d`: comp = `{d.slot: d.hash}` (everything else stubbed); run **only** `slot_test_class(d.slot)`. Outcomes are `pass` / `fail` / `error` / `timeout` / `inconclusive`. Skip when `test_slot_deps` already shows the test needs another slot (record `inconclusive` directly with detail `predicted`). Fills with `unresolved_refs` still get tested (they'll error; that's a fact).

**Phase 2 — composition search** (the only place ordering policy lives).
```
rank(d) = (evidence pass count desc, inconclusive desc, fan_in desc, arrival asc)  # fail/error/timeout last
cands[s] = sorted(store.defs_for_slot(s), key=rank)         # every slot needs ≥1 fill, else task = unsolved
frontier = deque([Composition({s: cands[s][0].hash for s in slots})]); tried = set()
while frontier and len(tried) < MAX_COMPS (default 64):
    comp = frontier.popleft()
    if comp.id in tried or comp.id already has outcomes for all test classes: continue
    tried.add(comp.id)
    outcomes = run_full_module(comp)          # ALL test classes, one subprocess
    add all outcomes to store
    if done(store, task): break               # ∃ verified — the exit
    implicated = ∪ over failing test classes t of (test_slot_deps(t) ∪ slot names in t's traceback frames) or all slots if empty
    for s in sorted(implicated, key=lambda s: alternatives_remaining(s, comp)):
        alt = next untried candidate for s given the rest of comp (walk cands[s] in rank order)
        if alt: frontier.append(comp.with(s, alt))
```
Everything the scheduler reads is a monotone set; everything it writes goes through `add_outcome`. Change the policy freely — it cannot change which facts are derivable, only how fast they appear.

**Phase 3 — repair round (OFF by default; separate arm `calm+repair`).** For the most-implicated slot, request `N_repair` new fills with the repair prompt (§3.7), then rerun Phase 1–2. Still grow-only.

Timestamps: `t_start`, `t_first_def`, `t_first_slot_pass`, `t_first_verified` (TTFV), `t_end`; plus separate accumulators for model wall time and test wall time.

### 3.13 `bench/baselines.py`

- **A — holistic pass@1.** Estimated with the unbiased pass@k estimator from C's samples (so A costs nothing extra), plus one greedy (`temperature=0`) sample reported separately as `A_greedy`.
- **B — incremental, single pass** (optional, cut first): methods in declaration order, previously generated methods in context, one sample each; optional single repair per method on its slot test. Report separately; it's the ClassEval "incremental" strategy with a verifier.
- **C — holistic pass@N with oracle verification** (the fair baseline). `N` whole-class samples at the same temperature; extract per the official pipeline; run the full test module per sample; solved iff any sample passes everything. Report `C@N` and **`C@tokens`**: use only the first `M ≤ N` samples whose cumulative completion tokens fit within the CALM arm's measured completion tokens for that task. Both are reported; `C@tokens` is the headline comparison.

Note: `CALM@1` (one fill per slot, no search) *is* ClassEval's "compositional" strategy with a verifier, and `A` is its "holistic." The N-sweep therefore shows the compositional penalty at N=1 and how much of it sampling + search recovers.

### 3.14 `bench/experiment.py`

`python -m calm_coder.bench.experiment --arms calm,c --N 8 --seeds 0,1,2 --subset data/subset.json --out runs/`
- Generate at `N_max=8` once per (task, arm, seed); evaluate `N ∈ {1,2,4,8}` by **nested subsampling** (first `n` fills per slot / first `n` class samples). No regeneration.
- One JSONL row per (task, arm, seed, N) with: solved, TTFV, compositions tested, completion tokens, prompt tokens, #defs, #distinct defs (exact / alpha), per-slot stub-pass counts, verified comp id, excluded flags.
- Resumable: skip (task, arm, seed) already present in the output.

### 3.15 `bench/metrics.py` + `bench/charts.py`

Per arm × N (and paired across arms on the same tasks):
1. **Class-level solve rate** with Wilson 95% CI; paired difference CALM@N − C@N with a 10k-resample bootstrap CI over tasks; McNemar exact p for CALM@8 vs C@tokens.
2. **Time-to-first-verified**: absolute median/p90 per arm; per-task ratio CALM/C on co-solved tasks (median, IQR).
3. **Independence prediction vs observed** (the coupling metric): per task, `p_s` = stub-context pass fraction for slot `s` (pass / (pass + fail + error + timeout), inconclusive excluded; if all inconclusive, fall back to the fraction of fills that appear in ≥1 composition passing that slot's test). Predicted `P_indep(N) = Π_s (1 − (1 − p_s)^N)`. Plot predicted vs observed solve rate across N; the gap is coupling.
4. **Interface-mismatch rate**: among tested compositions whose every bound fill has a stub-context `pass`, the fraction that fail the full module. **Systematic-mismatch rate (the falsifier F1)**: fraction of tasks where every slot has ≥1 stub-passing fill yet no composition verified within `MAX_COMPS`.
5. **Held-out variant** (post hoc, same logs): first composition that passes all *method-level* test classes — did it also pass the *class-level* classes? This is the honest "select on unit tests, score on integration tests" number.
6. **Dedup**: `(emitted − distinct)/emitted` at exact-unparse and alpha-normalized levels; fan-in histogram. State plainly that dedup saves test executions, not tokens.
7. **Unreachable fraction**: defs not in any verified composition / total (only on solved tasks). **Compositions tested before first verified**: distribution.
8. **Confluence test**: pass/fail count over real logs.
9. **Cost** (feeds H6 and §11; computed from existing logs, **no new runs**). Per arm × N: prompt, cached-prompt, and completion tokens. (a) *Tokens per verified class* = total tokens / #solved. (b) *Tokens-to-first-solve under a doubling schedule*: from nested subsampling, the smallest n ∈ {1,2,4,8} at which the arm solves the task; spend = completion tokens of the first n samples plus one prompt per wave actually sent (1, 2, 3 or 4 waves); unsolved tasks are charged the full N=8 spend. This is the "stop when verified" saving: legitimate because the exit is ∃ (§1), so stopping after success is a courtesy. (c) Dollars only via an explicit price table in the run config (input / cached input / output per 1M tokens), labeled as a *what-if price*, not a measurement. State the known overhead plainly: CALM sends the skeleton once per slot, C once per task. Caching shrinks that gap but doesn't close it.

Charts (matplotlib, PNG + SVG): (i) solve rate vs N for A/C@N/C@tokens/CALM with CIs; (ii) predicted-under-independence vs observed for CALM; (iii) TTFV CDFs; (iv) dedup vs N; (v) mismatch rates; (vi) cost frontier: solve rate (y) vs mean tokens per task (x) for CALM and C at N ∈ {1,2,4,8}, prompt and completion stacked, with cached share shaded if known. Every chart caption states temperature, model, subset size, seeds.

### 3.16 `viz/live.py`

`rich.live.Live` view fed by the store's event log: a slots × fills grid (cell = `hash[:8]`, colored by best evidence), counters `definitions / conflicts (always 0, by construction) / duplicates collapsed / compositions tested / verified roots`, and an event ticker. The root row turns green on the first verified composition. Stretch (cut first): D3 DAG in a browser over SSE.

### 3.17 `demo/`

See §7.

### 3.18 `cli.py` — `calm solve` (thin; Warp track, §11)

`python -m calm_coder.cli solve path/to/skeleton.py --tests path/to/test_x.py --N 4 [--live] [--out solved.py]`

A wrapper over existing modules only: `Task.from_files(skeleton, tests)` (same parsing as §3.11; test classes discovered by regex as in §3.10), Phases 0–2, then materialize the first verified composition to `--out`/stdout. `--live` attaches `viz/live.py`. Exit code 0 on verified, 1 on budget exhausted: the failure exit is the one barrier (§1), so it waits for all phases to finish. No new logic: if the wrapper needs something the library doesn't expose, log it in §9 rather than adding a special case.

Acceptance: `calm solve demo/task.py --tests demo/test_task.py --N 4` produces a class that passes the demo tests; running it on a ClassEval task exported to two files matches `experiment.py`'s result for the same seed.

---

## 4. Prior art to cite (and the honest contribution)

- **Parsel** (Zelikman et al., NeurIPS 2023): decompose into function descriptions, sample implementations per function, search over combinations with tests. The closest prior work — cite it first, unprompted. **Hypothesis Search** (Wang et al., 2023) recombines per-function implementations similarly; **FunCoder** (Chen et al., 2024) does divide-and-conquer with functional consensus.
- **Large Language Monkeys** (Brown et al., 2024): coverage scales log-linearly with repeated sampling given a verifier. **AlphaCode** (2022): sample–filter–cluster.
- **ClassEval** (Du et al., 2023): holistic vs incremental vs compositional generation. Holistic wins for strong models by 6–14 pts class-level pass@5; compositional (= our N=1) loses. **ClassEval-Pro** (Chen et al., Apr 2026): on 300 deeper-dependency classes, compositional pass@1 collapses to 1.3–21.7% vs 28–46% holistic; error analysis attributes 38% of failures to dependency errors. **ClassEval-TDD** (2026): repaired specs/tests; compositional "remains limited by cross-method composition."
- **Unison**: content-addressed definitions, names as metadata. **CALM**: Hellerstein & Alvaro, *Keeping CALM* (CACM 2020); theorem proved by Ameloot, Neven & Van den Bussche. **Bloom^L** (Conway et al., 2012): lattices for distributed programming — the store is a G-Set CRDT and the derivations are a monotone Datalog program.

**Contribution, stated without inflation:** (1) a CALM analysis of a multi-agent code-generation harness that locates the single genuine barrier (the failure exit) and shows the success path is coordination-free; (2) a content-addressed grow-only store that makes *N fills per slot* first-class and conflict-free and makes assembly mechanical (the literature's "compositional collapse" conflates semantic coupling with fragile assembly — the store eliminates the second by construction so the experiment can measure the first); (3) a confluence self-test; (4) an independence-vs-observed coupling metric. Not a contribution: per-function sample-and-verify itself (Parsel).

---

## 5. Pre-registered hypotheses (write these down before the first full run)

| ID | Claim | Threshold | Prior P | If it fails |
|----|-------|-----------|---------|-------------|
| H1 | CALM@8 beats holistic pass@1 (A) | ≥ 1.5× relative | 0.65 overall; ~0.80 if A ≤ 50% | Ceiling effect → pick a smaller model (§3.6); report absolute pts |
| H2 | CALM@8 beats equal-budget whole-class sampling (C@tokens) | paired difference > 0, McNemar p < 0.1 | 0.55 | Granularity loses on this benchmark; systems result (0 conflicts, confluence, TTFV) stands; report coupling as the finding |
| H2b | … by a margin | ≥ 1.2× relative | 0.35 | — |
| H3 | Faster to first verified class | median per-task TTFV ratio CALM/C ≤ 0.5 on co-solved tasks | 0.70 | Per-method decode isn't shorter in practice (long methods) or test time dominates |
| H4 | Dedup is real | ≥ 10% of emitted fills collapse at N=8, T=0.8 (alpha-normalized) | 0.65 (0.50 exact-unparse) | Methods too long/diverse; report fan-in anyway |
| H5 | Confluence | 20 shuffled replays, 0 derived-fact diffs, on every logged run | 0.97 | It's a bug — fix it; this one is not allowed to fail |
| H6 | Cheaper per verified class than C at the same model | median tokens-to-first-solve ratio CALM/C < 1 (doubling schedule, prompt + completion, §3.15 item 9) **and** lower total tokens per solved task | 0.45 | Per-slot prompt overhead outweighs shorter completions. Report the frontier chart (vi) and where the crossover N is; the cost claim becomes "cheaper in regime X", or is dropped |
| **F1** | **Falsifier** | systematic-mismatch rate > 30% of tasks | P(occurs) ≈ 0.25 | Kills H2/H2b; becomes the headline finding ("where coupling lives") |

H1 is not an equal-budget claim (single-shot uses 1/N of the tokens); it's the sanity check that sampling + a verifier helps at all. H2 is the real bet, and it is a coin flip on purpose — that's what makes it worth running. H6 is H2 re-expressed in total cost (prompt tokens included), so it is strictly harder than H2 on prompt tokens and easier on stop-when-verified. It costs nothing to compute and is pre-registered for the same reason.

---

## 6. Experiment protocol

### 6.1 Verifier = oracle tests, hidden from agents
Both CALM and C select with the same ClassEval tests, so the headline number is **coverage** (oracle-verified pass@N), exactly as in *Large Language Monkeys*. Say so on every chart. The question the experiment answers is *how to spend samples given a verifier*, not how to verify. The held-out metric (§3.15 item 5) is the stricter secondary number.

### 6.2 Pilot (first hour after serving works)
10 tasks, holistic greedy + `n=8` at T=0.8. Record holistic pass@1 and aggregate tokens/s. Choose the model per §3.6. Compute the projected run time for §6.3 before launching it.

Budget arithmetic (per seed, 50 tasks, N=8, ~5 slots, ~120 completion tokens/method, ~450/class): CALM ≈ 50·8·5·120 ≈ 240k tokens; C ≈ 50·8·450 ≈ 180k. ~420k completion tokens per seed. At 1,500 tok/s aggregate ≈ 5 min; at 200 tok/s ≈ 35 min. Three seeds fit on a single GPU; on a laptop, run one seed and bootstrap over tasks.

### 6.3 Main run
`--arms calm,c --N 8 --seeds 0,1,2` on the frozen subset; `--arms b` and `--arms calm+repair` only if §8 is on schedule. Nested subsampling gives N ∈ {1,2,4,8}.

### 6.4 Confluence on real logs
For every run directory: replay each task's event log in 20 shuffled orders; assert identical derived facts. Emit `confluence.json` with counts. Also run the git-baseline order-dependence check (§7) and put the two results side by side.

### 6.5 Statistics you can actually support at n=50
Each solve rate has roughly ±14-point 95% CI. Paired tests on the same tasks are much sharper than that suggests, but a 5-point difference is not detectable; a 15–20 point difference is. Report CIs and the paired bootstrap, not just point estimates. Three seeds tighten things but tasks, not seeds, are the unit of inference.

---

## 7. Demo (3 minutes, N=4, one hand-made task)

**Task** (`demo/task.py`): a 4-method class (e.g. `KVStore`: `set`, `get`, `delete`, `keys_with_prefix`) where two methods naturally share a private helper (`_normalize_key`) and the constructor is fixed. All four agents are asked for **all four slots**, so every slot receives 4 concurrent fills — the point is *same-slot* concurrency, not partitioned work; partitioned work would let a skeptic say "just use one file per method."

**Left terminal — git worktree per agent.** Four worktrees off `main`; each agent writes its full class into `kv.py`; merge sequentially. Same-method edits conflict on the same lines; resolve with `-X theirs` (last writer wins). Then **run it twice with the merge order reversed** and `diff` the two resulting `kv.py`: non-empty. Run the tests on both: at least one order fails or silently ships the other agent's `_normalize_key`. Caption: *output depends on merge order.*

**Right terminal — the store.** `viz/live.py`: definitions stream in, `conflicts: 0`, duplicates collapse, first verified composition turns the root green. Then `--replay-shuffled 20`: *derived facts identical in all 20 orders.* Caption: *nothing to order.*

Then one slide: solve-rate-vs-N and predicted-vs-observed, with the one-liner.

Rehearsal: `temperature=0.7`, fixed seed per agent (T=0 would make all 4 fills identical — dedup 100% is a degenerate demo). Rehearse ≥10×; record an asciinema/video fallback; the demo script must work offline from the recording if the GPU box dies.

---

## 8. Build order and time boxes (24h; cut from the bottom)

| Hours | Deliverable | Done when |
|------:|-------------|-----------|
| 0–2 | `store/`: defs, normalize (exact-unparse first), store, derive, materialize + tests | Reference methods materialize and pass 1 task's tests |
| 2–4 | `serve/`: client, prompts, extract; smoke test against the endpoint | `n=4` slot fills extracted into 4 defs |
| 4–6 | `runner/`: sandbox, `_run_tests.py`, tests.py; `bench/classeval.py` loader + subset selection | `data/subset.json` frozen; refs pass 3/3 |
| 6–9 | `agents/`: Phase 0–2; `bench/experiment.py` writing JSONL | One task solved end-to-end with N=4 |
| 9–11 | `bench/baselines.py` (C, A-from-C); token accounting; nested subsampling | `--arms calm,c` runs on 5 tasks |
| 11–13 | Pilot → model choice → **launch main run in background** (on the rented GPU if the Runpod track provides one, §11) | Run is progressing; ETA computed |
| 11–14 | `viz/live.py` (while the run goes) | Grid + counters + green root on a live task |
| 14–16 | `bench/metrics.py`, `charts.py`, confluence on real logs; cost metric + frontier chart (§3.15 item 9, chart vi) | 6 charts + `confluence.json` |
| 16–19 | `demo/`: task, git baseline, order-flip diff, rehearsals ×10, recording; `calm solve` wrapper (≤1h, §3.18) | Fallback recording exists |
| 19–22 | README/writeup: one-liner, charts, pre-registration table with outcomes, limitations; §11 sponsor paragraphs (≤30 min, reuse the same charts) | Judges could read it cold |
| 22–23 | Devpost: select sponsor tracks per §11 (only ones whose requirement was actually met) | Submitted |
| 22–24 | Buffer, sleep, polish | — |

**Cut list, in order:** optional sponsor items (OpenAI secondary arm → Devin-delegated module) → `calm solve` on arbitrary files (keep the demo path) → D3 viz → baseline B → repair arm → alpha-renaming (keep exact-unparse dedup) → seeds 1,2 → N sweep (keep N=8 + C@tokens) → subset 50→30 → held-out metric.

**Never cut:** I1–I3 tests, C@tokens baseline, confluence test, the demo's order-flip diff. No sponsor item ever displaces anything on this line or delays the main run.

---

## 9. Decision log / open questions (append, don't rewrite)

- 2026-09-19 — Slot refs late-bound, helper refs hash-frozen (I2). Rationale: pure Unison-style freezing prevents combinatorial composition.
- 2026-09-19 — Per-slot facts are composition facts; introduce stub context + `inconclusive`. Rationale: ClassEval method tests call other methods; the official pipeline has no ground-truth substitution.
- 2026-09-19 — Tests hidden from agents; oracle verification for both arms; headline = coverage; held-out metric secondary.
- 2026-09-19 — Fair baseline = whole-class pass@N at matched completion tokens (granularity axis). Feedback/repair is a separate axis, off by default.
- 2026-09-19 — Benchmark = original ClassEval (dependency depth ~1.8), not ClassEval-Pro (~3): shallower coupling gives the store a fair shot; note Pro as the stress test in limitations.
- 2026-09-19 — Sponsor tracks are packaging (§0, §11). Primary: Token Company, Warp. Conditional: Runpod (challenge still "TBA"). Optional: OpenAI, Cognition, Ramp. Skipped: Voloridge (ClassEval results are not dataset insight; would be a second project), Elastic, and the hardware/voice/domain tracks.
- 2026-09-19 — Slot prompt reordered so the slot name comes last (§3.7). Same content as ClassEval compositional; shared prefix across slots makes prompt caching possible. Cost-only change.
- 2026-09-19 — Added H6 (cost) and metric §3.15.9, derived from existing logs. Dollars reported only as what-if prices.
- 2026-09-19 — Materialize from `canonical_src`, not `raw_src` (§3.5 deviation). `Definition` identity is its hash; raw_src/meta are first-seen provenance only. Materializing from canonical makes the module a pure function of the hash set, so replay order can't change what gets tested.
- 2026-09-19 — Fill parameters are NOT alpha-renamed (§3.1 deviation): since we materialize from canonical form and ClassEval tests call methods by keyword, slot param names are interface. Helper params ARE renamed (private; identity = body). A keyword call into a renamed helper errors in tests, which is recorded as a fact.
- 2026-09-19 — `Outcome` carries the composition's `bindings` (not part of its identity, which is (test_id, comp_id, result)). This keeps the event log self-contained for replay and for `slot_evidence`. Still exactly two write ops.
- 2026-09-19 — `CalmStubHit` subclasses `BaseException` so generated `except Exception` can't swallow a stub hit and turn `inconclusive` into pass/fail.
- 2026-09-19 — Cyclic/self-recursive helpers: members are canonicalized with SCC-internal refs set to a placeholder, SCC hash = sha256 of sorted pre-forms, member hash = sha256(scc_hash|rank). Rank ties broken by name (the only name-dependent step).
- 2026-09-19 — `Task` lives in `calm_coder/task.py` (shared by bench/classeval.py and cli.py). Its `fields` (self.x assigned in `__init__`) are excluded from `unresolved_refs`.
- 2026-09-19 — Dev machine is an Apple M4 with 16 GB, no NVIDIA GPU: vLLM is not available locally. Local serving = Ollama or mlx_lm.server (`CALM_NO_N=1`); main run on Runpod if offered. RLIMIT_AS is a no-op on macOS (the sandbox ignores that error; wall + CPU limits still apply).
- 2026-09-19 — Ollama 0.21 ignores `n` on /v1/chat/completions and returns no `cached_tokens`: run with `CALM_NO_N=1`; sample i gets seed+i; cached_tokens stays null. With server-side n>1, usage is per request, so prompt tokens are charged to sample 0 and completion tokens are split across choices by text length (flagged `tokens_estimated`).
- 2026-09-19 — `extract_class` on bare methods (no class in output) splices them into the skeleton's class so the constructor is kept; `extract_functions` strips `<think>` blocks (Qwen3).
- 2026-09-19 — Subset frozen: 78/100 eligible, 50 chosen (seed 0), exclusions in `data/subset_excluded.jsonl`. Two generic normalizations applied before freezing (no run had used the subset): module-level string-literal statements in skeletons are inert, and test class names are whitespace-stripped (ClassEval_97 lists `' Words2NumbersTestMain'`). ClassEval_25/49 excluded because their tests call methods absent from the skeleton; 17 (date), 48 (network), 58 (random) are genuinely flaky; ClassEval_5's whole skeleton is inside a string literal.
- 2026-09-19 — `methods_info[i].test_class` sometimes names the suffix-less class (ClassEval_0 assigns `AccessGatewayFilterTest` to `set_current_user_info_and_log`), so some tasks have no class-level-only tests. The held-out metric (§3.15.5) must report how many tasks have ≥1 class-level class.
- 2026-09-19 — Ollama KV-cache sizing trap: `OLLAMA_CONTEXT_LENGTH` is the TOTAL across `OLLAMA_NUM_PARALLEL` sequences. The defaults gave 512 tokens/sequence while the slot prompt is ~615 tokens (silent truncation risk). The server must be started with `OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0` (4096/seq, 1.9 GB KV). Measured on the M4: qwen2.5-coder:7b ≈ 20 tok/s single stream, ≈ 25 tok/s aggregate at 8 in flight.
- 2026-09-19 — Phase 0 sends one request per sample (seed = run_seed·10⁴ + slot_order·100 + idx) instead of one `n=N` call per slot. On Ollama (no server-side n) it's the same traffic; it gives exact per-sample tokens and latency and lets each sample be ingested the moment it lands. Requests are issued sample-index-major so nested prefixes (N=1,2,4) finish first.
- 2026-09-19 — TTFV is reported as generation-done time of the samples used + stub-test time + search test time (CALM), and first-passing-sample arrival + its test time (C). Nested N<8 budgets are evaluated on the N=8 generation, so their generation time is the finish time of the samples they use. Stated on the chart.
- 2026-09-19 — Blame for a failing test = slots the test calls ∪ slots in the traceback, closed under the bound fills' own slot_refs (through helpers). Without the closure, a bad callee (e.g. a logging slot called by `filter`) was never swapped, found on a fake-model test.
- 2026-09-19 — Cuts applied (§8 order): baseline B, repair arm, D3 viz, OpenAI/Cognition/Runpod sponsor items, seeds 1–2, subset 50→30 (throughput: ~10 min/task for all three arms on the M4). Nested N ∈ {1,2,4,8} kept (free). `calm solve` kept (≈1h, Warp track).
- 2026-09-19 — Demo reality check: with qwen2.5-coder:7b at T=0.7 and T=1.0, the four whole-class agents write correct but differently worded code, so the two merge orders produce different kv.py files that both pass the tests. The demo claims only what it shows ("output depends on merge order"); no failing variant was cherry-picked or hand-written.
- 2026-09-19 — CLI tasks have no methods_info, so a test class is a slot's own test iff that slot is the only one it calls (structural, not by name).
- 2026-09-19 — Throughput: the experiment runs K tasks concurrently in one process (`--workers 3`, single writer, append-only). Greedy and CALM test phases left the 4 server slots idle. Cost: TTFV now includes contention from other tasks' requests, for both arms alike. Tasks 1–4 of the pilot ran with 1 worker. `OLLAMA_NUM_PARALLEL=8` with 32k context is NOT usable on the 16 GB M4: the 7.6 GB KV cache pushed all 29 layers off the GPU (CPU-only). Keep 4 × 4096.
- OPEN — Hardware and model (decide at the pilot, §6.2). Record here.
- OPEN — `MAX_COMPS` (64) and `per_class_timeout_s` (5) — revisit after the pilot.
- OPEN — Whether to include `field_refs` in the mismatch analysis.
- OPEN — Runpod challenge text ("to be announced" as of 2026-09-19). Whether a team may enter multiple sponsor tracks (not stated in the challenge doc; ask at the help desk). OpenAI API credits go only to registered challenge participants: register early if that arm stays in.

---

## 10. References

- Hellerstein & Alvaro, *Keeping CALM: When Distributed Consistency Is Easy*, CACM 2020. Ameloot, Neven & Van den Bussche, *Relational transducers for declarative networking*, JACM 2013 (CALM proof).
- Conway et al., *Logic and Lattices for Distributed Programming*, SoCC 2012.
- Zelikman et al., *Parsel: Algorithmic Reasoning with Language Models by Composing Decompositions*, NeurIPS 2023.
- Brown et al., *Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*, 2024.
- Li et al., *Competition-level code generation with AlphaCode*, Science 2022.
- Du et al., *ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation*, 2023. Dataset: `FudanSELab/ClassEval` (HF), CC BY-NC 4.0.
- Chen et al., *ClassEval-Pro: A Cross-Domain Benchmark for Class-Level Code Generation*, arXiv 2604.26923, 2026.
- *Scaling Test-Driven Code Generation from Functions to Classes* (ClassEval-TDD), arXiv 2602.03557, 2026.
- Unison Computing, *The Unison language: content-addressed code* (docs).

---

## 11. Sponsor tracks (packaging only, see §0)

Rule of thumb: every sponsor pitch uses the **same run, the same charts, and the same pre-registered outcomes** as the main writeup, including the ones that failed. If a pitch needs a claim the data doesn't support, the pitch changes, not the data. Challenge text is from the HackMIT 2026 challenge doc (2026-09-19). Odds are rough guesses meant for comparing tracks, not calibrated forecasts.

| Track | What they judge | What we show (already in the spec) | Extra work | Rough odds | Priority |
|---|---|---|---|---|---|
| **The Token Company**: LLM cost saving ($500 + interview) | Savings in the product, not dev cost | H6, tokens-to-first-solve under doubling schedule, cost frontier (chart vi); levers = slot-granular short completions, shared-prefix caching (§3.7), smallest model that works (§3.6), stop-when-verified (∃ exit) | Metric + chart from existing logs (~1h) | 25–35% if H6 holds, ~10% if not | Primary |
| **Warp**: best developer tool (keyboards) | Dev-experience improvement (Warp optional) | `calm solve` (§3.18): skeleton + tests in, verified class out, 0 conflicts by construction, live grid; demo recorded in Warp | ≤1h wrapper | 20–30% | Primary |
| **Runpod** (TBA) | Unknown | Main run on a rented GPU with vLLM + prefix caching; pilot throughput numbers (§6.2) | Env var change, if compute is offered | ? | Conditional: re-read when announced |
| **Ramp**: save time, save money | Anything that saves time and money | TTFV (H3) + cost (H6) | Select at submission | 5–10% | Free |
| **OpenAI**: API + Codex | API creativity; meaningful Codex use | *Secondary* arm on 10–20 subset tasks with a small hosted model via the same client; real `cached_tokens` from `usage`. Reported separately, never mixed into the headline | ~1–2h after main charts exist; Codex only if actually used for a bounded piece (e.g. `extract.py` fixtures) | 10–15% | Optional, first cut |
| **Cognition**: best use of Devin ($5k) | Creativity, novelty, polish | Devin builds one isolated module that already has acceptance tests (`serve/extract.py` + fixtures, `demo/git_baseline.py`, or `bench/charts.py`); merged only if §0 rules and its acceptance tests pass | Setup + review | 5–10% (ordinary dev use scores low on novelty) | Optional, first cut |

**Pitch lines** (one each, for the judge who walks up):
- *Token Company:* "At the same model and the same verifier, here's what a verified class costs with whole-class sampling vs. per-method sampling on a coordination-free store, including the prompt overhead we pay for going per-method." Then point at the frontier chart. If H6 failed, say where the crossover is. That's still a useful cost result.
- *Warp:* "Point it at a class skeleton and a test file; N agents write every method concurrently, nothing ever conflicts, and it hands you back a class that passes."
- *Runpod:* "The whole benchmark ran on one rented GPU; vLLM prefix caching is load-bearing because every slot prompt shares a prefix."

**Deliberately not doing:**
- **Voloridge.** It's judged on insight from their datasets, and a ClassEval harness produces none. Doing it properly means a second project.
- **Switching the headline model to a hosted frontier model** for any sponsor. It would push H1/H2 toward ceiling and contamination (§3.6).
- **Tuning the scheduler, subset, or N for a cost number.** H6 is pre-registered like everything else.
- **Claiming dedup saves tokens.** It saves test executions (§3.15 item 6).
