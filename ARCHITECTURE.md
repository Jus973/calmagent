# Architecture

Two layers, and the line between them is the whole claim.

```
producers ──write──▶  grow-only store  ──pure reads──▶  derivations ──▶  exit
(LLM, non-monotone)   (defs, outcomes)   (monotone)        ∃ verified composition
                             ▲
                             └── policy reads (scheduler, best_class, dead_slots): decide what to
                                 generate or test next; never decide what is derivable
```

## The store (`calm_coder/store/`)

Two write operations, `add_def` and `add_outcome`, both idempotent set inserts, both append to an
event log. Nothing is ever deleted, replaced, or picked-as-best inside the store (I1). Definitions
are keyed by the hash of their canonical form; slot references are late-bound so any fill composes
with any fill of its dependencies, while private helper references are frozen to the helper's hash
(I2). `store/derive.py` is pure and monotone: `reachable`, `complete`, `slot_evidence`, `verified`,
`done` (I3). Success is `∃ comp. ∀ test. pass` — coordination-free. The only barrier is the failure
exit, "nobody found anything within budget".

## Producers (`calm_coder/v2/producers.py`)

A producer is anything that writes candidates into the store:

| producer | prompt | writes |
| --- | --- | --- |
| `WholeClassProducer` | skeleton | the whole class, decomposed into one candidate per slot |
| `PerMethodProducer` | skeleton + "implement `m`" | one candidate for slot `m` |
| `RepairProducer` | skeleton + current best class + that slot's failure summary | one candidate for a dead slot |
| `WholeClassRepairProducer` | skeleton + current best class + failure summary | the whole class again |

They differ only in the prompt and in what they are told about the current state. Once written,
their outputs are indistinguishable: a candidate is a candidate, and recombination ranges over all
of them. That is what "producer-agnostic store" means, and it is why a whole-class sample and a
repair sample can be combined into a class neither producer ever emitted.

Every producer spends from a per-task `Budget` measured in decode tokens, so arms are comparable at
equal tokens rather than at equal request counts.

### Decomposition and no-loss (`calm_coder/v2/decompose.py`)

A whole-class sample is parsed, split per declared slot (non-slot functions ride along as helpers),
and inserted as ordinary definitions. The sample's own slot bindings are returned, and the harness
evaluates that exact composition before anything else. So pooling can only add: every class the
whole-class baseline would have tested is still tested, byte for byte. `tests/test_v2.py` pins this.

## Policy reads (`calm_coder/v2/state.py`)

`best_class` is an argmax and `dead_slots` is a negation, so neither may live in `derive.py`. They
live here, read only immutable sets, and are used exclusively to decide what to generate next:

* `verdict_facts` — (task, context, slot, candidate, test class, result, failure summary) per outcome.
* `best_class` — the fully bound composition with the most passing test classes; ties broken by
  fewest failures then lowest composition id, so it is deterministic given the store contents.
* `dead_slots` — slots no candidate has ever passed the method-level test class for, plus (for slots
  with no own test class) the slots blamed by the current class's failures.

## Feedback levels (`calm_coder/v2/feedback.py`)

Tests are the verifier, not the prompt. The single controlled exception is repair feedback:

| level | contents |
| --- | --- |
| F0 | which test classes fail, nothing else |
| F1 | exception type and message for up to 3 failures, 300 characters each |
| F2 | F1 plus the single source line that raised |

Assertion messages quote the test expression that produced them, so `parse_failure` scrubs any run
of ≥24 characters that also occurs in the test source before a `Failure` is built. Only
`raising_line`, which only F2 renders, is exempt. F2 results are always reported separately.

## Prompt prefixes (`calm_coder/serve/prompts.py`)

Every prompt for a task is `STATIC_INSTRUCTIONS + SKELETON_BLOCK + VARIABLE_SUFFIX`; nothing
variable precedes the skeleton, and within a repair round every dead slot additionally shares the
current-class block. One warm-up request per task primes the prefix before the fan-out, and its
tokens are charged to the budget like any other.

## Arms (`calm_coder/v2/harness.py`)

* **V2** — k whole-class samples → store → test each sample's own composition → recombination
  search → up to 3 rounds of targeted repair of dead slots (one request per dead slot, n samples
  each) → stop on the first verified composition or when the budget is gone.
* **V2+PM** — the same, with per-method samples pooled in as well.
* **WCR** — the fair repair baseline: same k samples, same feedback, same budget, but each round
  regenerates the whole class and the verdict comes from the samples themselves. No recombination,
  so the arms differ only in where new tokens are spent.

Budgets come from a real arm-C run (`--budget-from`), so "equal tokens" is measured, not estimated.

## Determinism

Sampling seeds are `hash(task_id, arm, slot, round, sample_idx)`. Everything after generation is a
function of the store's contents, and the event log replays into an identical store under any
permutation — including `best_class` and `dead_slots`, which the replay test checks explicitly.
