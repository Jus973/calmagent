# Targeted repair vs whole-file rewrite

An existing KVStore has working `set` / `delete` / `keys_with_prefix` and a `get` that
forgets to normalize keys. Two policies see the same failing tests:

- **repair** — keep passing fills, sample only dead slots, recombine.
- **rewrite** — regenerate the whole class (the usual agent loop). The model “fixes”
  `get` and drops normalization from `set`.

The store never retracts a passing method. The rewrite arm does not recombine with the
seed, so the regression stands. Same fake model, same budget, same feedback.

| policy | solved | decode tokens | model calls | seed methods kept | kept |
| --- | --- | --- | --- | --- | --- |
| repair | 1 | 80 | 2 | 3/4 | delete, keys_with_prefix, set |
| rewrite | 0 | 160 | 4 | 0/4 | - |

## What this is evidence for

Not “CALM beats whole-class sampling on ClassEval.” That bet (H2) failed. This is the
narrower claim that *did* show up: when a class is already mostly right, spending new
tokens only on dead slots does not throw away working methods, and whole-file rewrite can.

## ClassEval (7B pilot, not this fixture)

On 10 ClassEval tasks at equal decode budget, v2 (targeted repair + recombination) solved
8/10 and whole-class repair solved 6/10. Both arms solved the same 6 by sampling; the gap
was repair closing 9 of 11 dead slots while WCR closed none (`results/pilot.md`).
Underpowered (McNemar p = 0.50). Directionally the same policy as this fixture.

