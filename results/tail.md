# How much of a real completion is tail

684 per-slot emissions from runs/20260919T163449Z_main. A completion's *tail* is everything after the point the harness could have cut it and still stored the same fill (`serve/stream.truncation_point`). Measured in characters, since a cut request never reports a usage split; the token column applies the character rate and is a projection.

These are upper bounds: 12 completions stopped at the end of their method with nothing after to prove it, so they have no cut point and are not in the denominator — the tightest completions are the ones excluded.

| | value |
| --- | --- |
| completions with a tail worth cutting (>1%) | 4.7% |
| tail share of all characters decoded | 1.3% |
| median tail share of one completion | 0.6% |
| p90 tail share of one completion | 0.9% |
| decode tokens in these emissions | 107347 |
| tokens the cut would not have decoded (projected) | 1396 |

On a local server tokens are the wall clock, so this is the ceiling on what stop-at-fill can buy for per-slot requests on this model — the controlled speedups in `results/latency.md` are what the scheduler does with it.
