# Latency — TTFV on a deterministic server

What the harness claims about latency is that a monotone store lets generation, testing and search
overlap, and that a request may end as soon as the fact it was asked for exists. Both are scheduling
claims, so they are measured against a **fake server** (`bench/fake_model.py`): the completions are
canned (`bench/latency_samples.py`) and every mode draws from one queue at one aggregate throughput,
so a token costs the same server time in every mode and a difference in wall clock is the harness,
not the sampler. The tests are real subprocesses. These numbers are therefore *controlled*, not a
measurement of qwen2.5-coder on a laptop.

```bash
python -m calm_coder.bench.latency --repeat 3 --tail short   # prefill 400 ms, 24 tok/s shared, N=4
```

Each mode adds one lever to the one above it:

| mode | lever |
|---|---|
| sequential | the experiment's path: generate everything, stub-test everything, then search one composition at a time |
| pipelined | fills are stub-tested as they land; the search tests `--width` compositions at once and fails fast |
| adaptive | a slot stops sampling once one of its fills passes its own test in stub context |
| stop-at-fill | each request ends when its method (and the helpers it calls) is complete |

`tail` is what the model appends after the method it was asked for — `none` is a perfectly obedient
model, `short` is a fence plus a sentence, `long` is prose and the neighbouring methods. The prompt
asks for one method either way; the tail is the part nothing downstream reads.

## KVStore demo, N=4, seed 0, median of 3

**`--tail short`**

| mode | verified | TTFV (ms) | speedup | requests | tokens decoded | of offered | test procs |
|---|---|---:|---:|---:|---:|---:|---:|
| sequential | yes | 39610 | 1.00x | 16 | 794 | 794 | 17 |
| pipelined | yes | 39787 | 1.00x | 16 | 794 | 794 | 103 |
| adaptive | yes | 28901 | 1.37x | 13 | 577 | 643 | 56 |
| stop-at-fill | yes | 17965 | **2.20x** | 13 | 316 | 643 | 57 |

**`--tail none`** (nothing to cut but the fence)

| mode | verified | TTFV (ms) | speedup | requests | tokens decoded | of offered | test procs |
|---|---|---:|---:|---:|---:|---:|---:|
| sequential | yes | 25613 | 1.00x | 16 | 458 | 458 | 17 |
| pipelined | yes | 25893 | 0.99x | 16 | 458 | 458 | 105 |
| adaptive | yes | 18388 | 1.39x | 13 | 326 | 370 | 55 |
| stop-at-fill | yes | 17973 | 1.43x | 13 | 316 | 370 | 56 |

**`--tail long`** (the 7B failure mode: it writes the rest of the class)

| mode | verified | TTFV (ms) | speedup | requests | tokens decoded | of offered | test procs |
|---|---|---:|---:|---:|---:|---:|---:|
| sequential | yes | 127641 | 1.00x | 16 | 2907 | 2907 | 17 |
| pipelined | yes | 127948 | 1.00x | 16 | 2907 | 2907 | 102 |
| adaptive | yes | 94892 | 1.35x | 13 | 2162 | 2359 | 57 |
| stop-at-fill | yes | 17971 | **7.10x** | 13 | 316 | 2359 | 58 |

Every mode verifies, and the verified class is the same one: the levers change when facts appear and
how many are asked for, never which are derivable.

## Reading it honestly

- **Pipelining is free here, not fast.** On this server generation *is* the wall: every mode's tokens
  queue on one shared throughput, so starting tests earlier cannot make a token arrive sooner. What it
  buys is that the test time (5–6× more subprocesses, all overlapped) stops being additive — sequential
  pays 17 test processes in series after generation ends, pipelined pays 100+ during it and still
  finishes level. The win is real only when tests are the wall; set `--tok-s` high to see it.
- **Adaptive is the store's win.** 13 requests instead of 16 and 577 of 643 offered tokens: three slots
  stop after a fill passes its own test. Nothing is deleted or replaced to make that happen — a slot
  that stops just never adds more facts.
- **Stop-at-fill scales with how badly the model overshoots**, which is the point: 1.43x when the model
  is obedient, 7.1x when it writes the whole class. It is a decode-side lever, so it compounds with the
  others rather than competing.
- The absolute numbers are a function of `--ttft-ms` and `--tok-s`; only the ratios within a table mean
  anything, and only for this task shape (4 slots, one helper-using fill).
- Real-server TTFV on the pilot box is in `results/final.md` and is *not* comparable: it predates all
  four levers and measures a different arm.
