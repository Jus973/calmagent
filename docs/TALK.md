# CALM Proxy — 3-minute talk script

Slide-by-slide. Bold = the line to actually say; everything else is stage direction or backup for
questions. Numbers match `docs/RESULTS.md`; regenerate both if a run directory changes.

Total: **2:55** spoken. Leave the last 5 seconds silent on the results slide.

---

## Slide 1 — The hook (0:00–0:25)

*On screen: the two bars, 25,711 ms vs 253 ms.*

> **Your local model will re-read your agent's entire transcript for twenty-six seconds, or reuse
> it in a quarter of a second. Same prompt. And nothing in the OpenAI API tells you which one just
> happened.**
>
> **We measured that gap on a MacBook Air: a hundred times. Then we built the thing that tells you
> which side of it you're on.**

*Do not explain what a KV cache is. The bars do it.*

---

## Slide 2 — What it is (0:25–0:45)

*On screen: agent → [ CALM Proxy ] → Ollama.*

> **It's a proxy. It goes between any coding agent and any local model server — Ollama, vLLM,
> llama.cpp. You change nothing about your agent.**
>
> **It records every request, shows you where the prompt cache is being missed and why, and
> shortens the prompts that are safe to shorten.**

---

## Slide 3 — The part I'd want you to steal (0:45–1:25)

*On screen: the four-lever table, three rows struck through.*

> **We had four ideas. Before building any of them, we measured how much each was worth on a real
> agent's traffic. Three came back at zero and we deleted them.**
>
> **The prompt-cache linter — zero. Because this agent is already well behaved: it only ever
> appends to its transcript, and the server is already capturing ninety-five percent of the
> available cache saving. There was nothing to win.**
>
> **That's the measurement I'm proudest of, because it cancelled two thirds of our plan.**

*If asked which three: prefix-lint (0 s), memo (0 duplicate requests of 42), and calm-run —
which works, but a ClassEval test suite runs in 20 ms, so caching it saves 40 ms. We say that
rather than quoting the 64–75% from the old project as if it were wall clock.*

---

## Slide 4 — So where does the time actually go (1:25–1:55)

*On screen: the two-row table — prompt token 2.18×, output token 2.48× — and r² = 0.975.*

> **If the cache is fine, why does the agent keep getting slower? Because context isn't free.**
>
> **We fitted the measured request time against both kinds of token, each with its own attention
> term. A prompt token costs twice as much at twenty-seven thousand tokens of context as it does
> at one thousand. An output token costs two and a half times as much — generation drops from
> twenty-two tokens a second to under nine, with nothing changed but the length of the
> transcript.**
>
> **No prefix cache fixes that. Those tokens are cached. They're still being attended to.**

*The credibility line, if you have a second: the fit recovers 6.95 ms per prompt token against a
separately measured 5.96, and 23 tokens/sec against the model's known 20. It was given neither.*

---

## Slide 5 — The result (1:55–2:35)

*On screen: the 10-row distribution table, pooled figure highlighted.*

> **So we shortened the context — content-addressed dedup, replacing a repeated message with a
> stable reference to its first copy.**
>
> **Across all ten conversations our benchmark recorded — replayed with the agent taken out of the
> loop, so the lever is the only variable — thirty-seven percent fewer prompt tokens and two point
> one times less server time. Solve rate unchanged: four out of ten in both arms.**
>
> *(beat, point at the bottom four rows)*
>
> **Four of the ten show no effect at all. Those are the short runs — nothing repeated yet,
> nothing to remove. One is actually one percent slower, and that's in the table too.**
>
> **It does nothing when you don't need it, and returns two to three-point-seven times on the long
> looping runs that were burning your time. We're showing you the distribution rather than the
> best number, because the best number on its own is a sales pitch.**

---

## Slide 6 — Close (2:35–2:55)

*On screen: `python -m bench_agent.demo`*

> **Every number you've seen comes out of a run directory in the repo, printed by one command, with
> no model server running. Nothing to fail on stage.**
>
> **And the caveats are attached to the numbers, not in a footnote — including the one where our
> own test harness caused the effect we were measuring.**

---

# Questions you will get

**"Why is the A/B's wall clock not your headline?"**
> Because it's confounded and we'd be lying. At temperature 0 the agent is deterministic given its
> prompt, so the first message we rewrite forks the trajectory and the two arms stop being the same
> agent. Our very first pair showed a beautiful 3.8× — and it turned out the agent had just quit
> early after misreading an empty shell output as "tests passed". That's why the replay bench
> exists: it removes the agent entirely. The A/B's job is only to prove we don't cost solves.

**"Ten tasks is nothing."**
> Agreed, for solve rate — and we say so: a one- or two-task gap is not evidence either way. But
> the headline isn't a solve-rate claim. It's 149 requests replayed in a paired design where the
> only variable is the lever, cross-checked in both arm orders and in two separate invocations an
> hour apart, which agreed to within 4%.

**"What's the 13.7% cross-run number?"**
> Two runs of the same task an hour apart share only 13.7% of their prompt, and every conversation
> diverges at message 3 — on one line of `ls -la` output. **But our own harness caused it:** it's
> the parent temp directory's mtime, and we create that directory fresh per task. The mechanism is
> real and general — one volatile line forty tokens in costs everything behind it. The trigger is
> ours. That caveat is written into the script and the JSON so it can't get separated from the
> number.

**"Isn't this just prompt caching / context compaction?"**
> Compaction usually means dropping or summarising content, which loses information and breaks the
> prefix. This is content-addressed: a repeated message becomes a stable reference to its first
> copy, keyed by hash, so the rewrite is identical on every turn and the prefix stays aligned.
> That stability is the whole lever — a rewrite that moved would invalidate everything behind it,
> and at 6 ms per prompt token one broken 17k prefix costs about 100 seconds, more than the lever
> saves. It's enforced by tests, and the tests caught a real bug before it ever ran.

**"What would you build next?"**
> The cross-run linter: normalise volatile tool output so a second run can reuse the first's
> prefix. Our own measurement says the ceiling is set forty tokens in, and nobody finds that line
> without a proxy.

**"What happened to the original project?"**
> It's still in the repo, pre-registration and all. The hypothesis — that a coordination-free,
> content-addressed store raises class-level solve rate — came back null: +0.026, McNemar p = 1.000.
> We kept the failed hypotheses visible and built this out of what that work measured.

---

# If you only get 60 seconds

> **Your local model re-reads your agent's whole transcript — twenty-six seconds, versus a quarter
> second if the cache hits. Nothing in the API tells you which happened, so we built a proxy that
> does.**
>
> **It told us three of our four planned optimisations were worth exactly zero, so we deleted them.
> The fourth — content-addressed dedup — gives thirty-seven percent fewer prompt tokens and 2.1×
> less server time across every conversation we recorded, with solve rate unchanged. On short runs
> it does nothing, and we show that row too.**
