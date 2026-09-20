# CALM Coder v2 — final report

10 tasks in the union, seeds [0], 0 tasks excluded by the run.

## Solve rate (tasks are the unit of inference)

| arm | N | tasks | seeds/task | solve rate | 95% CI | interval | missing tasks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2 | 4 | 10 | 1 | 0.80 | [0.49, 0.94] | wilson | - |
| c | 8 | 10 | 1 | 0.60 | [0.31, 0.83] | wilson | - |
| wcr | 4 | 10 | 1 | 0.60 | [0.31, 0.83] | wilson | - |

## Paired comparisons (unit: task)

| comparison | tasks | diff | bootstrap 95% CI | McNemar tasks | arm only | baseline only | McNemar p | ambiguous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2-c | 10 | +0.200 | [+0.000, +0.500] | 10 | 2 | 0 | 0.500 | 0 |
| wcr-c | 10 | +0.000 | [+0.000, +0.000] | 10 | 0 | 0 | 1.000 | 0 |

## Systems

| arm | tasks | decode tok | prompt tok | cached prompt tok | requests | median TTFV ms | over budget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c | 10 | 3797 | 3969 | 3597 | 8.0 | 69486 | 0 |
| v2 | 10 | 2567 | 10599 | 9936 | 4.5 | 198898 | 0 |
| wcr | 10 | 2825 | 6298 | 6099 | 3.2 | 133984 | 0 |

## Repair

| arm | feedback | repair rounds | solved by samples alone | solved | dead slots after sampling | dead slots at end |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | F1 | 0.90 | 6 | 8 | 1.10 | 0.20 |
| wcr | F1 | 0.40 | 6 | 6 | 1.10 | 1.10 |
