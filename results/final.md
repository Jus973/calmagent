# CALM Coder v2 — final report

50 tasks in the union, seeds [0], 0 tasks excluded by the run.

## Solve rate (tasks are the unit of inference)

| arm | N | tasks | seeds/task | solve rate | 95% CI | interval | missing tasks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2_pm | 4 | 35 | 1 | 0.34 | [0.21, 0.51] | wilson | ClassEval_50, ClassEval_54, ClassEval_62, ClassEval_65, ClassEval_77, ClassEval_79, ClassEval_81, ClassEval_82, ClassEval_85, ClassEval_86, ClassEval_89, ClassEval_90, ClassEval_92, ClassEval_95, ClassEval_97 |
| v2 | 4 | 39 | 1 | 0.33 | [0.21, 0.49] | wilson | ClassEval_77, ClassEval_79, ClassEval_81, ClassEval_82, ClassEval_85, ClassEval_86, ClassEval_89, ClassEval_90, ClassEval_92, ClassEval_95, ClassEval_97 |
| c | 8 | 50 | 1 | 0.30 | [0.19, 0.44] | wilson | - |
| wcr | 4 | 34 | 1 | 0.29 | [0.17, 0.46] | wilson | ClassEval_50, ClassEval_54, ClassEval_55, ClassEval_62, ClassEval_65, ClassEval_77, ClassEval_79, ClassEval_81, ClassEval_82, ClassEval_85, ClassEval_86, ClassEval_89, ClassEval_90, ClassEval_92, ClassEval_95, ClassEval_97 |
| a_greedy | 1 | 50 | 1 | 0.12 | [0.06, 0.24] | wilson | - |

## Paired comparisons (unit: task)

| comparison | tasks | diff | bootstrap 95% CI | McNemar tasks | arm only | baseline only | McNemar p | ambiguous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_greedy-c | 50 | -0.180 | [-0.300, -0.080] | 50 | 0 | 9 | 0.004 | 0 |
| a_greedy-wcr | 34 | -0.147 | [-0.265, -0.029] | 34 | 0 | 5 | 0.062 | 0 |
| c-wcr | 34 | +0.059 | [+0.000, +0.147] | 34 | 2 | 0 | 0.500 | 0 |
| v2-c | 39 | +0.026 | [-0.077, +0.128] | 39 | 3 | 2 | 1.000 | 0 |
| v2-wcr | 34 | +0.059 | [+0.000, +0.147] | 34 | 2 | 0 | 0.500 | 0 |
| v2_pm-c | 35 | +0.000 | [-0.114, +0.114] | 35 | 2 | 2 | 1.000 | 0 |
| v2_pm-wcr | 34 | +0.059 | [-0.059, +0.176] | 34 | 3 | 1 | 0.625 | 0 |
| wcr-c | 34 | -0.059 | [-0.147, +0.000] | 34 | 0 | 2 | 0.500 | 0 |

## Systems

| arm | tasks | decode tok | prompt tok | cached prompt tok | requests | median TTFV ms | over budget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a_greedy | 50 | 616 | 583 | - | 1.0 | 30288 | 0 |
| c | 50 | 4794 | 4665 | 2522 | 8.0 | 46281 | 0 |
| v2 | 39 | 3745 | 20859 | 16415 | 8.6 | 203295 | 0 |
| v2_pm | 35 | 4256 | 23051 | 20778 | 9.3 | 143168 | 0 |
| wcr | 34 | 3981 | 9532 | 8386 | 4.1 | 50538 | 0 |

## Repair

| arm | feedback | repair rounds | solved by samples alone | solved | dead slots after sampling | dead slots at end |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | F1 | 2.05 | 8 | 13 | 1.74 | 1.31 |
| v2_pm | F1 | 1.00 | 7 | 12 | 1.20 | 1.06 |
| wcr | F1 | 0.74 | 9 | 10 | 1.38 | 1.32 |
