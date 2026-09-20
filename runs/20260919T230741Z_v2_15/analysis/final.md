# CALM Coder v2 — final report

50 tasks in the union, seeds [0], 0 tasks excluded by the run.

## Solve rate (tasks are the unit of inference)

| arm | N | tasks | seeds/task | solve rate | 95% CI | interval | missing tasks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2 | 4 | 14 | 1 | 0.43 | [0.21, 0.67] | wilson | ClassEval_27, ClassEval_32, ClassEval_33, ClassEval_35, ClassEval_37, ClassEval_41, ClassEval_42, ClassEval_47, ClassEval_50, ClassEval_54, ClassEval_55, ClassEval_62, ClassEval_64, ClassEval_65, ClassEval_67, ClassEval_68, ClassEval_70, ClassEval_71, ClassEval_74, ClassEval_76, ClassEval_77, ClassEval_79, ClassEval_80, ClassEval_81, ClassEval_82, ClassEval_83, ClassEval_85, ClassEval_86, ClassEval_88, ClassEval_89, ClassEval_90, ClassEval_91, ClassEval_92, ClassEval_95, ClassEval_96, ClassEval_97 |
| v2_pm | 4 | 13 | 1 | 0.38 | [0.18, 0.64] | wilson | ClassEval_26, ClassEval_27, ClassEval_32, ClassEval_33, ClassEval_35, ClassEval_37, ClassEval_41, ClassEval_42, ClassEval_47, ClassEval_50, ClassEval_54, ClassEval_55, ClassEval_62, ClassEval_64, ClassEval_65, ClassEval_67, ClassEval_68, ClassEval_70, ClassEval_71, ClassEval_74, ClassEval_76, ClassEval_77, ClassEval_79, ClassEval_80, ClassEval_81, ClassEval_82, ClassEval_83, ClassEval_85, ClassEval_86, ClassEval_88, ClassEval_89, ClassEval_90, ClassEval_91, ClassEval_92, ClassEval_95, ClassEval_96, ClassEval_97 |
| wcr | 4 | 13 | 1 | 0.38 | [0.18, 0.64] | wilson | ClassEval_26, ClassEval_27, ClassEval_32, ClassEval_33, ClassEval_35, ClassEval_37, ClassEval_41, ClassEval_42, ClassEval_47, ClassEval_50, ClassEval_54, ClassEval_55, ClassEval_62, ClassEval_64, ClassEval_65, ClassEval_67, ClassEval_68, ClassEval_70, ClassEval_71, ClassEval_74, ClassEval_76, ClassEval_77, ClassEval_79, ClassEval_80, ClassEval_81, ClassEval_82, ClassEval_83, ClassEval_85, ClassEval_86, ClassEval_88, ClassEval_89, ClassEval_90, ClassEval_91, ClassEval_92, ClassEval_95, ClassEval_96, ClassEval_97 |
| c | 8 | 50 | 1 | 0.30 | [0.19, 0.44] | wilson | - |
| a_greedy | 1 | 50 | 1 | 0.12 | [0.06, 0.24] | wilson | - |

## Paired comparisons (unit: task)

| comparison | tasks | diff | bootstrap 95% CI | McNemar tasks | arm only | baseline only | McNemar p | ambiguous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_greedy-c | 50 | -0.180 | [-0.300, -0.080] | 50 | 0 | 9 | 0.004 | 0 |
| v2-c | 14 | +0.071 | [+0.000, +0.214] | 14 | 1 | 0 | 1.000 | 0 |
| v2_pm-c | 13 | +0.000 | [-0.231, +0.231] | 13 | 1 | 1 | 1.000 | 0 |
| wcr-c | 13 | +0.000 | [+0.000, +0.000] | 13 | 0 | 0 | 1.000 | 0 |

## Systems

| arm | tasks | decode tok | prompt tok | cached prompt tok | requests | median TTFV ms | over budget |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a_greedy | 50 | 616 | 583 | - | 1.0 | 30288 | 0 |
| c | 50 | 4794 | 4665 | 2522 | 8.0 | 46281 | 0 |
| v2 | 14 | 3175 | 19144 | 14829 | 8.6 | 64918 | 0 |
| v2_pm | 13 | 4027 | 24972 | 22171 | 10.3 | 87290 | 0 |
| wcr | 13 | 3459 | 7915 | 6955 | 3.8 | 39039 | 0 |

## Repair

| arm | feedback | repair rounds | solved by samples alone | solved | dead slots after sampling | dead slots at end |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | F1 | 1.93 | 4 | 6 | 1.93 | 1.14 |
| v2_pm | F1 | 1.15 | 3 | 5 | 1.46 | 1.08 |
| wcr | F1 | 0.62 | 5 | 5 | 1.46 | 1.46 |
