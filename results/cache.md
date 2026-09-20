# What content-addressing saves the verifier

86 configs over 22 tasks, from runs/20260919T163449Z_main.

A test class's result is keyed by the test module and the fills that class can reach, both content hashes, so two pairs with equal keys are one unit of work between them and the saving is exact rather than sampled.

**Work avoided** — (composition, test class) pairs that need no fresh result:

| cache scope | pairs | need executing | avoided |
| --- | --- | --- | --- |
| one solve | 5940 | 2115 | 3825 (64.4%) |
| one task, all N | 5940 | 1457 | 4483 (75.5%) |
| the whole run | 5940 | 1457 | 4483 (75.5%) |

**Wall clock** — a composition's subprocess is narrowed to the classes that are not already known, so the saving is bracketed: the floor counts only compositions that are skipped whole, the estimate assumes time is proportional to the classes still executed and so does not credit the process startup a narrowed run still pays.

| cache scope | compositions | skipped whole | test wall | floor | estimate |
| --- | --- | --- | --- | --- | --- |
| one solve | 1111 | 72 (6.5%) | 82s | 4.8% | 62.9% |
| one task, all N | 1111 | 319 (28.7%) | 82s | 21.3% | 72.4% |
| the whole run | 1111 | 319 (28.7%) | 82s | 21.3% | 72.4% |
