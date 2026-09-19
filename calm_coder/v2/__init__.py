"""v2: producer-agnostic pooling over the same grow-only store, plus targeted repair.

Nothing here adds a store write. Producers call `add_def`/`add_outcome` only; everything else
is a pure read. `state.py` holds the *policy* reads (best_class, dead_slots) that use argmax and
therefore must not live in `store/derive.py` (I3).
"""
