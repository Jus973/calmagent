from calm_coder.runner import tests as rt


def test_slot_and_class_level_split(classeval_task0):
    t = classeval_task0
    assert rt.slot_test_class(t, "filter") == "AccessGatewayFilterTestFilter"
    # methods_info assigns the suffix-less class to a method: never infer by name (§3.10).
    assert rt.slot_test_class(t, "set_current_user_info_and_log") == "AccessGatewayFilterTest"
    assert rt.class_level_tests(t) == []


def test_deps_over_approximate(toy):
    assert rt.test_slot_deps(toy, "ToyTestGet") == {"put", "get"}
    assert rt.test_slot_deps(toy, "ToyTestNorm") == {"norm"}
    assert rt.test_slot_deps(toy, "Nope") == set(toy.slot_ids)
    assert rt.predicted_inconclusive(toy, "ToyTestGet", {"get"}) == {"put"}


def test_slots_in_traceback(toy):
    tb = 'File "mod.py", line 9, in get\n  ...\nCalmStubHit: \'put\''
    assert rt.slots_in_traceback(toy, tb) == {"get", "put"}


def test_test_class_source(toy):
    assert rt.test_class_source(toy, "ToyTestPut").startswith("class ToyTestPut")
