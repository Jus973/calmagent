import unittest


class KVStoreTestSet(unittest.TestCase):
    def test_set_normalizes(self):
        kv = KVStore()
        kv.set("  Foo ", 1)
        self.assertEqual(kv.data, {"foo": 1})

    def test_set_replaces(self):
        kv = KVStore()
        kv.set("a", 1)
        kv.set(" A", 2)
        self.assertEqual(kv.data, {"a": 2})


class KVStoreTestGet(unittest.TestCase):
    def test_get_normalizes(self):
        kv = KVStore()
        kv.data["foo"] = 1
        self.assertEqual(kv.get(" FOO "), 1)

    def test_get_default(self):
        kv = KVStore()
        self.assertIsNone(kv.get("x"))
        self.assertEqual(kv.get("x", default=7), 7)


class KVStoreTestDelete(unittest.TestCase):
    def test_delete(self):
        kv = KVStore()
        kv.data["a"] = 1
        self.assertTrue(kv.delete(" A "))
        self.assertFalse(kv.delete("a"))
        self.assertEqual(kv.data, {})


class KVStoreTestKeysWithPrefix(unittest.TestCase):
    def test_prefix(self):
        kv = KVStore()
        kv.data.update({"apple": 1, "apricot": 2, "banana": 3})
        self.assertEqual(kv.keys_with_prefix(" AP"), ["apple", "apricot"])
        self.assertEqual(kv.keys_with_prefix("z"), [])


class KVStoreTest(unittest.TestCase):
    def test_roundtrip(self):
        kv = KVStore()
        kv.set(" Apple", 1)
        kv.set("APRICOT ", 2)
        self.assertEqual(kv.get("apple"), 1)
        self.assertEqual(kv.keys_with_prefix("ap"), ["apple", "apricot"])
        self.assertTrue(kv.delete("apple"))
        self.assertEqual(kv.keys_with_prefix("ap"), ["apricot"])
