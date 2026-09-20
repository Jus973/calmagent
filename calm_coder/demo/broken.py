class KVStore:
    """
    A tiny in-memory key-value store. Keys are normalized before every use: surrounding
    whitespace is stripped and the key is lowercased, so " Foo " and "foo" are the same key.
    """

    def __init__(self):
        self.data = {}

    def set(self, key, value):
        """
        Store value under the normalized key, replacing any previous value.
        :param key: str
        :param value: any
        :return: None
        >>> kv = KVStore()
        >>> kv.set(" Foo ", 1)
        >>> kv.data
        {'foo': 1}
        """
        self.data[key.strip().lower()] = value

    def get(self, key, default=None):
        """
        Return the value stored under the normalized key, or default if it is absent.
        :param key: str
        :param default: any
        :return: any
        >>> kv = KVStore()
        >>> kv.set("foo", 1)
        >>> kv.get("  FOO")
        1
        """
        # the bug: looks up the raw key, so " FOO " misses the stored "foo"
        return self.data.get(key, default)

    def delete(self, key):
        """
        Remove the normalized key. Return True if it was present, False otherwise.
        :param key: str
        :return: bool
        >>> kv = KVStore()
        >>> kv.set("a", 1)
        >>> kv.delete(" A "), kv.delete("a")
        (True, False)
        """
        k = key.strip().lower()
        if k in self.data:
            del self.data[k]
            return True
        return False

    def keys_with_prefix(self, prefix):
        """
        Return the sorted list of stored keys that start with the normalized prefix.
        :param prefix: str
        :return: list of str
        >>> kv = KVStore()
        >>> kv.set("apple", 1); kv.set("Apricot", 2); kv.set("banana", 3)
        >>> kv.keys_with_prefix(" AP")
        ['apple', 'apricot']
        """
        p = prefix.strip().lower()
        return sorted(k for k in self.data if k.startswith(p))
