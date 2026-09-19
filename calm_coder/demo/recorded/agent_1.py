

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
        normalized_key = key.strip().lower()
        self.data[normalized_key] = value

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
        normalized_key = key.strip().lower()
        return self.data.get(normalized_key, default)

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
        normalized_key = key.strip().lower()
        if normalized_key in self.data:
            del self.data[normalized_key]
            return True
        else:
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
        normalized_prefix = prefix.strip().lower()
        matching_keys = [k for k in self.data if k.startswith(normalized_prefix)]
        return sorted(matching_keys)
