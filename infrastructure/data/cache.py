from typing import Dict, Any, Optional
from domain.interfaces import ICache

class InMemoryCache(ICache):
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def has(self, key: str) -> bool:
        return key in self._cache