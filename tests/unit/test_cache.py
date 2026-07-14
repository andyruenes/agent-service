"""Tests unitarios para el módulo de cache Redis."""

from app.cache.redis_cache import build_cache_key


class TestBuildCacheKey:
    """Tests de la generación de claves SHA-256."""

    def test_key_starts_with_prefix(self):
        key = build_cache_key("hola", "agent-1")
        assert key.startswith("cache:")

    def test_same_input_same_key(self):
        key1 = build_cache_key("hola mundo", "agent-1")
        key2 = build_cache_key("hola mundo", "agent-1")
        assert key1 == key2

    def test_different_query_different_key(self):
        key1 = build_cache_key("hola", "agent-1")
        key2 = build_cache_key("adiós", "agent-1")
        assert key1 != key2

    def test_different_agent_different_key(self):
        key1 = build_cache_key("hola", "agent-1")
        key2 = build_cache_key("hola", "agent-2")
        assert key1 != key2

    def test_case_insensitive(self):
        key1 = build_cache_key("Hola Mundo", "agent-1")
        key2 = build_cache_key("hola mundo", "agent-1")
        assert key1 == key2

    def test_trims_whitespace(self):
        key1 = build_cache_key("  hola  ", "agent-1")
        key2 = build_cache_key("hola", "agent-1")
        assert key1 == key2

    def test_key_is_valid_sha256_length(self):
        key = build_cache_key("test", "agent-1")
        # cache: + 64 hex chars
        assert len(key) == 6 + 64
