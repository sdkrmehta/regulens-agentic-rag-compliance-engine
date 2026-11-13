from pathlib import Path

import pytest


def test_frontend_source_compiles():
    source = Path("frontend/app.py").read_text(encoding="utf-8-sig")
    compile(source, "frontend/app.py", "exec")


def test_dashboard_uses_verified_integrity_and_dynamic_metrics():
    source = Path("frontend/app.py").read_text(encoding="utf-8-sig")
    assert 'requests.get(f"{base_url}/api/v1/metrics"' in source
    assert "verify_regulatory_document_integrity()" in source
    assert '"3.0"' not in source


def test_qdrant_healthcheck_requires_real_http_200_without_curl():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8").split("fastapi-backend:")[0]
    assert "/readyz" in compose and "GET /readyz" in compose and " 200 " in compose
    assert "curl" not in compose.lower()


def test_atomic_reindex_promotion_failure_preserves_existing_alias_and_docstore(monkeypatch):
    from src.retrieval.vector_store import RBIHybridVectorStore

    class Alias:
        def __init__(self, name, target):
            self.alias_name, self.collection_name = name, target

    class FakeClient:
        def __init__(self):
            self.target = "prod_v1"
        def get_collection_aliases(self):
            return type("Response", (), {"aliases": [Alias("prod_active", self.target)]})()
        def collection_exists(self, name):
            return name == "prod_v1"
        def update_collection_aliases(self, change_aliases_operations):
            raise RuntimeError("simulated promotion fault")
        def delete_collection(self, name):
            pass

    store = RBIHybridVectorStore.__new__(RBIHybridVectorStore)
    store.client = FakeClient()
    store._production_collection = "prod"
    store.collection_name = "prod_active"
    store._alias_target = lambda name: store.client.target
    store.init_collection = lambda **kwargs: None
    store.index_child_chunks = lambda **kwargs: None
    store._count_points_exact = lambda name: 1
    docstore_writes = []
    with pytest.raises(RuntimeError, match="simulated promotion fault"):
        store.atomic_reindex([{"child_id": "c1"}], on_success=lambda: docstore_writes.append(True))
    assert store.client.target == "prod_v1"
    assert docstore_writes == []
