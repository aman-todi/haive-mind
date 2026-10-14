from unittest.mock import MagicMock

import pytest

from haive.store.base import MemoryStore, normalize_filter
from haive.store.chroma_store import ChromaStore, chroma_where
from haive.store.pinecone_store import PineconeError, PineconeStore
from haive.types import Chunk


def chunks():
    return [
        Chunk.make(
            "a", "x.py", "code", "def fetch_user(): pass", symbol="fetch_user", start_line=1, end_line=1
        ),
        Chunk.make("a", "README.md", "doc", "# Deploy with docker", symbol="Deploy"),
        Chunk.make("b", "y.py", "code", "def compute_tax(): pass", symbol="compute_tax"),
    ]


def test_normalize_filter():
    assert normalize_filter({"repo": ["a"], "kind": "code", "x": None, "y": []}) == {
        "repo": {"$in": ["a"]},
        "kind": {"$eq": "code"},
    }
    assert chroma_where({"repo": ["a"]}) == {"repo": {"$in": ["a"]}}
    assert chroma_where({"repo": ["a"], "kind": ["code"]}) == {
        "$and": [{"repo": {"$in": ["a"]}}, {"kind": {"$in": ["code"]}}]
    }
    assert chroma_where(None) is None


@pytest.mark.parametrize("kind", ["memory", "chroma"])
def test_store_contract(kind, embedder):
    store = (
        MemoryStore(embedder) if kind == "memory" else ChromaStore(embedder, collection=f"t{id(embedder)}")
    )
    cs = chunks()
    store.upsert(cs)
    assert store.count() == 3
    assert store.count({"repo": ["a"]}) == 2
    hits = store.search("fetch user", 2, {"repo": ["a"], "kind": ["code"]})
    assert (
        hits[0].path == "x.py"
        and hits[0].symbol == "fetch_user"
        and hits[0].text.startswith("def fetch_user")
    )
    assert all(h.repo == "a" for h in hits)
    store.delete([cs[0].id])
    assert store.count({"repo": ["a"]}) == 1


def test_pinecone_store_with_mock_client(embedder):
    pc = MagicMock()
    pc.has_index.return_value = False
    pc.describe_index.return_value = {"dimension": 768, "metric": "cosine"}
    index = pc.Index.return_value
    store = PineconeStore(embedder, "haive", client=pc)
    assert store.ensure_index() is True
    pc.create_index.assert_called_once()
    store.upsert(chunks())
    sent = index.upsert.call_args.kwargs["vectors"]
    assert len(sent) == 3 and len(sent[0]["values"]) == 768 and sent[0]["metadata"]["repo"] == "a"
    index.query.return_value = {
        "matches": [
            {
                "id": "a:x.py:1",
                "score": 0.9,
                "metadata": {
                    "repo": "a",
                    "path": "x.py",
                    "kind": "code",
                    "start_line": 1,
                    "end_line": 1,
                    "text": "t",
                },
            }
        ]
    }
    hits = store.search("q", 5, {"repo": ["a"], "kind": ["code"]})
    assert index.query.call_args.kwargs["filter"] == {"repo": {"$in": ["a"]}, "kind": {"$in": ["code"]}}
    assert hits[0].path == "x.py" and hits[0].score == 0.9
    index.list.return_value = iter([["a:1", "a:2"], ["a:3"]])
    assert store.count({"repo": ["a"]}) == 3


def test_pinecone_dimension_mismatch(embedder):
    pc = MagicMock()
    pc.has_index.return_value = True
    pc.describe_index.return_value = {"dimension": 384, "metric": "cosine"}
    with pytest.raises(PineconeError, match="dimension 384"):
        PineconeStore(embedder, client=pc).ensure_index()


def test_pinecone_requires_key(embedder, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(PineconeError):
        PineconeStore(embedder)
