import os
import subprocess
import sys

from haive.index.embed import FastEmbedder, cache_bytes, model_cache_dir


def test_hub_progress_bars_are_off_once_haive_is_imported():
    # haive draws its own download bar; the Hub's switch only works if set before its first import.
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_DISABLE_PROGRESS_BARS"}
    code = (
        "import haive.cli, haive.runtime\n"
        "from huggingface_hub.utils import are_progress_bars_disabled\n"
        "assert are_progress_bars_disabled()\n"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_download_detection_and_cache_size(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    emb = FastEmbedder("jinaai/jina-embeddings-v2-base-code")
    assert not emb.is_downloaded() and cache_bytes() == 0
    assert emb.download_size() == 640_000_000

    snap = tmp_path / "models--jinaai--jina-embeddings-v2-base-code" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / "x").write_bytes(b"0" * 1000)  # xet keeps large files in a shared blobs dir
    assert emb.is_downloaded() and cache_bytes() == 1000


def test_embedding_model_cache_lives_in_data_dir(monkeypatch, haive_home):
    import logging
    import types

    seen = {}

    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir=None):
            seen["cache_dir"] = cache_dir

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    FastEmbedder().model  # noqa: B018 - property builds the model
    assert seen["cache_dir"] == str(haive_home / "data" / "models") == model_cache_dir()
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/custom/cache")
    assert model_cache_dir() == "/custom/cache"

    hub = logging.getLogger("huggingface_hub.utils._http")
    hidden = logging.LogRecord(
        "x", logging.WARNING, "", 0, "You are sending unauthenticated requests to the HF Hub.", None, None
    )
    shown = logging.LogRecord("x", logging.WARNING, "", 0, "Some other hub warning", None, None)
    assert not all(f.filter(hidden) for f in hub.filters)
    assert all(f.filter(shown) for f in hub.filters)
