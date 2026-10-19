import os
import subprocess
import sys

from haive.index.embed import FastEmbedder, cache_bytes


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
