"""Issue #13, the swap — fastembed in place of sentence-transformers.

Criteria 8-11. The tests that need the real model run it in a subprocess (so they test what the
server process does, and no test module's mock can stand in for it), against a model directory
taken from SYNAPTRA_TEST_MODEL_DIR, else the normal one, downloading it once if it is missing. They are
skipped when the model is neither there nor downloadable.

Nothing here touches the running service or its store: in-memory backend, throwaway HOME.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
GOLDEN = ROOT / "tests" / "data" / "golden_embeddings.json"
FORBIDDEN = {"torch", "transformers", "sentence_transformers", "sklearn"}


def _env(home: Path, model_dir: Path | None, **extra) -> dict:
    """An environment that cannot reach the owner's live store, with an in-memory engine."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SYNAPTRA_")}
    env.update(
        USERPROFILE=str(home),
        HOME=str(home),
        SYNAPTRA_BACKEND="surrealkv-file",
        SYNAPTRA_DB="mem://",
    )
    if model_dir is not None:
        env["SYNAPTRA_MODEL_DIR"] = str(model_dir)
    env.update(extra)
    return env


def _run(code: str, env: dict, timeout=300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(SRC)!r})\n" + code,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _json(done: subprocess.CompletedProcess) -> dict:
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------ the model files
@pytest.fixture(scope="session")
def model_files() -> Path:
    """A directory that holds the model, or skip."""
    sys.path.insert(0, str(SRC))
    from synaptra.embeddings import EmbeddingService, model_dir

    cache = Path(os.environ.get("SYNAPTRA_TEST_MODEL_DIR") or model_dir())
    cache.mkdir(parents=True, exist_ok=True)
    from fastembed import TextEmbedding

    try:
        TextEmbedding(
            EmbeddingService.MODEL_NAME,
            cache_dir=str(cache),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {exc}")
    return cache


# ------------------------------------------------------------------ criterion 8: no torch, no transformers
def test_the_package_no_longer_depends_on_sentence_transformers_torch_or_transformers():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["dependencies"]
    names = {
        re.split(r"[<>=!~ \[]", d, maxsplit=1)[0].lower().replace("_", "-")
        for d in deps
    }
    assert not names & {"sentence-transformers", "torch", "transformers"}, names
    assert "fastembed" in names


def test_no_source_file_imports_torch_transformers_or_sentence_transformers():
    pattern = re.compile(
        r"^\s*(?:import|from)\s+(torch|transformers|sentence_transformers)\b", re.M
    )
    hits = [
        str(p.relative_to(ROOT))
        for p in (SRC / "synaptra").rglob("*.py")
        if "tests" not in p.parts and pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert hits == []


def test_the_server_process_never_imports_torch_or_transformers(model_files, tmp_path):
    code = (
        "import json\n"
        "from synaptra import server\n"
        "eng = server._get_engine()\n"
        "eng.embeddings.warmup()\n"
        "eng.embeddings.embed('hello')\n"
        "top = {m.split('.')[0] for m in sys.modules}\n"
        f"print(json.dumps({{'state': eng.embeddings.state, 'bad': sorted(top & set({sorted(FORBIDDEN)!r}))}}))\n"
    )
    out = _json(_run(code, _env(tmp_path, model_files)))
    assert out == {"state": "ready", "bad": []}


# ------------------------------------------------------------------ criterion 9: golden parity
def test_the_golden_fixture_covers_what_it_has_to():
    g = json.loads(GOLDEN.read_text(encoding="utf-8"))
    texts = [i["text"] for i in g["items"]]
    assert len(texts) >= 20 and g["dimensions"] == 384 and g["max_seq_length"] == 256
    assert all(len(i["vector"]) == 384 for i in g["items"])
    assert "" in texts and any(
        len(t.split()) > 300 for t in texts
    )  # empty; past 256 tokens
    assert any("🎉" in t for t in texts) and any(
        "日本語" in t for t in texts
    )  # emoji; CJK
    norms = [np.linalg.norm(i["vector"]) for i in g["items"]]
    assert all(abs(n - 1.0) < 1e-4 for n in norms)


def test_the_new_embedder_matches_the_vectors_the_old_model_made(model_files, tmp_path):
    code = (
        "import json, numpy as np\n"
        f"g = json.load(open({str(GOLDEN)!r}, encoding='utf-8'))\n"
        "from synaptra.embeddings import EmbeddingService\n"
        "svc = EmbeddingService(); svc.warmup()\n"
        "new = svc.embed_batch([i['text'] for i in g['items']])\n"
        "old = np.array([i['vector'] for i in g['items']], dtype=np.float32)\n"
        "cos = (new * old).sum(axis=1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(old, axis=1))\n"
        "print(json.dumps({'cos': [float(c) for c in cos], 'texts': [i['text'][:30] for i in g['items']]}))\n"
    )
    out = _json(_run(code, _env(tmp_path, model_files)))
    worst = min(zip(out["cos"], out["texts"]))
    assert all(c >= 0.9999 for c in out["cos"]), f"worst: {worst}"
    print(f"\ngolden parity: {len(out['cos'])} texts, min cosine {min(out['cos']):.8f}")


# ------------------------------------------------------------------ criteria 10, 11: the model directory, offline
def test_no_source_file_hardcodes_an_absolute_path():
    drive = re.compile(r"""["'](?:[A-Za-z]:[\\/]|/(?:home|Users)/)""")
    hits = []
    for p in (SRC / "synaptra").rglob("*.py"):
        if "tests" in p.parts:
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if drive.search(line) and not line.lstrip().startswith("#"):
                hits.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()}")
    assert hits == []


_OFFLINE = dict(
    HTTP_PROXY="http://127.0.0.1:9",
    HTTPS_PROXY="http://127.0.0.1:9",
    http_proxy="http://127.0.0.1:9",
    https_proxy="http://127.0.0.1:9",
    NO_PROXY="",
    no_proxy="",
)  # a proxy nothing listens on: any attempt to reach the network fails


_BOOT = (
    "import json, time\n"
    "from synaptra.embeddings import EmbeddingService\n"
    "svc = EmbeddingService(); t = time.monotonic()\n"
    "try:\n"
    "    svc.warmup(); err = None\n"
    "except Exception as e:\n"
    "    err = f'{type(e).__name__}: {e}'\n"
    "ok = svc.state == 'ready' and svc.embed('hello').shape == (384,)\n"
    "print(json.dumps({'state': svc.state, 'ok': ok, 'seconds': round(time.monotonic() - t, 1), 'err': err}))\n"
)


def test_with_the_cache_filled_and_the_network_off_the_server_boots_to_ready_even_after_a_rename(
    model_files,
):
    import tempfile

    # A short base path: the model cache's own file names are long, and Windows stops at 260.
    with tempfile.TemporaryDirectory(prefix="cmswap") as base:
        base = Path(base)
        # the model directory sits under HOME, as it does in production
        home_a = base / "a"
        shutil.copytree(model_files, home_a / ".synaptra" / "models")
        first = _json(_run(_BOOT, _env(home_a, None, **_OFFLINE)))
        assert first["state"] == "ready" and first["ok"], first

        home_b = base / "b"  # the folder is renamed: nothing may remember the old name
        home_a.rename(home_b)
        second = _json(_run(_BOOT, _env(home_b, None, **_OFFLINE)))
        assert second["state"] == "ready" and second["ok"], second
        print(
            f"\noffline boot to ready: {first['seconds']} s, after a rename {second['seconds']} s"
        )


def test_the_network_really_is_off_in_that_test(tmp_path):
    """The control: with nothing cached, the same offline boot fails, so the test above proves something."""
    out = _json(_run(_BOOT, _env(tmp_path, tmp_path / "empty", **_OFFLINE)))
    assert out["state"] == "failed" and not out["ok"] and out["err"], out
