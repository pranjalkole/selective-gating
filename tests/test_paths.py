"""find_lol_root must cope with both Kaggle mount layouts:

- flat:      /kaggle/input/lol-dataset/lol_dataset/our485/low   (classic)
- nested:    /kaggle/input/datasets/<owner>/<slug>/...          (new CLI)
"""

import pytest

import selgate.paths as P


def _make_lol(root):
    """Create a minimal LOL-v1 layout under `root`."""
    for split in ("our485", "eval15"):
        for sub in ("low", "high"):
            (root / split / sub).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SELGATE_LOL_ROOT", raising=False)
    monkeypatch.delenv("SELGATE_OUT_DIR", raising=False)


def test_nested_kaggle_mount_layout(tmp_path, monkeypatch):
    """New-CLI layout: input/datasets/<owner>/<slug>/lol_dataset/..."""
    ki = tmp_path / "input"
    _make_lol(ki / "datasets" / "soumikrakshit" / "lol-dataset" / "lol_dataset")
    monkeypatch.setattr(P, "KAGGLE_INPUT", ki)

    found = P.find_lol_root()
    assert found is not None
    assert found.name == "lol_dataset"


def test_flat_kaggle_mount_layout(tmp_path, monkeypatch):
    """Classic layout: input/<slug>/our485/..."""
    ki = tmp_path / "input"
    _make_lol(ki / "lol-dataset")
    monkeypatch.setattr(P, "KAGGLE_INPUT", ki)

    found = P.find_lol_root()
    assert found is not None
    assert found.name == "lol-dataset"


def test_lol_root_not_found_returns_none(tmp_path, monkeypatch):
    ki = tmp_path / "input"
    (ki / "some-other-dataset").mkdir(parents=True)
    monkeypatch.setattr(P, "KAGGLE_INPUT", ki)

    assert P.find_lol_root() is None


def test_env_var_override(tmp_path, monkeypatch):
    root = _make_lol(tmp_path / "custom")
    monkeypatch.setenv("SELGATE_LOL_ROOT", str(root))

    assert P.find_lol_root() == root


def test_out_dir_kaggle_writes_to_working(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "KAGGLE_INPUT", tmp_path / "input")
    (tmp_path / "input").mkdir()
    monkeypatch.setattr(P, "KAGGLE_WORKING", tmp_path / "working")

    out = P.out_dir("smoke")
    assert out == tmp_path / "working" / "runs" / "smoke"
    assert out.is_dir()
