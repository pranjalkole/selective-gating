"""Split integrity: the test-15 must never leak into train or val."""

import json

from selgate.data import make_split


def test_split_counts_and_no_leakage(fake_lol, tmp_path):
    out = tmp_path / "split.json"
    split = make_split(fake_lol, val_fraction=0.1, seed=0, out_file=out, force=True)

    assert len(split["train"]) + len(split["val"]) == 40
    assert len(split["val"]) == 4  # 10% of 40
    assert len(split["test"]) == 15

    train_val = set(split["train"]) | set(split["val"])
    assert not train_val & set(split["test"]), "TEST LEAKAGE!"
    assert not set(split["train"]) & set(split["val"])


def test_split_deterministic(fake_lol, tmp_path):
    a = make_split(fake_lol, val_fraction=0.1, seed=0, out_file=tmp_path / "a.json", force=True)
    b = make_split(fake_lol, val_fraction=0.1, seed=0, out_file=tmp_path / "b.json", force=True)
    assert a["train"] == b["train"] and a["val"] == b["val"]

    c = make_split(fake_lol, val_fraction=0.1, seed=1, out_file=tmp_path / "c.json", force=True)
    assert c["val"] != a["val"], "different seed should give a different split"


def test_existing_split_reused(fake_lol, tmp_path):
    out = tmp_path / "split.json"
    make_split(fake_lol, val_fraction=0.1, seed=0, out_file=out, force=True)
    mtime = out.stat().st_mtime
    make_split(fake_lol, val_fraction=0.1, seed=0, out_file=out)  # no force -> reuse
    assert out.stat().st_mtime == mtime
    with open(out) as f:
        assert "train" in json.load(f)
