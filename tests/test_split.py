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


# the canonical LOL-v1 eval15 filenames
_CANONICAL_TEST15 = [
    "1.png", "111.png", "146.png", "179.png", "22.png", "23.png", "493.png",
    "547.png", "55.png", "665.png", "669.png", "748.png", "778.png",
    "780.png", "79.png",
]


def test_committed_split_is_the_real_lol_v1_split():
    """The committed split must be carved from the real 485+15 dataset.

    Guards against committing a split generated from local smoke/fake data
    (happened once: a 24-image fake split silently broke Kaggle training).
    """
    from selgate.paths import split_file

    p = split_file()
    assert p.exists(), "committed split file missing"
    d = json.loads(p.read_text())
    assert len(d["train"]) == 461, "committed split must cover the real our485"
    assert len(d["val"]) == 24
    assert sorted(d["test"]) == sorted(_CANONICAL_TEST15)
    tv = set(d["train"]) | set(d["val"])
    assert not tv & set(d["test"]), "TEST LEAKAGE!"
