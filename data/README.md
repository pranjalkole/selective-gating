# Where to get LOL-v1

The code auto-detects the dataset root (see `src/selgate/paths.py`):

1. `--root` / `data.root` config key
2. `SELGATE_LOL_ROOT` environment variable
3. `<repo>/data/LOL-v1` (this folder, local default)
4. any dataset attached under `/kaggle/input` on Kaggle

## Expected layouts (any of these works)

```
LOL-v1/
├── our485/{low,high}/0001.png … 0485.png     ← original release (preferred)
└── eval15/{low,high}/001.png … 015.png
```
```
LOL-v1/
├── Train/{input,target}/…                    ← Retinexformer-style split
└── Test/{input,target}/…
```
```
LOL-v1/
├── train/{low,high}/…                        ← generic split
└── test/{low,high}/…
```

## Sources

- **Original release:** `http://dao-lab.cn:6007/` (LOL-v1 "our485 + eval15")
- **On Kaggle:** search "LOL dataset" / "low light enhancement" datasets and
  attach one to the notebook. If its root folder differs from the layouts
  above, set the config key or env var to the folder that *directly contains*
  `our485/` (or `Train/`, `train/`).
- Extract/rename so that one of the layouts above holds, e.g.:

```bash
mkdir -p data/LOL-v1 && unzip lol-dataset.zip -d data/LOL-v1
# if it produced data/LOL-v1/LOLdataset/our485 … then:
#   SELGATE_LOL_ROOT=data/LOL-v1/LOLdataset
```

## Split rule (do not override)

`scripts/make_split.py` carves ~5% of the **485 train images** into a
validation split and writes `configs/split_lol_v1.json`. The 15 test images
are used only by `scripts/evaluate.py --split test`. `tests/test_split.py`
enforces that this never leaks.
