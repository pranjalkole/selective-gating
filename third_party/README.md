# third_party/

## Retinexformer (vendored)

- **Upstream:** https://github.com/caiyuanhao1998/Retinexformer
- **Pinned commit:** see `RETINEXFORMER_PIN.txt`
  (`1e9a0efce4b306b6701b824768370ff26066c32a`)
- **License:** MIT (see `Retinexformer/LICENSE.txt`)
- **Citation:** Cai et al., *Retinexformer: One-stage Retinex-based Transformer
  for Low-light Image Enhancement*, ICCV 2023, arXiv:2303.06705

Only `Retinexformer/basicsr/models/archs/RetinexFormer_arch.py` is imported by
`selgate` (loaded by file path in `src/selgate/models/backbone.py`); the rest of
the upstream repo is kept for reference (configs, training code, metrics).

To update the pin:

```bash
git clone https://github.com/caiyuanhao1998/Retinexformer.git /tmp/retinex-upstream
cd /tmp/retinex-upstream && git checkout <commit>
rsync -a --delete --exclude .git /tmp/retinex-upstream/ \
    third_party/Retinexformer/
git rev-parse HEAD > third_party/RETINEXFORMER_PIN.txt
```
