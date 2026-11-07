# artifacts/

Outputs of the research pipeline. Nothing here is committed except this file; the contents
are uploaded to a CDN (Cloudflare R2 or a Hugging Face dataset) for the live site.

This folder is the contract between `research/` and `web/`.

```
artifacts/
  models/<name>/          e.g. gpt2-l6 (GPT-2 small split at layer 6)
    manifest.json         split layer, shapes, quantization, verification report
    part_a.onnx           input_ids [1,T] -> resid [1,T,D], attn [1,L,H,T,T]
    part_b.onnx           resid [1,T,D] -> last_logits [1,V], topk_probs/topk_ids [1,T,k], attn [1,n-L,H,T,T]
    steering.json         unit-norm steering directions + typical residual norm
```

Produce them with:

```bash
cd research
uv run python export_onnx.py          # -> artifacts/models/gpt2-l6/
uv run python export_steering.py      # -> artifacts/models/gpt2-l6/steering.json
```
