# artifacts/

Outputs of the research pipeline. Nothing here is committed except this file; the contents
are uploaded to a public Hugging Face dataset for the live site (`docs/publishing.md`).

This folder is the contract between `research/` and `web/`. When it changes, update this file,
`web/src/engine/protocol.ts` and `CLAUDE.md` together.

```
artifacts/
  models/<name>/          e.g. gpt2-l6 (GPT-2 small split at layer 6)
    manifest.json         split layer, shapes, file hashes, verification report
    embeddings.f16.bin    token-embedding table [V, D], raw little-endian fp16
    part_a.onnx           tok_embeds [1,T,D] -> resid [1,T,D], attn [1,L,H,T,T]
    part_b.onnx           resid [1,T,D] -> last_logits [1,V], topk_probs/topk_ids [1,T,k], attn [1,n-L,H,T,T]
    steering.json         unit-norm steering directions, typical residual norm, strength ranges
```

Produce them with:

```bash
cd research
uv run python export_onnx.py          # -> artifacts/models/gpt2-l6/ (~180 MB)
uv run python export_steering.py      # -> artifacts/models/gpt2-l6/steering.json
```

## manifest.json

| Field | Meaning |
| --- | --- |
| `name`, `source` | Folder name; Hugging Face model id, or `random` for test models |
| `split_layer`, `hook` | The residual stream entering this block, e.g. `6` and `blocks.6.hook_resid_pre` |
| `n_layer`, `n_head`, `d_model`, `vocab_size`, `n_positions`, `top_k` | Shapes |
| `prepend_bos`, `bos_token_id` | The engine puts this token (`<|endoftext|>`, 50256) in front of every prompt. The steering calibration and the SAE assume it |
| `quantization` | `q8`, `q4` or `none`: weight quantization of the transformer blocks (`MatMulNBits`, block size 32, asymmetric) |
| `files` | File names of `embeddings`, `part_a`, `part_b` |
| `external_data` | Per part, the files it reads as ONNX external data. Part B has no unembedding matrix of its own: it points at `embeddings.f16.bin`, and the engine must pass those bytes when it creates the session |
| `bytes`, `sha256` | Size and hash of each file. The hash is the browser's cache key, so a re-export is downloaded again |
| `verification` | `fp32` (before quantization) and `final` reports against Hugging Face on real text: `top1_agreement`, `argmax_match`, `kl_mean`, `kl_max`, `onnx_vs_hf_max_abs`, `greedy` |

## steering.json

| Field | Meaning |
| --- | --- |
| `hook`, `d_model` | Where the vectors apply |
| `typical_resid_norm` | Median residual norm at the hook (BOS position excluded). Steering adds `strength * typical_resid_norm * vector` at every position |
| `suggested_range` | Fallback slider range for vectors without their own |
| `vectors[]` | `id`, `label`, `source` (`actadd` or `sae`), `detail`, `vector` (unit norm, `d_model` numbers), `raw_norm`, and optionally `range` `[lo, hi]`, `default_strength`, `url` |

SAE vectors have the id `sae-<feature id>` and a `url` to the feature's Neuronpedia page.
