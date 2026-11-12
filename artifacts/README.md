# artifacts/

Outputs of the research pipeline. Nothing here is committed except this file; the contents
are uploaded to a CDN (Cloudflare R2 or a Hugging Face dataset) for the live site.

This folder is the contract between `research/` and `web/`. The TypeScript side of the contract
is `web/src/engine/protocol.ts`; when a format changes, update this file, `protocol.ts` and
`CLAUDE.md` together.

```
artifacts/
  models/<name>/            e.g. gpt2-l6 (GPT-2 small split at layer 6)
    manifest.json           split layer, shapes, quantization, verification report
    embeddings.f16.bin      token-embedding table [vocab, D], fp16 (looked up in JS)
    part_a.onnx             tok_embeds [1,T,D] -> resid [1,T,D], attn [1,L,H,T,T]
    part_b.onnx             resid [1,T,D] -> last_logits [1,V], topk_probs/topk_ids [1,T,k], attn [1,n-L,H,T,T]
    steering.json           unit-norm steering directions + typical residual norm
    demo.json               precomputed example shown while the model downloads
    sae/                    the sparse autoencoder behind the Concepts and Steering views
      manifest.json         how to encode, shapes, file names, version, self-check, verification
      encoder.f16.bin       [d_sae, D] fp16; row f is the encoder weight of feature f
      biases.f32.bin        b_enc [d_sae] followed by b_dec [D], fp32
      decoder/NNNN.f16.bin  W_dec rows [NNNN * S, (NNNN + 1) * S), [rows, D] fp16 (S = decoder_shard_size)
      labels.json           {"source": ..., "labels": [string | null, ...]}, index = feature id
```

All binary files are raw little-endian arrays without headers.

Produce them with:

```bash
cd research
uv run python export_onnx.py          # -> artifacts/models/gpt2-l6/
uv run python export_steering.py      # -> .../steering.json
uv run python export_sae.py           # -> .../sae/ (open SAE: jbloom/GPT2-Small-SAEs-Reformatted)
uv run python export_labels.py        # -> .../sae/labels.json (Neuronpedia's public export)
cd ../web
npm run export-demo -- --vector sae-12102 --strength 0.5   # -> .../demo.json
```

## The SAE folder

`sae/manifest.json` tells the browser everything it needs, so the site works with any SAE that
fits this recipe. Encoding the residual vector `x` of one position:

```
if center_resid:          x = x - mean(x)          # mean over the D dimensions
if apply_b_dec_to_input:  x = x - b_dec
pre  = encoder @ x + b_enc
acts = relu(pre)                                   # activation "relu"
acts = relu(pre) on the k largest, 0 elsewhere     # activation "topk" (k in the manifest)
```

| Field | Meaning |
| --- | --- |
| `format` | Format version, currently `1` |
| `name`, `source` | Display name and where the weights came from |
| `hook`, `d_model` | Must match the model manifest (`blocks.6.hook_resid_pre`, 768) |
| `d_sae` | Number of features |
| `activation`, `k` | `"relu"` or `"topk"` (with `k`) |
| `center_resid` | The SAE expects the mean-centered residual. True for SAEs trained on a TransformerLens model with `center_writing_weights` (the default there, and what the open GPT-2 SAEs used) |
| `apply_b_dec_to_input` | Subtract `b_dec` before the encoder |
| `prepend_bos`, `bos_token_id` | The SAE was trained on contexts that start with `<\|endoftext\|>`. The engine starts every sequence with it when the model manifest or the SAE manifest sets `prepend_bos` |
| `version` | Hash of the weights; part of the browser cache key, so a swapped SAE is downloaded again |
| `files`, `decoder_shard_size`, `bytes` | File names (the decoder name has a `{shard}` placeholder, 4 digits) and sizes |
| `neuronpedia` | Optional `<model>/<sae>` id; the Concepts view links each feature to its Neuronpedia page |
| `selfcheck` | One residual vector and its 10 strongest features; the browser re-computes it after loading and refuses the SAE on a mismatch |
| `verification` | Explained variance and mean L0 on a few sentences, measured on the fp16 files, plus the explained variance with the opposite centering convention |

Steering with feature `f` (`sae-<f>` in share links and `steering.json`) adds
`strength * typical_resid_norm * W_dec[f] / |W_dec[f]|` to the residual; the browser fetches only
the decoder shard that holds row `f`.

### Swapping in our own SAE

No web code changes:

```bash
cd research
# first copy labels.json (and concepts/) from autointerp.py into artifacts/models/gpt2-l6/sae/:
# a labels.json with the right number of features is kept and registered in the manifest
uv run python export_sae.py --sae-path runs/<our-sae>     # a SAELens save: cfg.json + sae_weights.safetensors
cd ../web && npm run export-demo -- --vector sae-<id> --strength 0.5
```

`export_sae.py` reads the encode recipe from the SAELens `cfg.json` (top level, then `metadata`,
then the legacy defaults) and folds `rescale_acts_by_decoder_norm` into the weights. It only
rewrites the weight files and the manifest; other files in the folder are left alone.

To compare two SAEs side by side, export one into another folder with `--folder sae-own` (same
option on `export_labels.py`) and open the site with `?sae=sae-own`.
