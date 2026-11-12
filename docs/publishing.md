# Publishing: model files and the site

The site is static (GitHub Pages) and downloads the model from a public Hugging Face dataset.
Two things have to exist before the live site works: the dataset with the files from
`artifacts/models/`, and the `ARTIFACTS_URL` repository variable that tells the build where it is.

```
research/ export  ──▶  artifacts/models/  ──▶  Hugging Face dataset  ◀── fetch ──  browser
                                                       ▲
web/ build (VITE_ARTIFACTS_URL = ARTIFACTS_URL) ──▶ GitHub Pages ──────────────────┘
```

## One-time setup

1. **Hugging Face token.** Create a token with *Write* access at
   <https://huggingface.co/settings/tokens/new?tokenType=write> and save it as a repository
   secret named `HF_TOKEN` at <https://github.com/pBariskan/Prizma/settings/secrets/actions/new>.
   Never paste the token into a chat, an issue or a file.
2. **GitHub Pages.** At <https://github.com/pBariskan/Prizma/settings/pages> set *Source* to
   **GitHub Actions**. Pages on a private repository needs GitHub Pro (free with the GitHub
   Student Developer Pack); otherwise the repository has to be public.
3. **Publish the artifacts** (next section), then set the `ARTIFACTS_URL` repository variable at
   <https://github.com/pBariskan/Prizma/settings/variables/actions/new> to the URL the workflow
   prints: `https://huggingface.co/datasets/<hf-user>/prizma-artifacts/resolve/main/models`.
4. **Deploy.** Re-run *Deploy site to GitHub Pages* at
   <https://github.com/pBariskan/Prizma/actions/workflows/pages.yml> (the variable is read at build
   time). The site is then at <https://pbariskan.github.io/Prizma/>.

## Publishing the artifacts

### With the workflow (recommended)

Run *Publish model artifacts* at
<https://github.com/pBariskan/Prizma/actions/workflows/publish-artifacts.yml> → *Run workflow*.
It exports GPT-2 with real weights on the runner, refuses to continue if the export does not match
Hugging Face (`--min-top1`), adds the concept dictionary (`sae/`, the open GPT-2 SAE, checked with
`--min-ev`), its labels from Neuronpedia and the precomputed example (`demo.json`), and uploads
`artifacts/models/` to `<token owner>/prizma-artifacts` (created as a public dataset on the first
run). The run summary shows the `ARTIFACTS_URL` value and the manifest with the verification
numbers. The model export takes about 5 minutes; the SAE, labels and example add a few more.

The steps for the concept dictionary need nothing extra from you: no token, no setting.

The workflow can only be started once its file is on `main`.

### By hand

```bash
cd research
uv sync
uv run python export_onnx.py        # ~180 MB into ../artifacts/models/gpt2-l6
uv run python export_steering.py
uv run python export_sae.py         # concept dictionary, +76 MB into ../artifacts/models/gpt2-l6/sae
uv run python export_labels.py      # concept labels from Neuronpedia
(cd ../web && npm ci && npm run export-demo -- --vector sae-12102)   # the example shown while loading
uv run hf auth login                # paste the token here
uv run hf upload <hf-user>/prizma-artifacts dataset_card.md README.md --repo-type dataset --no-private
uv run hf upload <hf-user>/prizma-artifacts ../artifacts/models models --repo-type dataset
```

## After a re-export

Upload again (same command or workflow run). Visitors do not need to clear anything: the browser
caches each model file under its SHA-256 from `manifest.json`, so a changed file is downloaded
again and the old copy is dropped. `manifest.json` and `steering.json` are small and are fetched on
every visit.

## Checking the live site

- Open <https://pbariskan.github.io/Prizma/?model=gpt2-l6>. While the model downloads, an example
  text is already on screen and can be explored. When it is ready the status row shows `gpt2`
  and `WebGPU` (or `WASM (slower, ...)`).
- Press *Generate* with the default settings. The steered panel should differ from the original
  one; `docs/phase0-verification.md` lists what to expect for a few prompts.
- Click a word and open the *Concepts* tab: ten labelled concepts should appear a moment after the
  model is ready. If the tab says the model has no concept dictionary, `<ARTIFACTS_URL>/gpt2-l6/sae/manifest.json`
  was not uploaded.
- If the status row says the model could not be loaded, open
  `<ARTIFACTS_URL>/gpt2-l6/manifest.json` in a browser tab: a 404 there means the variable or the
  upload is wrong.
