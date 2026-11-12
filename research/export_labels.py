"""Download human-readable labels for an open SAE from Neuronpedia's public data export.

    uv run python export_labels.py          # labels for the SAE in artifacts/models/gpt2-l6/sae

Reads the Neuronpedia id from ``<model-dir>/sae/manifest.json`` (e.g. ``gpt2-small/6-res-jb``),
downloads the explanation batches from Neuronpedia's S3 export, keeps one explanation per feature
and writes ``<model-dir>/sae/labels.json``:

    {"source": "...", "labels": ["references to grants and funding", null, ...]}   # index = feature id

Our own SAE gets the same file from ``autointerp.py`` (Phase 1) instead.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_URL = "https://neuronpedia-datasets.s3.amazonaws.com/"
# Explanations exist from several models; later entries are only used when earlier ones are missing.
PREFERRED_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]


def list_batches(neuronpedia_id: str) -> list[str]:
    prefix = f"v1/{neuronpedia_id}/explanations/"
    keys: list[str] = []
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(EXPORT_URL + "?" + urllib.parse.urlencode(query), timeout=60) as res:
            xml = res.read().decode()
        keys += re.findall(r"<Key>([^<]+\.jsonl\.gz)</Key>", xml)
        match = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not match:
            return sorted(keys)
        token = match.group(1)


def rank(row: dict) -> int:
    model = row.get("explanationModelName")
    return PREFERRED_MODELS.index(model) if model in PREFERRED_MODELS else len(PREFERRED_MODELS)


def clean(text: str) -> str:
    return " ".join(text.split()).rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=REPO_ROOT / "artifacts" / "models" / "gpt2-l6")
    ap.add_argument("--folder", default="sae", help="SAE folder inside model-dir (see export_sae.py --folder)")
    ap.add_argument("--neuronpedia", default=None, help="override the id from the SAE manifest")
    args = ap.parse_args()

    sae_dir = args.model_dir / args.folder
    manifest = json.loads((sae_dir / "manifest.json").read_text())
    neuronpedia_id = args.neuronpedia or manifest.get("neuronpedia")
    if not neuronpedia_id:
        ap.error("this SAE has no Neuronpedia id; labels for our own SAE come from autointerp.py")

    batches = list_batches(neuronpedia_id)
    if not batches:
        ap.error(f"no explanation export found for {neuronpedia_id}")

    best: dict[int, tuple[int, str]] = {}
    for i, key in enumerate(batches):
        print(f"\r{i + 1}/{len(batches)} {key}", end="", flush=True)
        with urllib.request.urlopen(EXPORT_URL + urllib.parse.quote(key), timeout=300) as res:
            for line in gzip.decompress(res.read()).splitlines():
                row = json.loads(line)
                index, text = int(row["index"]), clean(row.get("description") or "")
                if text and 0 <= index < manifest["d_sae"] and (index not in best or rank(row) < best[index][0]):
                    best[index] = (rank(row), text)
    print()

    labels = [best[i][1] if i in best else None for i in range(manifest["d_sae"])]
    out = {"source": f"Neuronpedia explanations for {neuronpedia_id} (https://www.neuronpedia.org)", "labels": labels}
    (sae_dir / "labels.json").write_text(json.dumps(out, ensure_ascii=False))
    manifest["files"]["labels"] = "labels.json"
    (sae_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote labels for {len(best)} of {manifest['d_sae']} features to {sae_dir / 'labels.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
