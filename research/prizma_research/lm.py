"""Loading GPT-2 for the Phase 1 pipeline, with the same offline modes as ``export_onnx.py``.

    load_lm()                        real GPT-2 small (needs huggingface.co)
    load_lm(random=True)             full-size GPT-2 with random weights, no download
    load_lm(random=True, tiny=True)  the 4-layer, 64-dim model of ``export_onnx.py --random --tiny``

Every Phase 1 script reads the residual stream through :class:`HookedGPT2` or
:func:`resid_at`, which run our own split forward pass (``gpt2_split.GPT2``). That is the
same code the ONNX export traces, so an SAE trained here sees exactly the tensor that
part A hands to the browser: the raw Hugging Face residual, no TransformerLens weight
processing (no ``center_writing_weights``, no ``fold_ln``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn
from transformer_lens import HookedRootModule
from transformer_lens.hook_points import HookPoint

from prizma_research.gpt2_split import GPT2

# GPT-2's only special token. It is both BOS and EOS, and the browser prepends it to every prompt.
EOT_ID = 50256
TINY_CONFIG = dict(n_embd=64, n_layer=4, n_head=4, n_positions=128)
TINY_SPLIT_LAYER = 2
DEFAULT_SPLIT_LAYER = 6


def _gpt2_byte_of_char() -> dict[str, int]:
    """GPT-2's byte-level BPE spells every byte as one printable character; this is the way back.

    Same table as ``bytes_to_unicode`` in OpenAI's ``encoder.py``, inverted.
    """
    byte_values = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    chars = byte_values[:]
    extra = 0
    for b in range(256):
        if b not in byte_values:
            byte_values.append(b)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(byte_values, chars)}


_GPT2_BYTE_OF_CHAR = _gpt2_byte_of_char()


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """Offline stand-in for the GPT-2 tokenizer: one token per UTF-8 byte (ids 0-255).

    Only used with ``--random`` so that the pipeline can run without downloading anything.
    """

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return "".join("<|endoftext|>" if i == EOT_ID else chr(i) if 32 <= i < 127 else "�" for i in ids)

    def token_bytes(self, token_id: int) -> bytes:
        return b"<|endoftext|>" if token_id == EOT_ID else bytes([token_id % 256])


@dataclass
class LM:
    gpt: GPT2
    tokenizer: Tokenizer
    name: str  # "gpt2", "gpt2-random" or "gpt2-random-tiny" (matches the export_onnx.py folder names)
    layer: int  # split layer: the SAE reads blocks.{layer}.hook_resid_pre
    random: bool

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer}.hook_resid_pre"

    @property
    def d_model(self) -> int:
        return self.gpt.cfg.n_embd

    @property
    def device(self) -> torch.device:
        return self.gpt.wte.weight.device

    def token_bytes(self, token_id: int) -> bytes:
        """The UTF-8 bytes one token stands for.

        Not ``tokenizer.decode([id])``: GPT-2 splits most non-ASCII characters (curly quotes, dashes,
        accents) over two or three tokens, and decoding such a token alone gives "�". With the bytes,
        ``examples.merge_utf8`` can put the character back together.
        """
        if isinstance(self.tokenizer, ByteTokenizer):
            return self.tokenizer.token_bytes(token_id)
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        try:
            return bytes(_GPT2_BYTE_OF_CHAR[c] for c in token)
        except KeyError:  # not a byte-level token (an added special token)
            return token.encode("utf-8")


def load_lm(model: str = "gpt2", *, random: bool = False, tiny: bool = False, layer: int | None = None,
            device: str = "cpu") -> LM:
    from transformers import GPT2Config as HFConfig
    from transformers import GPT2LMHeadModel

    if random:
        # Must stay identical to export_onnx.load_model: same seed and config give the same weights,
        # so an SAE trained here matches artifacts/models/gpt2-random(-tiny).
        torch.manual_seed(0)
        hf = GPT2LMHeadModel(HFConfig(**TINY_CONFIG) if tiny else HFConfig()).eval()
        tokenizer: Tokenizer = ByteTokenizer()
        name = "gpt2-random-tiny" if tiny else "gpt2-random"
        default_layer = TINY_SPLIT_LAYER if tiny else DEFAULT_SPLIT_LAYER
    else:
        from transformers import GPT2TokenizerFast

        hf = GPT2LMHeadModel.from_pretrained(model).eval()
        tokenizer = GPT2TokenizerFast.from_pretrained(model)
        name = model.replace("/", "--")
        default_layer = DEFAULT_SPLIT_LAYER
    gpt = GPT2.from_hf(hf).to(device)
    for p in gpt.parameters():
        p.requires_grad_(False)
    layer = default_layer if layer is None else layer
    if not 0 < layer < gpt.cfg.n_layer:
        raise ValueError(f"layer must be in [1, {gpt.cfg.n_layer - 1}], got {layer}")
    return LM(gpt=gpt, tokenizer=tokenizer, name=name, layer=layer, random=random)


@torch.no_grad()
def resid_at(gpt: GPT2, tokens: torch.Tensor, layer: int) -> torch.Tensor:
    """Residual stream entering block ``layer`` (what part A outputs). tokens [B, T] -> [B, T, D]."""
    x = gpt.embed(tokens)
    for blk in gpt.blocks[:layer]:  # not run_blocks: we do not need the stacked attention patterns
        x, _ = blk(x)
    return x


@torch.no_grad()
def logits_from_resid(gpt: GPT2, resid: torch.Tensor, layer: int) -> torch.Tensor:
    """Finish the forward pass from the split (what part B does). resid [B, T, D] -> logits [B, T, V]."""
    x = resid
    for blk in gpt.blocks[layer:]:
        x, _ = blk(x)
    return gpt.unembed(x)


def next_token_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over the batch, in nats."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1))


class _ResidHook(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hook_resid_pre = HookPoint()


class HookedGPT2(HookedRootModule):
    """Our split GPT-2 behind the one TransformerLens hook SAELens needs.

    SAELens fetches training activations with
    ``model.run_with_cache(tokens, names_filter=[hook], stop_at_layer=..., prepend_bos=False)``.
    This wrapper exposes ``blocks.{layer}.hook_resid_pre`` and stops right after it, so SAELens
    trains on the output of our own forward pass instead of a TransformerLens model.
    """

    def __init__(self, gpt: GPT2, layer: int):
        super().__init__()
        self.gpt = gpt
        self.layer = layer
        self.blocks = nn.ModuleDict({str(layer): _ResidHook()})
        self.setup()

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer}.hook_resid_pre"

    def forward(self, tokens: torch.Tensor, stop_at_layer: int | None = None, prepend_bos: bool = False,
                **_: object) -> torch.Tensor:
        if prepend_bos:
            raise ValueError("HookedGPT2 expects token ids that already start with <|endoftext|>")
        return self.blocks[str(self.layer)].hook_resid_pre(resid_at(self.gpt, tokens, self.layer))
