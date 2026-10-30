#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "safetensors", "numpy", "silero-stress==1.5"]
# ///
"""Export the Ukrainian `silero-stress` accentor for the Rust port.

Reads `accentor-ukr.pt` from the `silero-stress` pip package (MIT), restores the
int8 embedding once, and writes f32 safetensors plus `ngrams.txt` into the
`v5_cis_base_nostress` model directory. The golden files pin the Python
tokenizer, the per-word logits and the accented sentences.

Usage:
    uv run scripts/export_accentor_uk.py
    uv run scripts/export_accentor_uk.py --out models/silero-v5-cis-ext
    uv run scripts/export_accentor_uk.py /path/to/accentor-ukr.pt

Output (`models/silero-v5-cis-base-nostress/`) is gitignored.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.package import PackageImporter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models/silero-v5-cis-base-nostress"
EXPECTED_SHA256 = "3a64944182f7e34ef831e990dc807246161e20c14c923506db0a20b8120ce13c"

# Sentences cover punctuation, uppercase, hyphens, apostrophes and one-vowel
# words. They are intentionally independent of the TTS golden.
SENTENCES = [
    "Я з раннього дитинства дуже люблю слухати цікаві казки.",
    "Привіт, як справи?",
    "Україна — велика держава.",
    "Моя сестра любить читати книжки ввечері.",
    "Він іде до школи, а вона працює вдома.",
    "Ґанок стоїть біля хати.",
    "П'ять, сім, дев'ять — це числа.",
    "Працювати-навчатися-відпочивати.",
    "І я, і ти, і всі ми разом.",
    "Авіаперевезення потребує уваги.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().clone()


def find_package(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    try:
        from importlib.resources import files

        import silero_stress

        path = files("silero_stress.data").joinpath("accentor-ukr.pt")
        if path.is_file():
            return Path(path)
    except Exception as error:  # noqa: BLE001 - report the actionable hint below
        raise SystemExit(
            "accentor-ukr.pt nicht gefunden; `uv run scripts/export_accentor_uk.py` "
            f"verwenden oder den Pfad als Argument angeben ({error})"
        ) from None
    raise SystemExit(
        "accentor-ukr.pt nicht gefunden; `uv run scripts/export_accentor_uk.py` "
        "verwenden oder den Pfad als Argument angeben"
    )


def dequantize(model) -> None:
    """`load_accentor` aus `silero_stress` macht genau diese eine Restaurierung."""
    quantized = model.embedding.weight.data.clone()
    restored = model.scale * (quantized - model.zero_point)
    model.embedding.weight.data = restored.to(torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ukrainische silero-stress-Gewichte exportieren."
    )
    parser.add_argument("package", nargs="?", type=Path, help="accentor-ukr.pt")
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help=f"Zielverzeichnis (Standard: {OUT.relative_to(ROOT)})",
    )
    args = parser.parse_args()
    out = args.out
    package = find_package(str(args.package) if args.package else None)
    digest = sha256(package)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"{package}: sha256 {digest} != {EXPECTED_SHA256}")

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    accentor = (
        PackageImporter(str(package))
        .load_pickle("accentor_models", "accentor")
        .accentor
    )
    model = accentor.model
    dequantize(model)

    # Cheap sanity gate: the restored model must reproduce known predictions.
    probes = {"привіт": 1, "дивитися": 1, "з": 0, "моя": 1}
    for word, expected in probes.items():
        logits, _ = model([word])
        prediction = int(torch.softmax(logits, dim=1).argmax(dim=1))
        if prediction != expected:
            raise SystemExit(
                f"probe {word!r}: class {prediction} statt {expected}"
            )

    out.mkdir(parents=True, exist_ok=True)
    state = {
        name: contiguous(value)
        for name, value in model.state_dict().items()
    }
    if "embedding.weight" not in state or state["embedding.weight"].dtype != torch.float32:
        raise SystemExit("embedding.weight fehlt oder ist nicht f32")
    save_file(state, str(out / "accentor.safetensors"))

    ngrams = model.embedding.ngram_dict
    ordered = sorted(ngrams, key=ngrams.__getitem__)
    if [ngrams[gram] for gram in ordered] != list(range(len(ordered))):
        raise SystemExit("ngram ids sind nicht 0..N-1")
    (out / "ngrams.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8")

    golden = {"package_sha256": digest, "sentences": [], "words": [], "logits": []}
    seen: dict[str, int] = {}
    for text in SENTENCES:
        raw_tokens, clean_tokens, mask = accentor._tokenize(text)
        logits, _ = model(clean_tokens)
        for token_index, word in enumerate(clean_tokens):
            if word not in seen:
                seen[word] = len(golden["words"])
                golden["words"].append(word)
                golden["logits"].append(logits[token_index].tolist())
        golden["sentences"].append(
            {
                "text": text,
                "raw_tokens": raw_tokens,
                "clean_tokens": clean_tokens,
                "prediction_mask": [bool(value) for value in mask],
                "accented": accentor(text),
            }
        )

    # Store logits once per unique clean token, in first-seen order.
    unique_logits = torch.tensor(golden["logits"], dtype=torch.float32)
    (out / "golden").mkdir(exist_ok=True)
    save_file({"logits": unique_logits}, str(out / "golden/accentor.safetensors"))
    (out / "golden/accentor.json").write_text(
        json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    metadata = {
        "lang": "ukr",
        "package_sha256": digest,
        "ngram_count": len(ordered),
        "embedding_dim": int(state["embedding.weight"].shape[1]),
        "max_vowels": int(state["stress_clf.6.weight"].shape[0]),
        "stress_threshold": 0.5,
        "stress_token": accentor.stress_token,
        "vowels": accentor.vowels,
    }
    (out / "accentor.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # If the target already holds a TTS config, enable the accentor there. This
    # keeps the two exporters order-independent for `v5_cis_ext`.
    config_path = out / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if "bert" in config or "tokenizer" in config:
            print(
                f"{config_path}: russisches Modell, accentor-Block nicht gesetzt",
                file=sys.stderr,
            )
        else:
            config["accentor"] = {"lang": "ukr"}
            config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    print(
        f"wrote {out}: {len(state)} tensors, {len(ordered)} ngrams, "
        f"{len(golden['sentences'])} sentences, {len(seen)} unique words",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
