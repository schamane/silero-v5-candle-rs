#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "safetensors", "numpy", "silero-stress==1.5"]
# ///
"""Export Silero v5_cis_base_nostress and a Ukrainian golden for the Rust port.

`v5_cis_base_nostress` has no accentor of its own, but it expects stress marks
for Slavic languages. The golden therefore applies the Python `silero-stress`
accentor to the prepared text and then runs the manual 48 kHz pipeline; that is
exactly the order the Rust crate uses.

Usage: `uv run scripts/export_silero_cis_nostress.py /path/to/v5_cis_base_nostress.pt`.
Output (`models/silero-v5-cis-base-nostress/`) is gitignored (MIT weights).
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from silero_stress import load_accentor
from torch.package import PackageImporter

from export_silero_cis import (
    KEEP_PREFIXES,
    contiguous,
    run_pipeline,
    sha256,
    write_safetensors,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models/silero-v5-cis-base-nostress"
EXPECTED_SHA256 = "0405777e332906f0644e08a680f7cfdc2137ea864090079c1fdd30a43c1b8761"
VOICE = "ukr_igor"
TEXTS = [
    "Я з раннього дитинства дуже люблю слухати цікаві казки.",
    "Я з раннього дитинства дуже люблю слухати цікаві к+азки.",
]


def export_stages(part, tsm, text: str, accentor) -> tuple[dict, str, str]:
    prepared, _, has_text = part.prepare_text_input(text, "ukr")
    if not has_text:
        raise SystemExit(f"Text ohne sprechbare Zeichen: {text!r}")
    accented = accentor(prepared)
    stages = run_pipeline(part, tsm, accented, VOICE, "ukr")
    reference = part.apply_tts(text=accented, speaker=VOICE, sample_rate=48000)
    if stages["audio"].shape != reference.shape or not torch.equal(
        stages["audio"], reference
    ):
        raise SystemExit(
            f"manual pipeline differs from apply_tts for {text!r}: "
            f"{tuple(stages['audio'].shape)} vs {tuple(reference.shape)}"
        )
    return stages, prepared, accented


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_silero_cis_nostress.py /path/to/v5_cis_base_nostress.pt")
    package = Path(sys.argv[1])
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    digest = sha256(package)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"{package}: sha256 {digest} != {EXPECTED_SHA256}")

    print(f"loading {package}", file=sys.stderr)
    model = PackageImporter(str(package)).load_pickle("tts_models", "model")
    part = model.packages[0]
    tsm = part.models[0]
    if part.accentor is not None:
        raise SystemExit("v5_cis_base_nostress sollte keinen eingebauten Akzentor haben")
    accentor = load_accentor("ukr")

    OUT.mkdir(parents=True, exist_ok=True)
    state = {
        name: contiguous(value)
        for name, value in tsm.state_dict().items()
        if name.startswith(KEEP_PREFIXES)
    }
    write_safetensors(OUT / "acoustic.safetensors", state)
    window = contiguous(tsm.vocoder.head.istft.window)
    write_safetensors(OUT / "istft_window.safetensors", {"window": window})

    reloaded = load_file(str(OUT / "acoustic.safetensors"))
    if set(reloaded) != set(state):
        raise SystemExit("safetensors key set changed on round trip")
    for name, value in state.items():
        if not torch.equal(reloaded[name], value):
            raise SystemExit(f"{name}: tensor changed on round trip")

    symbols = list(part.symbols)
    istft = tsm.vocoder.head.istft
    config = {
        "symbols": "".join(symbols),
        "symbol_to_id": {k: int(v) for k, v in part.symbol_to_id.items()},
        "speakers": {k: int(v) for k, v in part.speaker_to_ids[0].items()},
        "speaker_rows": int(tsm.tacotron.speaker_embedding.weight.shape[0]),
        "sample_rate": 48000,
        "n_fft": int(istft.n_fft),
        "hop_length": int(istft.hop_length),
        "win_length": int(istft.win_length),
        "max_positions": 5000,
        "sil_value": float(tsm.sil_value),
        "mean_std_coef": [float(x) for x in tsm.mean_std_coef],
        "punkt_symbols": [int(x) for x in tsm.punkt_symbols],
        "point_id": int(tsm.point_id),
        "comma_ids": [int(x) for x in tsm.comma_ids],
        "accentor": {"lang": "ukr"},
    }
    (OUT / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    golden = OUT / "golden"
    golden.mkdir(exist_ok=True)
    lines = []
    for index, text in enumerate(TEXTS):
        stages, prepared, accented = export_stages(part, tsm, text, accentor)
        write_safetensors(golden / f"uk-{index}.safetensors", stages)
        lines.append(
            json.dumps(
                {
                    "id": index,
                    "text": text,
                    "prepared": prepared,
                    "accented": accented,
                    "voice": VOICE,
                },
                ensure_ascii=False,
            )
        )
        print(f"golden {index}: {accented}", file=sys.stderr)
    (golden / "text.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"wrote {OUT}: {len(state)} tensors, {len(window)} window values, "
        f"{len(TEXTS)} goldens",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
