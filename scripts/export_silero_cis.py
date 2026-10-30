#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "safetensors", "numpy"]
# ///
"""Export Silero v5_cis_ext weights and a Ukrainian golden for the Rust port.

`v5_cis_ext` has no built-in accentor and no homograph solver. If
`models/silero-v5-cis-base-nostress/` contains the exported Ukrainian accentor,
this script copies `accentor.safetensors`/`ngrams.txt` and enables
`accentor.lang = "ukr"`; otherwise the caller supplies `+` marks. The accentor is
only applied to `ukr_*` voices. The acoustic and vocoder networks are the same
shape as `v5_ru` apart from the symbol/speaker embedding rows, so the Rust port
can reuse its pipeline. 48 kHz is written straight from the vocoder head; the
8/24 kHz PQMF paths are not exported.

Usage: `uv run scripts/export_silero_cis.py /path/to/v5_cis_ext.pt`.
Output (`models/silero-v5-cis-ext/`) is gitignored (CC BY-NC weights).
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.package import PackageImporter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models/silero-v5-cis-ext"
ACCENTOR_SOURCE = ROOT / "models/silero-v5-cis-base-nostress"
ACCENTOR_FILES = ("accentor.safetensors", "ngrams.txt")
EXPECTED_SHA256 = "e8185323fc5a341c229f8a8c9928512f665bf2736021bc600512499d0506eed2"
VOICE = "ukr_mykyta"
TEXT = "+Я з р+аннього дит+инства д+уже любл+ю сл+ухати цік+аві к+азки."

# Keys the Rust port loads. The PQMF filters are only needed for 8/24 kHz.
KEEP_PREFIXES = (
    "tacotron.",
    "dur_predictor.dur_pred.",
    "pitch_predictor.pitch_pred.",
    "vocoder.backbone.",
    "vocoder.head.out.",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().clone()


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: contiguous(value) for name, value in tensors.items()}, str(path))


def postprocess_durations(raw: torch.Tensor) -> torch.Tensor:
    dur = torch.exp(raw) - 1
    dur = torch.where(dur < 0, torch.zeros_like(dur), dur)
    dur = torch.round(dur)
    dur = torch.round(dur / 1.0)
    if bool(dur[0, 0] > 5):
        dur[0, 0] = 5
    if bool(dur[0, -1] > 7):
        dur[0, -1] = 7
    dur[0, -2] = 13
    if bool(dur[0, -3] > 13):
        dur[0, -3] = 13
    return dur


def run_pipeline(part, tsm, text: str, voice: str, lang: str):
    text, _, _ = part.prepare_text_input(text, lang)
    speaker_ids = torch.tensor([part.speaker_to_ids[0][voice]])
    sequence, _, _, pitch_coefs = part.merge_batch_model([text], [None], [1.0], [1.0])
    seq_len = sequence.shape[1]
    mask = torch.zeros(1, seq_len, dtype=torch.bool)

    raw_dur = tsm.dur_predictor(sequence, speaker_ids, mask, 1.0, None)
    dur = postprocess_durations(raw_dur)
    raw_pitch = tsm.pitch_predictor(sequence, speaker_ids, mask, 1.0)
    # `update_pitch_coef` mutiert sein Argument; der Golden braucht den Rohwert.
    pitch = tsm.update_pitch_coef(raw_pitch.clone(), pitch_coefs, speaker_ids)

    tacotron = tsm.tacotron
    embedded = tacotron.embedding(sequence)
    speaker = tacotron.speaker_embedding(speaker_ids).unsqueeze(1).expand(-1, seq_len, -1)
    encoder = tacotron.encoder(embedded, mask, None)
    cond = encoder + speaker
    cond = cond + tacotron.pitch_proj(pitch).transpose(1, 2)
    expanded = tacotron.len_reg(cond, dur)
    decoder = tacotron.decoder(expanded, None, None)
    mel = tacotron.lin(decoder).transpose(1, 2)

    backbone = tsm.vocoder.backbone(mel)
    packed = tsm.vocoder.head.out(backbone).transpose(1, 2)
    mag_raw, phase = torch.chunk(packed, 2, 1)
    mag = torch.clamp(torch.exp(mag_raw), None, 100.0)
    cos = torch.cos(phase)
    sin = torch.sin(phase)
    # 48 kHz goes straight from the vocoder head; the PQMF branch is not used.
    audio = tsm.vocoder(mel, 48000).reshape(-1)

    return {
        "sequence": sequence,
        "dur": dur,
        "dur_raw": raw_dur,
        "pitch": pitch,
        "pitch_raw": raw_pitch,
        "encoder": encoder,
        "cond": cond,
        "expanded": expanded,
        "decoder": decoder,
        "mel": mel,
        "backbone": backbone,
        "mag": mag,
        "cos": cos,
        "sin": sin,
        "audio": audio,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_silero_cis.py /path/to/v5_cis_ext.pt")
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
    lang = VOICE.split("_")[0]

    # Gate: the manual pipeline must reproduce apply_tts before anything is written.
    stages = run_pipeline(part, tsm, TEXT, VOICE, lang)
    reference = part.apply_tts(text=TEXT, speaker=VOICE, sample_rate=48000)
    if stages["audio"].shape != reference.shape or not torch.equal(
        stages["audio"], reference
    ):
        diff = (
            (stages["audio"] - reference).abs().max().item()
            if stages["audio"].shape == reference.shape
            else float("nan")
        )
        raise SystemExit(
            f"manual pipeline differs from apply_tts: diff={diff} "
            f"manual={tuple(stages['audio'].shape)} reference={tuple(reference.shape)}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    for name in ACCENTOR_FILES:
        target = OUT / name
        source = ACCENTOR_SOURCE / name
        if not target.is_file() and source.is_file():
            shutil.copy2(source, target)
            print(f"copied {source} -> {target}", file=sys.stderr)
    state = {
        name: contiguous(value)
        for name, value in tsm.state_dict().items()
        if name.startswith(KEEP_PREFIXES)
    }
    write_safetensors(OUT / "acoustic.safetensors", state)
    window = contiguous(tsm.vocoder.head.istft.window)
    write_safetensors(OUT / "istft_window.safetensors", {"window": window})

    # Round-trip gate for the serialization itself.
    from safetensors.torch import load_file

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
    }
    if all((OUT / name).is_file() for name in ACCENTOR_FILES):
        config["accentor"] = {"lang": "ukr"}
    else:
        print(
            "hinweis: kein Akzentor in models/silero-v5-cis-ext; "
            "scripts/export_accentor_uk.py --out models/silero-v5-cis-ext ausführen",
            file=sys.stderr,
        )
    (OUT / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    golden = OUT / "golden"
    write_safetensors(golden / "uk-0.safetensors", stages)
    (golden / "text.jsonl").write_text(
        json.dumps({"line": 0, "text": TEXT, "voice": VOICE}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {OUT}: {len(state)} tensors, {len(window)} window values, "
        f"{len(stages['audio'])} samples",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
