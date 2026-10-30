#!/usr/bin/env python3
"""Liest einen kurzen ukrainischen Text mit Silero v5_cis_ext vor.

v5_cis_ext ist das ukrainische (und turksprachige) Schwestermodell von
v5_ru. Es hat keinen Akzentor: Stress muss im Text als `+` vor dem
betonten Vokal stehen, z. B. `Прив+іт`. Ohne Stressmarks spricht das
Modell trotzdem, betont aber nach eigenem Ermessen.

Beispiele:
    uv run --with torch scripts/ukrainian_demo.py
    uv run --with torch scripts/ukrainian_demo.py --voice ukr_tetiana
    uv run --with torch scripts/ukrainian_demo.py --text "Прив+іт, як спр+ави?" --no-play
    uv run --with torch --with silero-stress scripts/ukrainian_demo.py --text "Привіт, як справи?"

Das Modell wird bei Bedarf von models.silero.ai geladen (88 MB) und unter
~/.cache/silero/ abgelegt. Es gilt die Silero-Lizenz (CC BY-NC).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path

MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_cis_ext.pt"
MODEL_SHA256 = "e8185323fc5a341c229f8a8c9928512f665bf2736021bc600512499d0506eed2"

# Kurzer, bereits korrekt betonter Testtext (offizielles Silero-Beispiel).
DEFAULT_TEXT = "+Я з р+аннього дит+инства д+уже любл+ю сл+ухати цік+аві к+азки."

UKRAINIAN_VOICES = (
    "ukr_kateryna",
    "ukr_lada",
    "ukr_mykyta",
    "ukr_oleksa",
    "ukr_tetiana",
)


def cache_path() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "silero" / "v5_cis_ext.pt"


def find_model(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("SILERO_CIS_EXT")
    if env:
        return Path(env).expanduser()
    cached = cache_path()
    if cached.is_file():
        return cached
    # Lokale Arbeitskopie aus früheren Export-Experimenten.
    if Path("/tmp/v5_cis_ext.pt").is_file():
        return Path("/tmp/v5_cis_ext.pt")
    return cached


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"lade {url} -> {target}", file=sys.stderr)
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_report = 0
        with partial.open("wb") as handle:
            while block := response.read(1 << 20):
                handle.write(block)
                done += len(block)
                percent = int(done * 100 / total) if total else 0
                if percent >= next_report:
                    print(f"  {percent:3d} %  {done / 1e6:6.1f} MB", file=sys.stderr)
                    next_report = percent + 10
    partial.replace(target)


def ensure_model(explicit: str | None) -> Path:
    path = find_model(explicit)
    if not path.is_file():
        if explicit:
            raise SystemExit(f"Modelldatei fehlt: {path}")
        download(MODEL_URL, path)
    actual = sha256(path)
    if actual != MODEL_SHA256:
        raise SystemExit(
            f"SHA-256 passt nicht: {path}\n  erwartet {MODEL_SHA256}\n  erhalten {actual}"
        )
    return path


def load_package(path: Path):
    try:
        import torch  # noqa: F401  (nur zur klaren Fehlermeldung importiert)
        from torch.package import PackageImporter
    except ImportError as exc:  # pragma: no cover - Umgebungsfehler
        raise SystemExit(
            "torch fehlt. Aufruf z. B.: uv run --with torch scripts/ukrainian_demo.py"
        ) from exc
    package = PackageImporter(str(path)).load_pickle("tts_models", "model")
    return package


def auto_stress(text: str) -> str:
    """Setzt ukrainische Stressmarks, falls silero-stress installiert ist."""
    try:
        from silero_stress import load_accentor
    except ImportError:
        print(
            "Hinweis: kein '+' im Text und silero-stress nicht installiert.\n"
            "         Betonung bleibt dem Modell überlassen. Für exakte Betonung\n"
            "         '+vor dem betonten Vokal' setzen oder\n"
            "         'uv run --with torch --with silero-stress ...' verwenden.",
            file=sys.stderr,
        )
        return text
    print("akzentuiere Text mit silero-stress ...", file=sys.stderr)
    return load_accentor(lang="ukr")(text)


def write_wav(path: Path, samples, sample_rate: int) -> None:
    import torch

    if not torch.is_tensor(samples):
        samples = torch.as_tensor(samples)
    pcm = samples.detach().to("cpu", torch.float32).clamp(-1.0, 1.0)
    pcm = (pcm * 32767.0).round().to(torch.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.numpy().tobytes())


def play(path: Path) -> None:
    player = shutil.which("afplay") or shutil.which("aplay") or shutil.which("ffplay")
    if not player:
        print(f"kein Audioplayer gefunden; WAV liegt unter {path}", file=sys.stderr)
        return
    if player.endswith("ffplay"):
        command = [player, "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
    else:
        command = [player, str(path)]
    print(f"spiele {path} mit {Path(player).name} ...", file=sys.stderr)
    subprocess.run(command, check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ukrainischen Text mit Silero v5_cis_ext vorlesen.",
        epilog=f"Ukrainische Stimmen: {', '.join(UKRAINIAN_VOICES)}",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text; '+' markiert die Betonung")
    parser.add_argument("--voice", default="ukr_mykyta", help="Sprecher (Standard: ukr_mykyta)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ukrainian_demo.wav"),
        help="Ziel-WAV (Standard: ukrainian_demo.wav)",
    )
    parser.add_argument("--model", help="Pfad zur v5_cis_ext.pt (Standard: Cache, sonst Download)")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        choices=(8000, 24000, 48000),
        help="Ausgaberate in Hz (Standard: 48000)",
    )
    parser.add_argument(
        "--no-auto-stress",
        action="store_true",
        help="silero-stress nicht automatisch verwenden",
    )
    parser.add_argument("--no-play", action="store_true", help="WAV nur schreiben, nicht abspielen")
    parser.add_argument("--list-voices", action="store_true", help="ukrainische Stimmen ausgeben")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_voices:
        print("\n".join(UKRAINIAN_VOICES))
        return 0

    model_path = ensure_model(args.model)
    model = load_package(model_path)

    if args.voice not in model.speakers:
        raise SystemExit(
            f"unbekannte Stimme {args.voice!r}; verfügbar: {', '.join(model.speakers)}"
        )

    text = args.text
    if "+" not in text and not args.no_auto_stress:
        text = auto_stress(text)
    print(f"Text: {text}", file=sys.stderr)

    samples = model.apply_tts(
        text=text,
        speaker=args.voice,
        sample_rate=args.sample_rate,
    )
    write_wav(args.out, samples, args.sample_rate)
    seconds = len(samples) / args.sample_rate
    print(f"{args.out} geschrieben ({seconds:.2f} s, {args.sample_rate} Hz)", file=sys.stderr)
    if not args.no_play:
        play(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
