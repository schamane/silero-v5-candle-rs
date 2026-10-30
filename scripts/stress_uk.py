#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["silero-stress", "torch", "numpy"]
# ///
"""Setzt ukrainische Stressmarken (`+` vor dem betonten Vokal).

`v5_cis_ext` hat keinen eigenen Akzentor; ohne `+` rät das Modell die Betonung.
Dieses Skript nutzt den ukrainischen Akzentor aus `silero-stress` und gibt den
markierten Text aus. Danach mit `read_text` synthetisieren:

    uv run scripts/stress_uk.py examples/lina.txt > /tmp/lina_stressed.txt
    target/release/examples/read_text \\
        --file /tmp/lina_stressed.txt --lang uk --out lina.wav
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ukrainische Stressmarken mit silero-stress setzen."
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Textdatei (UTF-8); ohne Angabe wird stdin gelesen",
    )
    parser.add_argument("-o", "--out", type=Path, help="Zieldatei statt stdout")
    args = parser.parse_args()

    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    if not text.strip():
        raise SystemExit("leerer Text")

    # Der Akzentor erkennt Wörter mit ASCII-Apostroph; `пір’я` mit U+2019 wird
    # sonst nicht betont. Der Rust-Frontend akzeptiert `'` ebenfalls.
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")

    from silero_stress import load_accentor

    accented = load_accentor(lang="ukr")(text)
    if args.out:
        args.out.write_text(accented, encoding="utf-8")
    else:
        sys.stdout.write(accented)
        if not accented.endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()
