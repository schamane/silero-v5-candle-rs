# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Convert a Wiktionary-derived stress list to the crate's `stress_dict.tsv`.

Input: one `word<TAB>position` per line, position 1-based; `.gz` is accepted.
Output: the same columns, sorted by word, one reading per word. The TTS accentor
uses the position only for words it does not already find in `exceptions.tsv`.

Source used by the 60-sentence benchmark:
https://github.com/eugenysilin/russian-ukrainian-stress-data (CC BY-SA 4.0)
"""

import argparse
import gzip
import sys
from pathlib import Path

VOWELS = set("аеёиоуыэюя")


def position(word: str, raw: str, line_no: int) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise SystemExit(f"{line_no}: ungültige Position {raw!r}") from error
    if value < 1 or value > len(word):
        raise SystemExit(f"{line_no}: Position {value} außerhalb von {word!r}")
    if word[value - 1] not in VOWELS:
        raise SystemExit(f"{line_no}: Position {value} zeigt in {word!r} auf {word[value - 1]!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="input .tsv oder .tsv.gz")
    parser.add_argument("output", type=Path, help="output stress_dict.tsv")
    args = parser.parse_args()

    opener = gzip.open if args.source.suffix == ".gz" else open
    entries: dict[str, int] = {}
    with opener(args.source, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            try:
                word, raw = line.split("\t")
            except ValueError as error:
                raise SystemExit(f"{line_no}: erwartet word<TAB>position") from error
            word = word.strip().lower()
            value = position(word, raw.strip(), line_no)
            old = entries.get(word)
            if old is not None and old != value:
                raise SystemExit(f"{line_no}: {word!r} hat {old} und {value}")
            entries[word] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for word in sorted(entries):
            handle.write(f"{word}\t{entries[word]}\n")
    print(f"{len(entries)} Einträge -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
