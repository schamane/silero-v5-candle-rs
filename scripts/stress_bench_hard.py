# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Evaluate explicit targets from the hard-case stress suite.

Input cases are JSON objects with `text`, `target`, and `gold` (plus optional
`accept` alternatives). The accented lines come from `examples/accent_suite.rs`.
Unlike the 60-sentence baseline, this script does not score every word against
the Wiktionary dictionary; it checks the one researched hard target per
sentence, including context-dependent homographs.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

EDGE = re.compile(r"^[^\wа-яёА-ЯЁ+]+|[^\wа-яёА-ЯЁ+]+$")


def form(token: str) -> str:
    return EDGE.sub("", token).lower()


def bare(token: str) -> str:
    return form(token).replace("+", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("accented", type=Path)
    parser.add_argument("output", type=Path, nargs="?", default=None)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text())
    lines = args.accented.read_text().splitlines()
    if len(cases) != len(lines):
        raise SystemExit(f"{len(cases)} Fälle, aber {len(lines)} Ausgabezeilen")

    errors = []
    passed = 0
    categories = Counter()
    for case, line in zip(cases, lines):
        categories[case["category"]] += 1
        target = case["target"].lower()
        expected = [value.lower() for value in [case["gold"], *case.get("accept", [])]]
        tokens = [form(token) for token in line.split()]
        bare_tokens = [bare(token) for token in line.split()]
        if target not in bare_tokens:
            errors.append({"id": case["id"], "target": target, "reason": "target missing", "line": line})
            continue
        if any(value in tokens for value in expected):
            passed += 1
        else:
            errors.append({
                "id": case["id"],
                "target": target,
                "expected": expected,
                "got": [token for token in tokens if target in bare(token) or bare(token) in target],
                "line": line,
            })

    result = {
        "cases": len(cases),
        "passed": passed,
        "accuracy": passed / len(cases) if cases else None,
        "categories": dict(sorted(categories.items())),
        "errors": errors,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
