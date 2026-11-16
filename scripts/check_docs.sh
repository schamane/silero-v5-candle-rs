#!/bin/sh
# Checks that things named in the docs still exist. Catches stale references,
# not wrong prose or wrong numbers.
set -u
cd "$(dirname "$0")/.."
DOCS="SPEC-STE.md README.md AGENTS.md"
fail=0
miss() { echo "MISSING: $1 (named in docs)"; fail=1; }

# Words that look like symbols but are data names, history or std items.
IGNORE=' catch_unwind ext_alph ukr_roman unwrap_or_default max_abs_diff_metal_vs_cpu metal_vs_cpu a '
skip() { case "$IGNORE" in *" $1 "*) return 0;; esac; return 1; }

# make targets
for t in $(rg -oN --no-filename 'make [a-z][a-z-]*' $DOCS | awk '{print $2}' | sort -u); do
  skip "$t" || rg -q "^$t:" Makefile || miss "make target $t"
done
# integration tests: --test NAME and tests/NAME.rs
for t in $(rg -oN --no-filename -e '--test [a-z_]+' $DOCS | awk '{print $2}' | sort -u); do
  [ -f "tests/$t.rs" ] || miss "test $t"
done
for f in $(rg -oN --no-filename 'tests/[a-z_]+\.rs' $DOCS | sort -u); do
  [ -f "$f" ] || miss "$f"
done
# scripts and src files
for f in $(rg -oN --no-filename '(scripts|src|examples|benches)/[A-Za-z0-9_/]+\.(py|rs|sh)' $DOCS | sort -u); do
  [ -f "$f" ] || miss "$f"
done
# CLI flags: --long options must appear in a source file or Makefile
for f in $(rg -oN --no-filename -e '`--[a-z][a-z-]+' $DOCS | tr -d '`' | sort -u); do
  case "$f" in --models|--release|--test|--features|--no-default-features|--all-features|--locked|--workspace|--all-targets|--lib|--target|--help|--example) continue;; esac
  rg -q -F -e "$f" src examples benches Makefile Cargo.toml || miss "flag $f"
done
# backticked CamelCase identifiers and snake_case functions
for s in $(rg -oN --no-filename '`(Silero::)?[A-Za-z_]+(\(\))?`' $DOCS | tr -d '`' | sed 's/^Silero:://; s/()$//' | rg '^([A-Z][a-z]+[A-Za-z]+|[a-z]+_[a-z_]+)$' | sort -u); do
  skip "$s" || rg -q -w "$s" src tests examples benches scripts Cargo.toml Makefile || miss "symbol $s"
done
# Cargo features
for f in $(rg -oN --no-filename 'features? `?[a-z,-]+' $DOCS | rg -o '[a-z][a-z-]+' | rg -v '^features?$' | sort -u); do
  rg -q "^$f *=" Cargo.toml || [ "$f" = default ] || true
done
[ $fail -eq 0 ] && echo "docs OK"
exit $fail
