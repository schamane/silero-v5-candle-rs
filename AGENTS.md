# AGENTS.md

Rust port of Silero TTS v5 on candle (`silero-v5-candle-rs`). Applies to every coding agent.

## Source of truth

- `SPEC-STE.md` is the authority. When it conflicts with code, README or any other doc, the SPEC wins. Fix the code, or change the SPEC deliberately and say why in the commit message.
- Do not copy SPEC content into this file. Point to the rule ID (`R-TTS-xx`) instead.

## Keep docs in sync

Update `SPEC-STE.md` in the same change whenever you alter any of: public API, CLI flags, Cargo features, env vars, golden tolerances, tests, Make targets, model file layout. Update `README.md` for anything a user sees. Run `make check-docs` before finishing.

## Commands

- `make lint-rust`: clippy on three feature sets with `-D warnings`. This is the gate.
- `make test`: release tests. Goldens run only in release builds and need models in `models/`.
- `make check`: lint, Linux cross-check, release tests on Metal, CPU and Accelerate.
- `make check-docs`: verify that commands, flags, tests and symbols named in the docs exist.
- `SILERO_REQUIRE_GOLDEN=1`: fail instead of skipping when a golden or model is missing.
- `SILERO_DEVICE=auto|cpu|metal`: device choice for `Silero::load` and the CLI (R-TTS-33).

## Hard rules

- No MSRV bump (`rust-version` in `Cargo.toml`) and no public API change without explicit user approval.
- Every commit is authored and committed as `Nazar Kulyk <nazar.kulyk@ibsolution.de>` (the GitHub `nazarkulyk` account); no `Co-Authored-By` trailers, no other identities, no contributor credits for AI tools. Rewriting history to enforce this is allowed and force-pushing the result to `main` is expected.
- `has_cyrillic` stays а–я/ё; `has_text` uses `is_alphabetic` (A-142).
- Output is always 48 kHz. Plain text only, no SSML (R-TTS-31, R-TTS-32).
- `accelerate` and `mkl` features are mutually exclusive.
- `debug_stages`, `load_reference*` and related items exist only under feature `test-util`; they are not stable API.
- Do not generate or commit `stress_dict.tsv` (CC BY-SA 4.0 data, gitignored) or model weights.
- Never assume benchmark numbers: measure with `benches/sbench.rs` (`cargo bench --bench sbench -- run cpu short`) and record raw data in `docs/messungen-rohdaten.json`.

## Layout

See SPEC §2. Briefly: `src/` library and CLI, `tests/` golden and backend tests, `scripts/` Python exporters, `models/` model packages, `listening/` demo audio, `benches/` measurement program, `docs/` measurements.
