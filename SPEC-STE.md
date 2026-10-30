# Functional specification of `silero-v5-ru-candle-rs`

Read this when you change the Silero v5_ru pipeline, the golden tests or a finding of this crate. The chat-harness side (TTS worker, playback, threads) is not part of this repository.

## 1 Scope

`silero-v5-ru-candle-rs` turns one Russian plain-text chunk into 48 kHz mono PCM. It is a Rust port of Silero v5_ru on `candle`, CPU only. The crate owns text preparation (`prepare_text_input`, `clean_star_text`, `split_chunks`), the homograph BERT, the accentor, the FastPitch acoustic model, the ConvNeXt vocoder with ISTFT head, the `silero` example binary and the golden tests. The consuming harness owns the TTS worker, ring buffer, playback and pause/stop.

## 2 Data

Model files live in `models/silero-v5-ru/` (repository root, not in git). `scripts/export_silero.py` writes them as f32 safetensors after it checks the package SHA-256. The weights derive from the CC BY-NC-SA 4.0 package and stay local.

- `config.json`: 47 symbols, 5 speakers, FFT geometry, BERT/tokenizer config.
- `acoustic.safetensors`, `istft_window.safetensors`.
- Accentor: `accentor.safetensors`, `ngrams.txt`, `exceptions.tsv`.
- Homograph: `homosolver.safetensors`, `homodict.json`, `bert_vocab.txt`.
- `golden/`: per-sentence tensors, `text.jsonl` (34 sentences, ids 0–33), `chunks.json`.

## 3 Pipeline and API

`Silero::synthesize(text, voice) -> Result<Vec<f32>, String>` (`src/lib.rs:92`) calls `run_stages`. `debug_stages` calls the same helper and returns every intermediate tensor for the golden tests. The steps run in this fixed order:

1. `prepare_text_input`: lower case, map `—` and `‑`, keep `symbols[3:]` plus `^`, collapse whitespace and star syntax.
2. Fail with `Text is empty` when no Cyrillic letter (а–я, ё) survives step 1 (`src/lib.rs:100`). Punctuation, digits and `-` alone do not count (A-142).
3. Homograph solver: find dictionary words, batch BERT, sigmoid and round, insert `+` and `ё`.
4. Accentor: n-gram embedding, stress and yo heads, exceptions, single-vowel rule.
5. Symbol IDs in the form `|<text>~`; speaker ID from `config.json`.
6. Acoustic model: duration and pitch predictors, encoder with speaker and pitch conditioning, LengthRegulator, HourGlass decoder, `lin`.
7. Vocoder: ConvNeXt backbone, head (`exp`, cos/sin), ISTFT with the exported window.

`split_chunks(text, 400)` (`src/text/normalize.rs:112`):

1. Split after `.`, `!`, `?` or `…` before whitespace, and at runs of `\n`.
2. A sentence longer than the limit is cut at the last `, `, `; ` or ` — `, if that cut lies at or past one third of the limit. Otherwise cut at the last space, if that cut lies at or past one third. Otherwise cut at the limit.
3. Limits below 3 are raised to 3, so the loop always advances.

ISTFT contract (`src/vocoder/istft.rs`): each frame writes `n_fft` samples, multiplies them with a window of length `win_length` and divides the overlap by the window envelope. `validate_fft`, called from `Head::load` (`istft.rs:44`), requires even `n_fft > 0`, `win > 0`, `hop > 0`, `hop <= win` and `n_fft == win_length`. Zero frames give an empty vector. `realfft` errors are returned, not asserted.

Config (`src/config.rs:60`): `SYMBOL_COUNT = 47`, `SPEAKER_COUNT = 5`, `MAX_POSITIONS = 5000`. `Config::load` checks symbol IDs 0..46, speaker IDs 0..4, `sample_rate == 48_000` and `max_positions == 5000`. `Silero::sample_rate()` and the WAV writer in `src/bin/silero.rs` use the validated rate.

## 4 Rules

- **R-TTS-30** The pipeline order is fixed: normalise, homograph, accent, symbol IDs, acoustic model, vocoder, ISTFT. (`src/lib.rs:96-138`)
- **R-TTS-31** The output rate is always 48 kHz. (`src/lib.rs:23`)
- **R-TTS-32** Plain text only. No SSML pauses, no `speaker='random'`.
- **R-TTS-33** CPU only. (`src/lib.rs:38`)
- **R-TTS-34** `Config::load` fails unless `config.json` has 47 symbols with IDs 0..46, 5 speakers with IDs 0..4, `sample_rate` 48 kHz and `max_positions` 5000. (`src/config.rs:60-125`)
- **R-TTS-35** An unknown character or voice fails the chunk (`unbekanntes Symbol`, `unbekannte Stimme`). (`src/config.rs:144`, `src/config.rs:151`)
- **R-TTS-36** Golden tests compare token IDs exactly, tensors per stage within 1e-4 to 5e-3 (`tests/golden.rs:104-248`) and audio within 5e-3 (`tests/golden.rs:256`). They run only in a release build with the model present. Sentence 33 batches a one-token and a two-token homograph and so exercises padding. (`tests/golden.rs:148-260`)

## 5 Tests and verification

- `cargo test` runs the unit tests. Golden tests skip in debug builds and without the model, with a visible `SKIP:` line on stderr (A-60).
- `cargo test --release --test golden` compares all 34 sentences (`baya` in every sentence; ids 0–4 also with kseniya, xenia, aidar and eugene, `tests/golden.rs:213-217`) plus homosolver logits.
- `cargo clippy --all-targets -- -D warnings`.
- Sentence 33 (`Цвета атласа на солнце кажутся ярче.`, golden id 33) measures logits 3.8e-6 and audio 3.9e-5 against Python.
- The mask in `HomoSolver::forward` stays `None`. The reference pads identically and passes no mask. Do not add one without regenerating the goldens and comparing both paths.

## 6 Finding register

Finding IDs use the prefix `A-` to keep them distinct from the consuming harness register's `F-` numbers. Status: `FIXED (<ref>)` or `OPEN`. Commit references point at the private predecessor repository; every `FIXED` fix is part of the initial commit of this repository. `uncommitted` rows are in the working tree.

| ID | sev | area | file | finding | status |
|---|---|---|---|---|---|
| A-60 | low | tests | `tests/golden.rs` | Golden tests passed silently in debug builds and without the model. | FIXED (eb71244): visible `SKIP:` line |
| A-132 | medium | ISTFT | `src/vocoder/istft.rs` | `n_fft != win_length` panicked or left a wrong tail; `hop > win` and zero frames underflowed; `expect` hid `realfft` errors. | FIXED (eb71244): `validate_fft`, empty output for zero frames, errors propagated |
| A-133 | medium | accentor | `src/text/accentor.rs` | Missing or malformed tensors panicked despite `Result`; MLP lengths unchecked, `zip` truncated silently. | FIXED (eb71244): `tensor_vec` returns `Err`, `Linear::load` checks lengths |
| A-134 | medium | golden tests | `tests/golden.rs` | Missing homosolver logits skipped the comparison and stayed green. | FIXED (eb71244): `Some(logits)` required |
| A-135 | medium | golden tests | `tests/corpus.txt` | Padding path (unequal BERT token counts, no mask) was untested. | FIXED (eb71244): sentence 33 added |
| A-136 | low | homosolver | `src/text/homosolver.rs` | `unwrap_or_default()` turned tensor errors into an empty predictor list. | FIXED (eb71244): `values` returns `Result` |
| A-137 | low | normalize | `src/text/normalize.rs` | A limit below 3 gave `cut = 0`; the loop never advanced. | FIXED (eb71244): `limit.max(3)` |
| A-138 | low | config | `src/config.rs`, `src/lib.rs`, `src/bin/silero.rs` | Only the symbol count was validated; other IDs, rate and WAV metadata were hardcoded or unchecked. | FIXED (eb71244): shared constants, full validation |
| A-139 | low | pipeline | `src/lib.rs` | `synthesize` and `debug_stages` repeated the same step sequence. | FIXED (eb71244): shared `run_stages` |
| A-140 | low | examples | harness example binaries (not part of this repository) | `--models` only worked for a directory named `models`; query errors looked like rank 0; missing chunk key panicked; malformed input passed silently. | FIXED (eb71244) |
| A-141 | low | dead code | `src/acoustic/transformer.rs`, `src/config.rs` | Unused helper and unused config fields. | FIXED (eb71244): removed |
| A-142 | low | text | `src/text/normalize.rs:45-47` | Input of only `-` passed `Text is empty`: `has_text` checked the clean set, which keeps `-`. | FIXED: `has_text` requires а–я or ё; test `hyphen_only_is_empty_text` |
