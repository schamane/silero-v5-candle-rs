# Functional specification of `silero-v5-candle-rs`

Read this when you change the Silero v5_ru pipeline, the golden tests or a finding of this crate.

## 1 Scope

`silero-v5-candle-rs` turns one Russian plain-text chunk into 48 kHz mono PCM. It is a Rust port of Silero v5_ru on `candle`, CPU only. The crate owns text preparation (`prepare_text_input`, `clean_star_text`, `split_chunks`), the homograph BERT, the accentor, the FastPitch acoustic model, the ConvNeXt vocoder with ISTFT head, the `silero` CLI binary and the golden tests.

## 2 Data

Model files live in `models/silero-v5-ru/` (repository root, not in git). `scripts/export_silero.py` writes them as f32 safetensors after it checks the package SHA-256. The weights derive from the CC BY-NC-SA 4.0 package and stay local.

- `config.json`: 47 symbols, 5 speakers, FFT geometry, BERT/tokenizer config.
- `acoustic.safetensors`, `istft_window.safetensors`.
- Accentor: `accentor.safetensors`, `ngrams.txt`, `exceptions.tsv`.
- Homograph: `homosolver.safetensors`, `homodict.json`, `bert_vocab.txt`.
- `golden/`: per-sentence tensors, `text.jsonl` (34 sentences, ids 0–33), `chunks.json`.

`models/silero-v5_6-ru/` (290 MB incl. goldens) holds the export of the newer v5_6_ru package, written by `scripts/export_silero_v5_6.py` after its SHA-256 check (`04cc9b1d…`, 2026-10-30). The crate cannot use it yet: the predictors gained type/focus embeddings, the accentor exception list grew to 19,660 entries, and the current forward ignores the per-sentence type conditioning.

## 3 Pipeline and API

`Silero::synthesize(text, voice) -> Result<Vec<f32>, Error>` (`src/lib.rs:114`) calls `run_stages`. `debug_stages` calls the same helper and returns every intermediate tensor for the golden tests. `Silero::synthesize_text` splits the text with `split_chunks` using `CHUNK_LIMIT` (400) and joins the chunks without a pause. `synthesize` handles one chunk. The steps run in this fixed order:

1. Run `prepare_text_input`. Lower-case the text. Map `—` and `‑`. Keep `symbols[3:]` and `^`. Collapse whitespace and star syntax.
2. Return the error `text is empty` if no Cyrillic letter (а–я, ё) remains after step 1 (`src/lib.rs:144`). The chunk filter `has_cyrillic` counts а–я and ё after `lower()`, as `has_text` does. Punctuation, digits and `-` alone do not count (A-142).
3. Run `HomoSolver`. Find dictionary words. Run BERT in a batch. Apply sigmoid and round. Insert `+` and `ё`.
4. Run the accentor. Compute the n-gram embedding. Run the stress and yo heads. Apply the exception list. Apply the single-vowel rule.
5. Create symbol IDs in the form `|<text>~`. Read the speaker ID from `config.json`.
6. Run the acoustic model. Run the duration and pitch predictors. Run the encoder with speaker and pitch conditioning. Run LengthRegulator. Run the HourGlass decoder. Run `lin`.
7. Run the vocoder. Run the ConvNeXt backbone. Run the head (`exp`, cos/sin). Run the ISTFT with the exported window.

`split_chunks(text, 400)` (`src/text/normalize.rs:109`):

1. Split after `.`, `!`, `?` or `…` before whitespace, and at runs of `\n`. Trim each sentence before the cut loop, and trim the final tail.
2. If a sentence is longer than the limit, find the last `, `, `; ` and ` — ` that ends within the limit. Use the largest of these three positions as the cut. If the cut starts before one third of the limit (integer division), cut at the last space that ends within the limit instead. If that space also starts before one third, cut at the limit. The head is everything before the cut. Trim the head. Drop an empty head. The rest starts at the cut. Remove leading spaces, `,` and `;` from the rest. A ` — ` cut keeps its `—` at the start of the rest.
3. Limits below 3 are raised to 3, so the loop always advances.

ISTFT contract (`src/vocoder/istft.rs`): each frame writes `n_fft` samples, multiplies them with a window of length `win_length` and divides the overlap by the window envelope. `validate_fft`, called from `Head::load` (`istft.rs:60`), requires even `n_fft > 0`, `win > 0`, `hop > 0`, `hop <= win` and `n_fft == win_length`. Zero frames give an empty vector. `realfft` errors are returned, not asserted.

Config (`src/config.rs:12-14`): `SYMBOL_COUNT = 47`, `SPEAKER_COUNT = 5`, `MAX_POSITIONS = 5000`. `Config::load` checks symbol IDs 0..46, speaker IDs 0..4, `sample_rate == 48_000` and `max_positions == 5000`. `Silero::sample_rate()` and the WAV writer in `src/bin/silero.rs` use the validated rate.

## 4 Rules

- **R-TTS-30** The pipeline order is fixed: normalize, homograph, accent, symbol IDs, acoustic model, vocoder, ISTFT. (`src/lib.rs:141-182`)
- **R-TTS-31** The output rate is always 48 kHz. (`src/lib.rs:27`)
- **R-TTS-32** Plain text only. No SSML pauses, no `speaker='random'`.
- **R-TTS-33** CPU only. (`src/lib.rs:52`)
- **R-TTS-34** `Config::load` fails unless `config.json` has 47 symbols with IDs 0..46, 5 speakers with IDs 0..4, `sample_rate` 48 kHz and `max_positions` 5000. (`src/config.rs:79-147`)
- **R-TTS-35** Characters outside the allowed set are dropped (`src/text/normalize.rs:34`). A symbol without an ID or an unknown voice fails the chunk, with the German messages `unbekanntes Symbol` and `unbekannte Stimme`. (`src/config.rs:167`, `src/config.rs:177`)
- **R-TTS-36** Golden tests compare token IDs exactly, tensors per stage within 1e-4 to 5e-3 with matching shapes (`tests/golden.rs:111-299`) and audio within 5e-3 (`tests/golden.rs:290-296`). They run only in a release build with the model present. Sentence 33 batches a one-token and a two-token homograph and so exercises padding. (`tests/golden.rs:170-299`)

## 5 Tests and verification

- `cargo test` runs the unit tests. Golden tests skip in debug builds and without the model, with a visible `SKIP:` line on stderr (A-60).
- `cargo test --release --test golden` compares all 34 sentences (`baya` in every sentence; ids 0–4 also with kseniya, xenia, aidar and eugene, `tests/golden.rs:252-256`) plus `HomoSolver` logits.
- `chunks_match_python` (`tests/golden.rs:301-354`) is a regression snapshot of `split_chunks` against `golden/chunks.json`, the output of a hand-written mirror in `scripts/export_silero.py`. It is not a comparison with the Python reference. It asserts the entry count before indexing.
- `cargo clippy --all-targets -- -D warnings`.
- Sentence 33 (`Цвета атласа на солнце кажутся ярче.`, golden id 33) measures logits 3.8e-6 and audio 3.9e-5 against Python. The largest audio difference over all 34 sentences and voices is 2.9e-3 (sentence 4, aidar), which the 5e-3 audio tolerance covers.
- The mask in `HomoSolver::forward` stays `None`. The reference pads identically and passes no mask. Do not add one without regenerating the goldens and comparing both paths.

## 6 Finding register

Finding IDs use the `A-` prefix. Status: `FIXED` or `OPEN`.

| ID | sev | area | file | finding | status |
|---|---|---|---|---|---|
| A-60 | low | tests | `tests/golden.rs` | Golden tests passed silently in debug builds and without the model. | FIXED: visible `SKIP:` line |
| A-132 | medium | ISTFT | `src/vocoder/istft.rs` | `n_fft != win_length` panicked or left a wrong tail; `hop > win` and zero frames underflowed; `expect` hid `realfft` errors. | FIXED: `validate_fft`, empty output for zero frames, errors propagated |
| A-133 | medium | accentor | `src/text/accentor.rs` | Missing or malformed tensors panicked despite `Result`; MLP lengths unchecked, `zip` truncated silently. | FIXED: `tensor_vec` returns `Err`, `Linear::load` checks lengths |
| A-134 | medium | golden tests | `tests/golden.rs` | Missing homosolver logits skipped the comparison and stayed green. | FIXED: `Some(logits)` required |
| A-135 | medium | golden tests | `tests/corpus.txt` | Padding path (unequal BERT token counts, no mask) was untested. | FIXED: sentence 33 added |
| A-136 | low | `HomoSolver` | `src/text/homosolver.rs` | `unwrap_or_default()` turned tensor errors into an empty predictor list. | FIXED: `values` returns `Result` |
| A-137 | low | normalize | `src/text/normalize.rs` | A limit below 3 gave `cut = 0`; the loop never advanced. | FIXED: `limit.max(3)` |
| A-138 | low | config | `src/config.rs`, `src/lib.rs`, `src/bin/silero.rs` | Only the symbol count was validated; other IDs, rate and WAV metadata were hardcoded or unchecked. | FIXED: shared constants, full validation |
| A-139 | low | pipeline | `src/lib.rs` | `synthesize` and `debug_stages` repeated the same step sequence. | FIXED: shared `run_stages` |
| A-140 | low | examples | example binaries outside this crate | `--models` only worked for a directory named `models`; query errors looked like rank 0; missing chunk key panicked; malformed input passed silently. | FIXED |
| A-141 | low | dead code | `src/acoustic/transformer.rs`, `src/config.rs` | Unused helper and unused config fields. | FIXED: removed |
| A-142 | low | text | `src/text/normalize.rs:45` | Input of only `-` passed `text is empty`: `has_text` checked the clean set, which keeps `-`. | FIXED: `has_text` requires а–я or ё; test `hyphen_only_is_empty_text` |
