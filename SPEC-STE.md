# Functional specification of `silero-v5-candle-rs`

Read this when you change the Silero v5 pipeline, the golden tests or a finding of this crate.

## 1 Scope

`silero-v5-candle-rs` turns one plain-text chunk into 48 kHz mono PCM. It is a Rust port of Silero v5 on `candle`, CPU only. Three model families are supported: `v5_ru` (Russian, with homograph BERT and accentor), `v5_cis_base_nostress` (60 CIS voices with the Ukrainian n-gram accentor for `ukr`) and `v5_cis_ext` (Ukrainian and other CIS voices; without an exported accentor the caller sets `+`, with one the accentor runs for `ukr_*` voices). The crate owns text preparation (`prepare_text_input`, `clean_star_text`, `split_chunks`), the homograph BERT, both n-gram accentors, the FastPitch acoustic model, the ConvNeXt vocoder with ISTFT head, the `silero` CLI binary, the `read_text` example and the golden tests.

## 2 Data

Model files live under `models/` (repository root, not in git). Each export script checks the package SHA-256, writes f32 safetensors and a golden set. The weights stay local and keep their source package's license: CC BY-NC-SA 4.0 for `v5_ru`/`v5_cis_ext`, MIT for `v5_cis_base_nostress` and the Ukrainian accentor.

`models/silero-v5-ru/` is written by `scripts/export_silero.py` (SHA-256 `7ba04d42…`):

- `config.json`: 47 symbols, 5 speakers, FFT geometry, BERT/tokenizer config.
- `acoustic.safetensors`, `istft_window.safetensors`.
- Accentor: `accentor.safetensors`, `ngrams.txt`, `exceptions.tsv`.
- Homograph: `homosolver.safetensors`, `homodict.json`, `bert_vocab.txt`.
- `golden/`: per-sentence tensors, `text.jsonl` (34 sentences, ids 0–33), `chunks.json`.

`models/silero-v5-cis-ext/` is written by `scripts/export_silero_cis.py` (SHA-256 `e8185323…`):

- `config.json`: 85 symbols, 35 speakers with IDs 0–35 (one gap), `speaker_rows` 36, no BERT; an optional `accentor.lang` `"ukr"`.
- `acoustic.safetensors`, `istft_window.safetensors`; optionally `accentor.safetensors`, `ngrams.txt`, `accentor.json`.
- `golden/uk-0.safetensors`: one Ukrainian sentence with `ukr_mykyta`.

`scripts/export_silero_cis.py` copies `accentor.safetensors` and `ngrams.txt` from `models/silero-v5-cis-base-nostress/` when they exist and adds the `accentor` block. `scripts/export_accentor_uk.py --out models/silero-v5-cis-ext` can also write them directly and updates an existing `config.json`; both orders work.

`models/silero-v5-cis-base-nostress/` is written by `scripts/export_silero_cis_nostress.py` and `scripts/export_accentor_uk.py` (accentor from `silero-stress`, MIT):

- `config.json`: 85 symbols, 60 speakers, `speaker_rows` 60, `accentor.lang` `"ukr"`, no BERT and no tokenizer.
- `acoustic.safetensors`, `istft_window.safetensors`, `accentor.safetensors`, `ngrams.txt`, `accentor.json` metadata.
- `golden/`: `text.jsonl` (2 sentences), `uk-0.safetensors` and `uk-1.safetensors`, plus `accentor.json` and `accentor.safetensors` for the accentor tokenizer and logits.

`models/silero-v5_6-ru/` (290 MB incl. goldens) holds the export of the newer v5_6_ru package, written by `scripts/export_silero_v5_6.py` after its SHA-256 check (`04cc9b1d…`, 2026-10-30). The crate cannot use it yet: the predictors gained type/focus embeddings, the accentor exception list grew to 19,660 entries, and the current forward ignores the per-sentence type conditioning.

## 3 Pipeline and API

`Silero::synthesize(text, voice) -> Result<Vec<f32>, Error>` calls `run_stages`. `debug_stages` calls the same helper and returns every intermediate tensor for the golden tests. `Silero::synthesize_text` splits the text with `split_chunks` using `CHUNK_LIMIT` (400) and joins the chunks without a pause. `synthesize` handles one chunk. `Silero::has_accentor` reports whether the loaded package sets stress itself; `Silero::has_accentor_for(voice)` adds the voice-language check; `Silero::voices` returns the sorted speaker names from `config.json`. The steps run in this fixed order:

1. Run `prepare_text_input`. Lower-case the text. Map `—`/`–` to whichever dash the package has, `‑` to `-`, and typographic apostrophes to `'` when the package has it. Keep every package symbol except `_`, `~`, `|`, plus `^`. Collapse whitespace and star syntax.
2. Return the error `text is empty` if no alphabetic character of the package alphabet remains after step 1 (A-142). The chunk filter `has_cyrillic` counts only а–я and ё, as before.
3. For `v5_ru`, run `HomoSolver`. Find dictionary words. Run BERT in a batch. Apply sigmoid and round. Insert `+` and `ё`.
4. Set stress. For `v5_ru`, run the accentor: n-gram embedding, stress and yo heads, exception list, single-vowel rule. For `v5_cis_base_nostress` and `v5_cis_ext` (present `accentor` block), run the Ukrainian accentor only when the voice language matches `accentor.lang` (the prefix before the first `_`, e.g. `ukr`). It splits on runs of whitespace and `.,!?;:<>=()/\\`, splits each token on `-` (keeping the hyphen on every part except the last), lowercases, and feeds the token to the n-gram embedding and the 64-128-64-10 MLP. `argmax` is the ordinal of the stressed vowel; the mark is inserted only if `softmax[argmax] > 0.5`, except that one-vowel words follow `stress_single_vowel` (default `true`). Existing `+` marks are kept unchanged. The Python reference builds `re_cond` from a *list*, so its alphabet filter effectively matches nothing; the Rust tokenizer therefore keeps apostrophes, hyphens and non-Ukrainian letters in the model input. Without an `accentor` block in `v5_cis_ext`, skip this stage: the caller supplies `+`.
5. Create symbol IDs in the form `|<text>~`. Read the speaker ID from `config.json`.
6. Run the acoustic model. Run the duration and pitch predictors. Run the encoder with speaker and pitch conditioning. Run LengthRegulator. Run the HourGlass decoder. Run `lin`.
7. Run the vocoder. Run the ConvNeXt backbone. Run the head (`exp`, cos/sin). Run the ISTFT with the exported window.

`split_chunks(text, 400)` (`src/text/normalize.rs:109`):

1. Split after `.`, `!`, `?` or `…` before whitespace, and at runs of `\n`. Trim each sentence before the cut loop, and trim the final tail.
2. If a sentence is longer than the limit, find the last `, `, `; ` and ` — ` that ends within the limit. Use the largest of these three positions as the cut. If the cut starts before one third of the limit (integer division), cut at the last space that ends within the limit instead. If that space also starts before one third, cut at the limit. The head is everything before the cut. Trim the head. Drop an empty head. The rest starts at the cut. Remove leading spaces, `,` and `;` from the rest. A ` — ` cut keeps its `—` at the start of the rest.
3. Limits below 3 are raised to 3, so the loop always advances.

ISTFT contract (`src/vocoder/istft.rs`): each frame writes `n_fft` samples, multiplies them with a window of length `win_length` and divides the overlap by the window envelope. `validate_fft`, called from `Head::load` (`istft.rs:60`), requires even `n_fft > 0`, `win > 0`, `hop > 0`, `hop <= win` and `n_fft == win_length`. Zero frames give an empty vector. `realfft` errors are returned, not asserted.

Config (`src/config.rs`): `MAX_POSITIONS = 5000` is the only fixed architecture constant; symbol and speaker counts come from the package. `Config::load` derives the symbol row count from `max(symbol_to_id) + 1` and rejects gaps or duplicates, derives the speaker rows from `speaker_rows` or `max(speakers) + 1`, and rejects rows that do not cover a speaker ID or duplicate speaker IDs. `bert` and `tokenizer` must be present together, an `accentor` block must have `lang == "ukr"`, `sample_rate` must be 48 kHz and `max_positions` 5000. `Silero::sample_rate()` and the WAV writer in `src/bin/silero.rs` use the validated rate.

## 4 Rules

- **R-TTS-30** The pipeline order is fixed: normalize; for `v5_ru` homograph then accentor, for `v5_cis_base_nostress` the Ukrainian accentor; symbol IDs, acoustic model, vocoder, ISTFT. `v5_cis_ext` runs the Ukrainian accentor when the export contains one and the voice language is `ukr`; otherwise it skips the stress stage. (`src/lib.rs`)
- **R-TTS-31** The output rate is always 48 kHz. (`src/lib.rs`)
- **R-TTS-32** Plain text only. No SSML pauses, no `speaker='random'`.
- **R-TTS-33** CPU only. (`src/lib.rs`)
- **R-TTS-34** `Config::load` fails when symbol or speaker IDs have gaps or duplicates, when `speaker_rows` does not cover a speaker ID, when only one of `bert`/`tokenizer` is present, when an `accentor` block has `lang != "ukr"`, when `accentor` and `bert` are both present, when the accentor package lacks `+` in its symbol table, when `ngrams.txt` lacks the `UNK` row, when `sample_rate != 48_000` or when `max_positions != 5000`. (`src/config.rs`, `src/text/ngram.rs`)
- **R-TTS-35** Characters outside the allowed set are dropped (`src/text/normalize.rs`). The allowed set is every package symbol except `_`, `~` and `|`, plus `^`. A symbol without an ID or an unknown voice fails the chunk, with the German messages `unbekanntes Symbol` and `unbekannte Stimme`. (`src/config.rs`)
- **R-TTS-36** Golden tests compare token IDs exactly, tensors per stage within 1e-4 to 5e-3 with matching shapes and audio within 5e-3 (`tests/golden.rs`, `tests/golden_cis.rs`, `tests/golden_cis_nostress.rs`). The nostress test uses scaled tolerances for the ConvNeXt and magnitude intermediates: a 1e-5 mel difference amplifies to ~5e-3 there, while the same Rust backbone on Python's mel differs by 1e-4 and the audio stays below 7e-4. The accentor test compares logits with a relative term and accented strings exactly. They run only in a release build with the model present. Sentence 33 batches a one-token and a two-token homograph and so exercises padding. (`tests/golden.rs`)

## 5 Tests and verification

- `cargo test` runs the unit tests. Golden tests skip in debug builds and without the model, with a visible `SKIP:` line on stderr (A-60).
- `cargo test --release --test golden` compares all 34 sentences (`baya` in every sentence; ids 0–4 also with kseniya, xenia, aidar and eugene, `tests/golden.rs:252-256`) plus `HomoSolver` logits.
- `cargo test --release --test golden_cis` compares one Ukrainian sentence with `ukr_mykyta`: `sequence` and `dur` exact, tensors and audio within the same tolerances. The sentence carries explicit `+` marks; the accentor (when bundled) must leave them unchanged. The Ukrainian accentor is gated by the voice prefix, so `kaz_*`/`tat_*` in the same package are untouched.
- `cargo test --release --test golden_cis_nostress` compares 2 sentences of `v5_cis_base_nostress` with the Python accentor: `prepared`, `accented`, `sequence` and `dur` exact. The acoustic stages use the same tolerances; the ConvNeXt output uses `5e-3 + 1e-3·|expected|` and the magnitude `5e-3 + 5e-3·|expected|` because a 1e-5 mel difference amplifies there. `cos`/`sin` use 2.5e-2 (phase of low-magnitude bins). Audio stays within 5e-3; the measured maximum is 6.5e-4.
- `cargo test --release --test golden_accentor_uk` compares the Ukrainian accentor logits (relative tolerance `1e-4 + 1e-5·|expected|`), softmax probabilities (1e-5) and accented sentences (exact) against `golden/accentor.json`.
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
| A-143 | medium | accentor | `src/text/accentor_uk.rs` | `silero-stress` builds `re_cond` from a `sorted(set(...))` list; the resulting regex needs a literal `]` after the character and removes nothing from ordinary words. An alphabet-filtered Rust tokenizer produced different stress for `дев'ять` and `п'ять`. | FIXED: keep apostrophes, hyphens and non-Ukrainian letters in the model input; golden covers both words |
| A-144 | low | golden tests | `tests/golden_cis_nostress.rs` | A 1e-5 mel difference amplifies to ~5e-3 in the ConvNeXt output and ~5e-2 in `mag`; fixed absolute tolerances failed on the second sentence even though audio stayed at 6.5e-4. | FIXED: scaled tolerances for `backbone`/`mag`; `cos`/`sin` 2.5e-2; audio 5e-3 |
| A-145 | medium | pipeline | `src/lib.rs`, `src/config.rs` | A bundled Ukrainian accentor would run for every voice in the multilingual CIS packages (Kazakh, Tatar, …) and for ext voices without a built-in accentor it did nothing at all. | FIXED: `accentor_matches` gates by voice prefix; `v5_cis_ext` can bundle the accentor and `has_accentor_for(voice)` reports it |
