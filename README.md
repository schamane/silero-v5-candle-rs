# silero-v5-candle-rs

Silero v5 text-to-speech in Rust on [candle](https://github.com/huggingface/candle).
One plain-text chunk in, 48 kHz mono f32 PCM out. CPU only. Russian (`v5_ru`),
the CIS base model (`v5_cis_base_nostress`) and `v5_cis_ext` are supported. Both
CIS packages can bundle the Ukrainian n-gram accentor; for `v5_cis_ext` it is
optional and only applied to `ukr_*` voices. The other CIS languages in those
packages are not wired up yet: they need the `ext_alph` conversion.

The crate is a port of the Silero TorchScript pipeline and owns all of it:

- text frontend: normalization, symbol mapping, chunk splitting;
- `HomoSolver` and Russian accentor (`v5_ru`): rubert-tiny2-shaped BERT plus
  n-gram stress/`ё` heads and the exception list; Ukrainian n-gram accentor for
  `v5_cis_base_nostress` and, if bundled, `v5_cis_ext`;
- acoustic model: duration/pitch predictors, encoder, LengthRegulator, HourGlass;
- vocoder: ConvNeXt backbone, ISTFT head.

Not affiliated with or endorsed by Silero.

## Requirements

- Rust 1.85+ (edition 2024)
- `candle` 0.11, CPU backend
- Model files under `models/silero-v5-ru/`, `models/silero-v5-cis-ext/` and/or
  `models/silero-v5-cis-base-nostress/` (not in git, see below)

## Usage

```rust
use silero_v5_candle::Silero;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Silero::load(std::path::Path::new("models/silero-v5-ru"))?;
    let samples: Vec<f32> = model.synthesize("Привет, как дела?", "baya")?;
    Ok(())
}
```

Long text should use `Silero::synthesize_text`, which splits it into chunks of at most 400 characters and joins the audio without a pause.

Five voices: `aidar`, `baya`, `kseniya`, `xenia`, `eugene`. Plain text only: no SSML
pauses, no `speaker="random"`.

A small CLI writes a WAV:

```sh
cargo run --release --bin silero -- --out out.wav --voice baya "Привет, как дела?"
```

`examples/read_text.rs` reads a text file:

```sh
cargo run --release --example read_text -- \
    --file examples/ukrainian.txt --lang uk --out ukrainian.wav
```

`--lang ru` uses `models/silero-v5-ru`; `--lang uk` uses
`models/silero-v5-cis-base-nostress`. `--voice` defaults to `baya` / `ukr_igor`;
`--no-play` skips playback. If the voice exists only in `v5_cis_ext` (for example
`ukr_mykyta`), `read_text` loads `models/silero-v5-cis-ext` automatically unless
`--model` is given.

### Ukrainian

`v5_cis_base_nostress` has no built-in stress model, but the crate loads the
Ukrainian `silero-stress` accentor and applies it after text preparation. Plain
Ukrainian text is enough:

```sh
cargo run --release --example read_text -- \
    --file examples/lina.txt --lang uk --out lina.wav
```

Voices: `ukr_igor`, `ukr_roman`. The accentor is imperfect (it is a per-word
n-gram model, not a language model); explicit `+` marks are preserved and take
precedence.

`v5_cis_ext` ships without an accentor, but the export can bundle the same
Ukrainian accentor (see below). If it is present, `read_text` applies it to
`ukr_*` voices automatically and leaves other languages alone. Without the
bundle, put `+` before the stressed vowel (`Прив+іт`); the model still speaks,
but the stress may be wrong. Voices: `ukr_kateryna`, `ukr_lada`, `ukr_mykyta`,
`ukr_oleksa`, `ukr_tetiana`. Selecting one of them loads `v5_cis_ext`
automatically; `--model` forces a directory.

```sh
cargo run --release --example read_text -- \
    --file examples/ukrainian.txt --lang uk \
    --voice ukr_mykyta --out ukrainian.wav
```

`scripts/stress_uk.py` remains a standalone way to produce `+` marks with the
Python `silero-stress` package, for a `v5_cis_ext` export without the bundled
accentor or for checking the Rust accentor:

```sh
uv run scripts/stress_uk.py examples/lina.txt -o /tmp/lina_stressed.txt
```

The Rust text frontend maps typographic apostrophes to `'` and keeps `!`, `'` and
dashes, which the original CIS filter used to drop.

## Model files

The weights are **not** distributed here. They derive from the Silero v5_ru package,
which is licensed CC BY-NC-SA 4.0. `scripts/export_silero.py` reads the package from
the torch-hub cache (SHA-256 `7ba04d42…97ab4`), dequantises the BERT embeddings once
and writes f32 safetensors plus golden tensors to `models/silero-v5-ru/` (~298 MB).
Run it once with `uv run scripts/export_silero.py`.

For Ukrainian, `scripts/export_silero_cis.py` reads `v5_cis_ext.pt`
(SHA-256 `e8185323…06eed2`) and writes `models/silero-v5-cis-ext/` (~97 MB with
the accentor).
`scripts/export_accentor_uk.py --out models/silero-v5-cis-ext` adds the
Ukrainian accentor; the exporter updates the model config, so the order of the
two commands does not matter:

```sh
uv run scripts/export_silero_cis.py /tmp/v5_cis_ext.pt
uv run scripts/export_accentor_uk.py --out models/silero-v5-cis-ext
```

The script declares `torch`, `safetensors` and `numpy` in its `uv` header; the
weights are CC BY-NC, the accentor is MIT.

For the CIS base model, `scripts/export_silero_cis_nostress.py` reads
`v5_cis_base_nostress.pt` (SHA-256 `0405777e…1b8761`) and writes
`models/silero-v5-cis-base-nostress/`; `scripts/export_accentor_uk.py` adds the
Ukrainian `silero-stress` accentor (MIT, SHA-256 `3a649441…0ce13c`):

```sh
uv run scripts/export_silero_cis_nostress.py /tmp/v5_cis_base_nostress.pt
uv run scripts/export_accentor_uk.py
```

## Tests

```sh
cargo test                                    # unit tests; goldens skip without the model
cargo test --release --test golden            # 34 sentences against the Python reference
cargo test --release --test golden_cis        # Ukrainian sentence against the Python reference
cargo test --release --test golden_cis_nostress   # 2 CIS-base sentences with the Python accentor
cargo test --release --test golden_accentor_uk    # Ukrainian accentor logits and sentences
cargo clippy --all-targets -- -D warnings
```

Set `SILERO_REQUIRE_GOLDEN=1` to fail instead of skipping when the model is missing, and to make a debug build fail instead of skipping the golden test.

Golden tests compare token IDs exactly, per-stage tensors within 1e-4…5e-3 (shapes must
match) and audio within 5e-3. The largest measured audio difference is 2.9e-3 (sentence 4,
aidar); sentence 33 is about 3.9e-5. Sentence 33 covers the padding path with two homographs
of different BERT token counts. For `v5_cis_base_nostress`, the ConvNeXt and magnitude
intermediates use scaled tolerances because a 1e-5 mel difference amplifies to ~5e-3; the
same Rust backbone on Python's mel differs by only 1e-4, and the measured audio difference
is 6.5e-4. The accentor compares logits with a relative term and exact accented
strings. `chunks_match_python` is a regression snapshot of
`split_chunks` against `golden/chunks.json`, a mirror of the chunking in
`scripts/export_silero.py`; it does not compare against the Python reference. See
`SPEC-STE.md` for the pipeline contract and the finding register.

## Performance

Apple M1, CPU, `--release`: RTF ≈ 0.033, first PCM after 0.09 s cold, peak RSS 452 MB
(measured against the Python reference at 0.49 s / 1.96 s / 931 MB).

## License

The crate source is CC BY-NC-SA 4.0, inherited from the Silero v5_ru package. See
`LICENSE` and `NOTICE`. Model weights keep the license of their source package:

| Component | License |
|---|---|
| crate source | CC BY-NC-SA 4.0 |
| `v5_ru` weights | CC BY-NC-SA 4.0 |
| `v5_cis_ext` weights | CC BY-NC |
| `v5_cis_base_nostress` weights | MIT |
| Ukrainian accentor (`silero-stress`) | MIT |
