# silero-v5-ru-candle-rs

Silero v5_ru text-to-speech in Rust on [candle](https://github.com/huggingface/candle).
One Russian plain-text chunk in, 48 kHz mono f32 PCM out. CPU only.

The crate is a port of the Silero v5_ru TorchScript pipeline and owns all of it:

- text frontend: normalisation, symbol mapping, chunk splitting;
- homograph solver: rubert-tiny2-shaped BERT plus its classifier;
- accentor: n-gram embedding, stress and `ё` heads, exception list;
- acoustic model: duration/pitch predictors, encoder, LengthRegulator, HourGlass;
- vocoder: ConvNeXt backbone, ISTFT head.

Not affiliated with or endorsed by Silero.

## Requirements

- Rust 1.85+ (edition 2024)
- `candle` 0.11, CPU backend
- Model files under `models/silero-v5-ru/` (not in git, see below)

## Usage

```rust
use silero_v5_ru_candle::Silero;

let model = Silero::load(std::path::Path::new("models/silero-v5-ru"))?;
let samples: Vec<f32> = model.synthesize("Привет, как дела?", "baya")?;
```

Five voices: `aidar`, `baya`, `kseniya`, `xenia`, `eugene`. Plain text only: no SSML
pauses, no `speaker="random"`.

A small CLI writes a WAV:

```sh
cargo run --release --bin silero -- --out out.wav --voice baya "Привет, как дела?"
```

## Model files

The weights are **not** distributed here. They derive from the Silero v5_ru package,
which is licensed CC BY-NC-SA 4.0. `scripts/export_silero.py` reads the package from
the torch-hub cache (SHA-256 `7ba04d42…97ab4`), dequantises the BERT embeddings once
and writes f32 safetensors plus golden tensors to `models/silero-v5-ru/` (~298 MB).
Run it once with `uv run scripts/export_silero.py`.

## Tests

```sh
cargo test                                    # unit tests; goldens skip without the model
cargo test --release --test golden            # 34 sentences against the Python reference
cargo clippy --all-targets -- -D warnings
```

Golden tests compare token IDs exactly, per-stage tensors within 1e-4…5e-3 and audio
within 5e-3. Sentence 33 covers the padding path with two homographs of different BERT
token counts. See `SPEC-STE.md` for the pipeline contract and the finding register.

## Performance

Apple M1, CPU, `--release`: RTF ≈ 0.033, first PCM after 0.09 s cold, peak RSS 452 MB
(measured against the Python reference at 0.49 s / 1.96 s / 931 MB).

## License

CC BY-NC-SA 4.0, inherited from the Silero v5_ru package. See `LICENSE` and `NOTICE`.
No commercial use; derivative works must keep the same license.
