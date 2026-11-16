# silero-v5-candle-rs

Silero v5 text-to-speech in Rust on [candle](https://github.com/huggingface/candle).
One plain-text chunk in, 48 kHz mono f32 PCM out. Metal on macOS; CPU where Metal
is unavailable or disabled. Russian (`v5_ru`),
the CIS base model (`v5_cis_base_nostress`) and `v5_cis_ext` are supported. Both
CIS packages can bundle the Ukrainian n-gram accentor; for `v5_cis_ext` it is
optional and only applied to `ukr_*` voices. The other CIS languages in those
packages are not wired up yet: they need the `ext_alph` conversion.

**This project is 100% AI-based.** All code, tests, benchmarks and documentation
were written by AI coding agents; the maintainer directs the work, reviews the
result and owns the releases.

The crate is a port of the Silero TorchScript pipeline and owns all of it:

- text frontend: normalization, symbol mapping, chunk splitting;
- `HomoSolver` and Russian accentor (`v5_ru`): rubert-tiny2-shaped BERT plus
  n-gram stress/`ё` heads, the exception list and an optional Wiktionary stress
  dictionary; Ukrainian n-gram accentor for `v5_cis_base_nostress` and, if
  bundled, `v5_cis_ext`;
- acoustic model: duration/pitch predictors, encoder, LengthRegulator, HourGlass;
- vocoder: ConvNeXt backbone, ISTFT head.

Not affiliated with or endorsed by Silero.

## Requirements

- Rust 1.88+ (edition 2024, Let-Chains)
- `candle` 0.11.1 with the Metal or CPU backend. The `metal` feature is on by
  default and only compiles on Apple targets; elsewhere build with
  `--no-default-features` for the CPU-only variant.
- Optional on macOS: Apple Accelerate via `--features accelerate` (CPU BLAS;
  about 1.7–1.9× faster than candle's pure-Rust CPU GEMM on an M1 in earlier measurements, not re-measured).
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

`Silero::load` probes Metal once: if the backend starts and a small GEMM round-trips,
all model tensors run there; otherwise the whole model is loaded on the CPU.
`Silero::device()` reports the choice. `default_device()` runs the same probe and
caches its result for the process, so a caller that only reports the device creates
no second Metal instance and `Silero::load` reuses the probed device.
`Silero::load_with_device(dir, &Device::Cpu)`
forces a device, for example for CPU/Metal comparisons.

The CLI selects the backend with `--device auto|cpu|metal`. For the fastest CPU
path on macOS, build without Metal and with Accelerate:

```sh
cargo build --release --no-default-features --features accelerate
target/release/silero --device cpu --out out.wav "Привет, как дела?"
```

`auto` is `Silero::load` (Metal when usable, otherwise CPU); `metal` fails
cleanly when no Metal device is available.

The device can also be set with `SILERO_DEVICE=auto|cpu|metal` (read by `Silero::load` and
the CLI; `--device` wins) or in code with `Silero::load_with_choice(dir, DeviceChoice::Cpu)`.
On a base M1, Metal was faster for a single chunk (about 1.6x for a short and 1.2x for a long
chunk); for multi-chunk text with parallel chunks, CPU and Metal were close in the measurements
(see Performance).

Concurrency: on Metal a process-wide lock serialises the GPU stages (candle 0.11.1 can
crash when buffers are released concurrently); the CPU ISTFT runs outside it. A Metal
failure mid-run surfaces as `Err(Error::Pipeline)` with no CPU fallback, and a panic in a
thread poisons candle's command encoder for later Metal calls.

Long text should use `Silero::synthesize_text`, which splits it into chunks of at most 400 characters and joins the audio without a pause. Chunks run in parallel on both CPU and Metal, up to `MAX_PARALLEL_CHUNKS` (8).

Five voices: `aidar`, `baya`, `kseniya`, `xenia`, `eugene`. Plain text only: no SSML
pauses, no `speaker="random"`.

### Optional stress dictionary

`v5_ru` already ships a 16,969-entry `exceptions.tsv`. For broader coverage, the
RU accentor also reads an optional `models/silero-v5-ru/stress_dict.tsv`
(`word<TAB>position`, lowercase, 1-based stressed character, sorted by word). Generate it
from the public [Wiktionary-derived list](https://github.com/eugenysilin/russian-ukrainian-stress-data):

```sh
uv run scripts/export_stress_dict.py /tmp/stress-data/ru-stress-dict.gz \
    models/silero-v5-ru/stress_dict.tsv
```

The dictionary takes precedence over the n-gram heads but not over the package's
`exceptions.tsv` or explicit `+` marks. On the 60-sentence stress benchmark it
raises the Rust accentor from 331/335 to 335/335 and from 56/60 to 60/60 exact
sentences. The source data is CC BY-SA 4.0; the file is not part of the repo.

`Silero::accent_text(text, voice)` exposes the same stress stage without
synthesizing audio, for A/B tests. Context homographs are handled where the
package's classifier is insufficient: `на двери` uses the prepositional
`двер+и` (the genitive/dative `дв+ери` is kept after `у`/`к`).

A separate 60-sentence hard-case suite (mobile stress, short forms, `-ся`,
compounds, homographs, clitics) lives in `tests/data/stress_hard60.json` and scores
58/60 with the dictionary. The two remaining failures are context homographs
(`права` after a copula, plural `стены` before `были`); they are tracked as A-148 in
`SPEC-STE.md`. A separate Ukrainian hard-case suite
(`tests/data/stress_hard60_uk.json`, voice `ukr_igor`, 15 categories) scores 45/60;
it exposes the missing dictionary/homograph layer of the Ukrainian n-gram accentor
(A-149).

Both scores are reproducible and pinned by `tests/stress_hard.rs`:

```sh
cargo test --release --test stress_hard -- --nocapture
# or manually, per suite:
cargo run --release --example accent_suite -- models/silero-v5-ru \
    tests/data/stress_hard60.json baya > /tmp/ru.txt
python3 scripts/stress_bench_hard.py tests/data/stress_hard60.json /tmp/ru.txt
cargo run --release --example accent_suite -- models/silero-v5-cis-base-nostress \
    tests/data/stress_hard60_uk.json ukr_igor > /tmp/uk.txt
python3 scripts/stress_bench_hard.py tests/data/stress_hard60_uk.json /tmp/uk.txt
```

The RU score includes the optional `stress_dict.tsv` layer; without the generated
dictionary the test skips.

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
Run it once with `uv run scripts/export_silero.py`. An optional
`stress_dict.tsv` (see above) can be added afterwards.

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
cargo test --release --no-default-features --test golden   # same suite on the CPU
cargo test --release --no-default-features --features accelerate --test golden
cargo test --release --test golden_cis        # Ukrainian sentence against the Python reference
cargo test --release --test golden_cis_nostress   # 2 CIS-base sentences with the Python accentor
cargo test --release --test golden_accentor_uk    # Ukrainian accentor logits and sentences
cargo test --release --test accent_dictionary      # optional stress_dict.tsv + accent_text
cargo test --release --test parallel               # synthesize_chunks vs synthesize_text, bit-identical
cargo test --release --test metal_vs_cpu           # Metal against CPU audio (macOS, Metal)
cargo test --release --test metal_concurrency      # concurrent synthesize on Metal (macOS, Metal)
make test                                         # all tests, release build (goldens run)
make test-debug                                   # all tests, debug build (goldens skip)
make check-linux                                  # cargo check --lib for Linux; skips if the target is not installed
make lint-rust                                    # clippy, default, CPU and Accelerate paths
make check                                        # lint + release tests on Metal, CPU and Accelerate
```

Set `SILERO_REQUIRE_GOLDEN=1` to fail instead of skipping when the model is missing, and to make a debug build fail instead of skipping the golden test. Golden tests run on the default device (Metal on macOS); `--no-default-features` pins the CPU path; `--features accelerate` runs the CPU path through Apple's BLAS. Compare the runs before changing tolerances.

Golden tests compare token IDs exactly, per-stage tensors within 1e-4…5e-3 (shapes must
match) and audio within 5e-3. On Metal, `backbone` uses 5e-3 (measured 3.7e-3) and `mag`
2.5e-2 (measured 1.7e-2) because f32 GEMMs accumulate in a different order; all other
bounds, including audio at 5e-3, are shared. The Accelerate/MKL CPU path uses a
`backbone` bound of 2e-3 (measured 1.43e-3) for the same reason; the CIS goldens
already pass unchanged. Metal and CPU audio differ by at most 1.6e-5 on a short
text, 2.1e-4 on a long text and 2.8e-4 over the corpus (about 9 int16 LSB; test tolerance
1e-3 in `tests/metal_vs_cpu.rs`); Accelerate vs. pure CPU differs by at most 2.2e-3
over the 34-sentence corpus. The largest measured audio difference is 2.9e-3 (sentence 4,
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

Apple M1 (8 Cores, 16 GiB), macOS 27.2.1, rustc 1.100.0, candle 0.11.1, `--release`, Stimme `baya`.
`MAX_PARALLEL_CHUNKS` ist 8; die Chunks laufen auf CPU und Metal parallel.

Gemessen am 2026-11-16 auf dem Endstand (Median aus 55 Läufen (Laden: 45), 5 Runden, je 2 Aufwärmläufe),
ohne `stress_dict.tsv`. Eingaben: kurz = 32 Zeichen (2,14 s Audio), lang = 305 Zeichen, ein Chunk
(22,09 s), Corpus = die ersten 20 Zeilen von `tests/corpus.txt` (58,6 s, parallele Chunks
über `synthesize_text`). RTF = Zeit / Audiodauer. Die Spalte „Faktor“ ist die Beschleunigung gegenüber
dem Stand vor der Optimierung (`67cbc11`) auf CPU / Metal.

| Eingabe | CPU ms | Metal ms | RTF CPU | RTF Metal | Faktor CPU / Metal |
|---|---:|---:|---:|---:|---:|
| kurz | 45,7 | 29,2 | 0,0214 | 0,0137 | 1,32× / 1,71× |
| lang | 345 | 282 | 0,0156 | 0,0127 | 1,42× / 1,52× |
| Corpus, parallel | 694 | 666 | 0,0118 | 0,0114 | 2,12× / 1,99× |
| Corpus, `synthesize_chunks` sequenziell | 1 132 | 812 | 0,0193 | 0,0139 | – |

Metal ist bei einem einzelnen Chunk schneller (kurz 1,56×, lang 1,23×). Beim Corpus mit parallelen
Chunks liegen CPU und Metal dicht beieinander, und die Streuung zwischen den Runden überlappt;
sequenziell ist Metal schneller. Metal streut auf diesem Rechner stärker als die CPU.
Alle Zahlen, Streubereiche, Stufenaufteilung und Methodik: [docs/messungen.html](docs/messungen.html),
Rohdaten: [docs/messungen-rohdaten.json](docs/messungen-rohdaten.json).

Messprogramm: `benches/sbench.rs` (Bench-Ziel ohne Harness). Aufruf zum Beispiel
`cargo bench --bench sbench -- run cpu short,long,corpus`; `RUNS` setzt die Zahl der
gemessenen Läufe (Standard 11, je Eingabe zwei Aufwärmläufe davor) und `MODEL` den Modellordner
(Standard `models/silero-v5-ru` im Repo). Die Zahlen der Stände A, B und C stammen aus je einem
Build desselben Programms gegen den jeweiligen Commit; das Programm im Baum misst nur den
aktuellen Stand.
Weitere Modi: `load`, `rss`, `dump` und `cold` (Laden plus erste Synthese in einem frischen Prozess).

Kaltstart: Die erste Synthese eines frischen Metal-Starts kompilierte in der Messung vom
2026-11-16 die Shader (Stand `67cbc11`: 509 ms Laden, 11 120 ms erste Synthese; frühere Messung der
Fable-Review: 595 ms und 11 176 ms in einem neuen Verzeichnis). Die danach gestarteten Stände
`911e96d` und der Endstand (`15e24c1` plus Arbeitsbaum) brauchten in neuen Verzeichnissen nur 53 bzw. 48 ms für die erste Synthese,
der Cache war dann warm. Wovon er abhängt, wurde nicht untersucht. Die CPU hat keinen Kaltstart.
Die Ladezeiten dieser Kaltstarts sind nicht direkt vergleichbar: Stand A hatte noch keinen
Wörterbuch-Lader, B und C luden `stress_dict.tsv`, falls es im Modellordner lag.

Laden und Speicher (Median, ohne `stress_dict.tsv`): warmes Laden 43 ms (CPU) und 56 ms (Metal);
Spitzen-RSS von Laden plus Corpus 431 MiB (CPU) und 448 MiB (Metal). Mit dem 75 MB großen
`stress_dict.tsv` im Modellordner liegt die Ladezeit bei etwa 240 ms.

Die folgenden Zahlen sind historische Generationenvergleiche vor Accelerate und
Chunk-Parallelität:

| Generation | Commit | Device | Synth | RTF |
|---|---|---|---:|---:|
| CPU-only, whole text | `3317ffc` | CPU | 0.270 s | 0.033 |
| Metal implementation | `67cbc11` | Metal | 0.145 s | 0.018 |
| Post-Metal | `f024297` | Metal | 0.110 s | 0.014 |

Historisches Profil eines 4,94-s-Chunks (395 Mel-Frames, Median aus 9 Läufen, vor
Accelerate und Chunk-Parallelität; das aktuelle Profil steht in
[docs/messungen.html](docs/messungen.html), Abschnitt e):

| Stage | ms | Share |
|---|---:|---:|
| Backbone (ConvNeXt) | 91 | 62 % |
| Acoustic (predictors, Tacotron, HourGlass) | 28 | 19 % |
| Head (`lin` 512 → 2402) | 14 | 10 % |
| ISTFT | 7.4 | 5 % |
| Homosolver (BERT) | 4.7 | 3 % |
| Accentor and text frontend | 0.3 | <1 % |

Inside the backbone about 62 % are the GEMMs (`pwconv1`/`pwconv2`). GELU (erf) was
about 31 %; the CPU path now uses a parallel polynomial approximation (max error
<1e-6, covered by `gelu_matches_gelu_erf_on_cpu`), other devices keep `gelu_erf`.
LayerNorm, layout copies and the depthwise conv together are about 4 %; the layout
stays NTC. The hand-written depthwise conv is roughly eleven times faster than
`conv1d(groups = 512)`, because candle splits the groups into 512 separate
convolutions.

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
| optional `stress_dict.tsv` | CC BY-SA 4.0 (Wiktionary-derived) |
| `tests/data/stress_hard*.json` | CC BY-SA 4.0 (Wiktionary/Gramota-derived gold) |
