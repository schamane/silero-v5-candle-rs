//! Silero v5 nativ in Rust: russischer und ukrainischer Akzentor, Homograph-Löser,
//! FastPitch-Netze und ISTFT-Vocoder in `candle`.

mod acoustic;
mod config;
mod error;
mod text;
mod vocoder;
mod wav;

use std::borrow::Cow;
use std::fmt;
use std::path::Path;
use std::sync::LazyLock;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::{ThreadPool, ThreadPoolBuilder};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use acoustic::forward::Acoustic;
#[cfg(not(feature = "test-util"))]
use acoustic::forward::AcousticOutput;
use config::{Config, StressConfig};
use error::stage;
use text::accentor::{AccentOptions, Accentor};
use text::homosolver::HomoSolver;
#[cfg(not(feature = "test-util"))]
use text::normalize::PreparedText;
use text::normalize::prepare_text_input;
use vocoder::convnext::Backbone;
use vocoder::istft::{Head, Spectrum};

pub use error::{Error, Result};
pub use text::accentor_uk::{StressOptions, UkrainianAccentor};
pub use text::normalize::{has_cyrillic, split_chunks};
pub use wav::write_wav;

// Nur für die Golden-Tests (opt-in über das Feature `test-util`, nicht Teil der stabilen API).
#[cfg(feature = "test-util")]
pub use acoustic::forward::AcousticOutput;
#[cfg(feature = "test-util")]
pub use text::normalize::PreparedText;

pub const SAMPLE_RATE: u32 = 48_000;

/// Maximale Chunk-Länge in Codepoints; die Satzteilung folgt `SPEC-STE.md`
/// (`split_chunks(text, 400)`) und hält Text unter den Positionsgrenzen von BERT und Akustik.
pub const CHUNK_LIMIT: usize = 400;

/// Rechengerät für [`Silero::load`]: Metal, wenn der Backend-Start und ein kleiner
/// GEMM gelingen, sonst CPU. Mit `--no-default-features` gibt es kein Metal.
///
/// Der Probelauf läuft einmal je Prozess und wird gemerkt: [`Silero::load`] und
/// ein Aufrufer, der das Gerät nur anzeigen will, nennen dasselbe Gerät, und der
/// zweite Aufruf erzeugt keine zweite Metal-Instanz.
pub fn default_device() -> Device {
    static DEVICE: LazyLock<Device> = LazyLock::new(probe_device);
    DEVICE.clone()
}

/// Führt den Probelauf; nur [`default_device`] ruft das.
#[cfg(feature = "metal")]
fn probe_device() -> Device {
    // candle indiziert `MTLCopyAllDevices()`; ohne Metal-Gerät panikt der Konstruktor,
    // statt `Err` zu liefern. Eine Panik heißt hier "kein Metal".
    let metal = std::panic::catch_unwind(|| Device::new_metal(0).ok())
        .ok()
        .flatten();
    metal
        .filter(|device| metal_probe(device).is_ok())
        .unwrap_or(Device::Cpu)
}

#[cfg(not(feature = "metal"))]
fn probe_device() -> Device {
    Device::Cpu
}

/// Prüft Rechnen und Rücktransfer auf `device`. `Device::new_metal` meldet auch dann
/// Erfolg, wenn der erste Kernel scheitert.
#[cfg(feature = "metal")]
fn metal_probe(device: &Device) -> candle_core::Result<()> {
    let square = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?;
    square.matmul(&square)?.to_vec2::<f32>()?;
    Ok(())
}

/// Betonungsmodell eines Pakets; die Variante macht ungültige Kombinationen
/// unrepräsentierbar.
#[expect(
    clippy::large_enum_variant,
    reason = "ein Modell hält genau eine Instanz; Boxen spart hier nichts"
)]
enum StressModel {
    /// `v5_ru`: Homograph-Löser läuft vor dem Akzentor.
    Ru {
        accentor: Accentor,
        homosolver: HomoSolver,
    },
    /// `v5_cis_base_nostress`: ein Akzentor ohne Homograph-Löser.
    Uk(UkrainianAccentor),
}

/// Aufgelöste Stimme: Sprecher-ID und ob der Akzentor des Pakets für sie gilt.
#[derive(Clone, Copy, Default)]
struct Voice {
    id: u32,
    accent: bool,
}

/// Ergebnis der Betonungsstufe; `Cow` spart die Kopie, wenn eine Stufe nichts ändert.
struct Accented<'s> {
    #[cfg_attr(
        not(feature = "test-util"),
        expect(dead_code, reason = "nur `debug_stages` liest die Zwischenstufe")
    )]
    homo: Cow<'s, [char]>,
    accented: Cow<'s, [char]>,
}

/// Obergrenze paralleler Chunks in [`Silero::synthesize_text`] auf der CPU.
pub const MAX_PARALLEL_CHUNKS: usize = 4;

/// Eigener Pool statt des globalen: Er begrenzt, wie viele Chunks gleichzeitig ihre
/// Zwischentensoren halten (grob 25 MB je Chunk bei T=1767), und candles eigenes rayon
/// in den Chunks läuft verschachtelt im selben Pool, ohne Deadlock. Wartet ein Thread in
/// einem inneren `join`, kann er einen weiteren Chunk stehlen; die Grenze gilt also
/// näherungsweise. `None`, wenn der Pool nicht startet: dann läuft alles sequentiell.
fn chunk_pool() -> Option<&'static ThreadPool> {
    static POOL: LazyLock<Option<ThreadPool>> = LazyLock::new(|| {
        let threads = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(MAX_PARALLEL_CHUNKS);
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("silero-chunk-{index}"))
            .build()
            .ok()
    });
    POOL.as_ref()
}

/// Lädt das ISTFT-Fenster aus `istft_window.safetensors` und prüft die Länge gegen
/// `win_length`.
///
/// Das Fenster lebt als `Vec<f32>` weiter; auf der CPU laden spart einen GPU-Umweg über
/// ein Metal-Buffer.
fn load_window(dir: &Path, win_length: usize) -> Result<Vec<f32>> {
    let window = candle_core::safetensors::load(dir.join("istft_window.safetensors"), &Device::Cpu)
        .map_err(|e| Error::model("istft_window.safetensors", e))?
        .remove("window")
        .ok_or_else(|| Error::model_msg("istft_window.safetensors: window fehlt"))?
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| Error::model("istft_window.safetensors", e))?;
    if window.len() != win_length {
        return Err(Error::model_msg(format!(
            "ISTFT-Fenster hat {} statt {win_length} Werte",
            window.len()
        )));
    }
    Ok(window)
}

/// Das Silero-v5-Modell: Akustik, Vocoder und gegebenenfalls Betonungsmodell eines Pakets.
pub struct Silero {
    config: Config,
    /// Sprecher-Namen, sortiert; `voices` und `has_voice` brauchen so keine Kopie.
    voices: Vec<String>,
    acoustic: Acoustic,
    /// `None` bei `v5_cis_ext`: der Aufrufer setzt die Betonung mit `+`.
    stress: Option<StressModel>,
    backbone: Backbone,
    head: Head,
    device: Device,
}

impl fmt::Debug for Silero {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Silero")
            .field("device", &self.device)
            .field("voices", &self.voices)
            .finish_non_exhaustive()
    }
}

impl Silero {
    /// Lädt Konfiguration, Gewichte und Wörterbücher aus `dir` auf das schnellste
    /// verfügbare Gerät: Metal, wenn der Probelauf gelingt, sonst CPU.
    ///
    /// Welches Gerät gewählt wurde, sagt [`Silero::device`].
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Io`] zurück, wenn eine Datei nicht gelesen werden kann, und
    /// [`Error::Model`], wenn eine Datei fehlt, ungültig ist oder nicht zur Architektur passt.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with_device(dir, &default_device())
    }

    /// Lädt das Modell auf ein explizit gewähltes Gerät, etwa `&Device::Cpu` für einen
    /// Vergleich mit Metal oder für Umgebungen ohne Metal.
    ///
    /// # Errors
    ///
    /// Gibt dieselben Fehler wie [`Silero::load`] zurück; ein fehlschlagender
    /// Geräte-Operator erscheint als [`Error::Model`] oder [`Error::Pipeline`].
    pub fn load_with_device(dir: &Path, device: &Device) -> Result<Self> {
        let config = Config::load(&dir.join("config.json"))?;

        let weights = dir.join("acoustic.safetensors");
        // SAFETY: Die Modelldateien in `dir` dürfen während des Mappings nicht geändert
        // oder ersetzt werden. Sonst ist der Inhalt des gemappten Speichers undefiniert.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, device)
                .map_err(|e| Error::model(weights.display().to_string(), e))?
        };
        let acoustic = Acoustic::load(
            vb.clone(),
            device,
            config.symbol_count(),
            config.speaker_count(),
        )
        .map_err(|e| Error::model("acoustic", e))?;
        let backbone =
            Backbone::load(vb.pp("vocoder.backbone")).map_err(|e| Error::model("vocoder", e))?;
        let window = load_window(dir, config.win_length)?;
        let head = Head::load(
            vb.pp("vocoder.head"),
            window,
            config.n_fft,
            config.hop_length,
            config.win_length,
        )
        .map_err(|e| Error::model("vocoder.head", e))?;
        let stress = match config.stress() {
            StressConfig::Ru { bert, tokenizer } => Some(StressModel::Ru {
                accentor: Accentor::load(dir)?,
                homosolver: HomoSolver::load(dir, device, bert, tokenizer)?,
            }),
            StressConfig::Uk(_) => Some(StressModel::Uk(UkrainianAccentor::load(dir)?)),
            StressConfig::None => None,
        };
        let mut voices: Vec<String> = config.speakers.keys().cloned().collect();
        voices.sort_unstable();
        Ok(Self {
            config,
            voices,
            acoustic,
            stress,
            backbone,
            head,
            device: device.clone(),
        })
    }

    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn prepare_text(&self, text: &str) -> PreparedText {
        prepare_text_input(&self.config, text)
    }

    /// `true`, wenn das Modell Stress selbst setzt (`v5_ru`, `v5_cis_base_nostress`,
    /// optional `v5_cis_ext`).
    #[must_use]
    pub fn has_accentor(&self) -> bool {
        self.config.has_accentor()
    }

    /// `true`, wenn der Akzentor für die Sprache dieser Stimme gilt. Ein
    /// ukrainischer Akzentor im CIS-Paket betrifft nur `ukr_*`-Stimmen.
    #[must_use]
    pub fn has_accentor_for(&self, voice: &str) -> bool {
        self.config.has_accentor_for(voice)
    }

    /// Alle Sprecher-Namen aus `config.json`, sortiert.
    #[must_use]
    pub fn voices(&self) -> &[String] {
        &self.voices
    }

    /// `true`, wenn `config.json` die Stimme `voice` kennt.
    #[must_use]
    pub fn has_voice(&self, voice: &str) -> bool {
        self.voices
            .binary_search_by(|name| name.as_str().cmp(voice))
            .is_ok()
    }

    /// `apply_tts` für einen Chunk; Satzteilung und Pausen liegen beim Aufrufer.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Input`] zurück, wenn der Text keine sprechbaren Zeichen enthält oder
    /// `voice` unbekannt ist, und [`Error::Pipeline`], wenn eine Stufe fehlschlägt.
    pub fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<f32>> {
        let prepared = self.prepare_text(text);
        if !prepared.has_text {
            return Err(Error::Input("text is empty".into()));
        }
        self.synthesize_prepared(prepared, self.resolve_voice(voice)?)
    }

    /// Synthetisiert beliebig langen Text lazy: zerlegt ihn in Chunks von höchstens
    /// [`CHUNK_LIMIT`] Codepoints und liefert pro Chunk mit sprechbaren Zeichen ein
    /// Audio-Stück, der Reihe nach und nacheinander berechnet. So lässt sich streamen.
    ///
    /// Eine unbekannte Stimme erscheint als einziges Element `Err(`[`Error::Input`]`)`,
    /// bevor ein Chunk berechnet wird. Text ohne sprechbare Zeichen liefert keine Elemente.
    pub fn synthesize_chunks<'model>(
        &'model self,
        text: &str,
        voice: &str,
    ) -> impl Iterator<Item = Result<Vec<f32>>> + use<'model> {
        let (voice_error, voice, chunks) = match self.resolve_voice(voice) {
            Ok(voice) => (None, voice, self.prepared_chunks(text)),
            Err(error) => (Some(error), Voice::default(), Vec::new()),
        };
        voice_error.map(Err).into_iter().chain(
            chunks
                .into_iter()
                .map(move |prepared| self.synthesize_prepared(prepared, voice)),
        )
    }

    /// Synthetisiert beliebig langen Text: zerlegt ihn in Chunks von höchstens
    /// [`CHUNK_LIMIT`] Codepoints und hängt die Audiodaten ohne Pausen aneinander.
    ///
    /// Auf der CPU laufen mehrere Chunks parallel (höchstens [`MAX_PARALLEL_CHUNKS`]
    /// Threads); auf Metal nacheinander. Das Ergebnis ist in beiden Fällen
    /// bitgleich zu [`Silero::synthesize_chunks`], aneinandergehängt.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn ein Chunk nicht synthetisiert werden kann
    /// ([`Error::Input`] oder [`Error::Pipeline`]), oder wenn der Text keine sprechbaren
    /// Zeichen enthält ([`Error::Input`]). Scheitern mehrere Chunks parallel, ist
    /// unbestimmt, welcher Fehler zurückkommt.
    pub fn synthesize_text(&self, text: &str, voice: &str) -> Result<Vec<f32>> {
        let voice = self.resolve_voice(voice)?;
        let chunks = self.prepared_chunks(text);
        if chunks.is_empty() {
            return Err(Error::Input("text is empty".into()));
        }
        let parallel = chunks.len() > 1 && matches!(self.device, Device::Cpu);
        let pool = if parallel { chunk_pool() } else { None };
        let parts: Vec<Vec<f32>> = match pool {
            Some(pool) => pool.install(|| {
                chunks
                    .into_par_iter()
                    .map(|prepared| self.synthesize_prepared(prepared, voice))
                    .collect::<Result<Vec<_>>>()
            })?,
            None => chunks
                .into_iter()
                .map(|prepared| self.synthesize_prepared(prepared, voice))
                .collect::<Result<Vec<_>>>()?,
        };
        let mut audio = Vec::with_capacity(parts.iter().map(Vec::len).sum());
        for part in parts {
            audio.extend(part);
        }
        Ok(audio)
    }

    /// Zerlegt `text` und bereitet jeden Chunk einmal vor; Chunks ohne sprechbare
    /// Zeichen entfallen.
    fn prepared_chunks(&self, text: &str) -> Vec<PreparedText> {
        split_chunks(text, CHUNK_LIMIT)
            .iter()
            .map(|chunk| self.prepare_text(chunk))
            .filter(|prepared| prepared.has_text)
            .collect()
    }

    /// Löst die Stimme einmal je Aufruf auf, vor jeder BERT-Arbeit.
    fn resolve_voice(&self, voice: &str) -> Result<Voice> {
        Ok(Voice {
            id: self.config.speaker_id(voice)?,
            accent: self.config.accentor_matches(voice),
        })
    }

    /// Betonung: Homograph-Löser (nur `v5_ru`), dann Akzentor. Ohne Änderung bleibt
    /// der Satz geborgt, es gibt keine Kopie.
    fn accent<'s>(&self, sentence: &'s [char], voice: Voice) -> Result<Accented<'s>> {
        let homo = match &self.stress {
            Some(StressModel::Ru { homosolver, .. }) => {
                Cow::Owned(homosolver.solve(sentence, AccentOptions::default())?)
            }
            _ => Cow::Borrowed(sentence),
        };
        let accented = match &self.stress {
            Some(StressModel::Ru { accentor, .. }) => {
                Cow::Owned(accentor.accent(&homo, AccentOptions::default()))
            }
            Some(StressModel::Uk(accentor)) if voice.accent => {
                Cow::Owned(accentor.accent(sentence, StressOptions::default()))
            }
            Some(StressModel::Uk(_)) | None => Cow::Borrowed(sentence),
        };
        Ok(Accented { homo, accented })
    }

    fn acoustic_stage(&self, ids: &[u32], voice: Voice) -> Result<AcousticOutput> {
        self.acoustic
            .forward(ids, voice.id)
            .map_err(stage("akustik"))
    }

    fn features_stage(&self, mel: &Tensor) -> Result<Tensor> {
        self.backbone.forward(mel).map_err(stage("vocoder"))
    }

    fn spectrum_stage(&self, features: &Tensor) -> Result<Spectrum> {
        self.head.forward(features).map_err(stage("vocoder-head"))
    }

    fn istft_stage(&self, spectrum: &Spectrum) -> Result<Vec<f32>> {
        self.head.istft(spectrum).map_err(stage("istft"))
    }

    /// Die schlanke Produktionspipeline: gibt jede Zwischengröße frei, sobald die nächste
    /// Stufe sie verbraucht hat, und kopiert den Satz nicht.
    fn synthesize_prepared(&self, prepared: PreparedText, voice: Voice) -> Result<Vec<f32>> {
        if !prepared.has_text {
            return Err(Error::Input("text is empty".into()));
        }
        let ids = {
            let accented = self.accent(&prepared.sentence, voice)?;
            self.config.sequence(&accented.accented)?
        };
        drop(prepared);
        let features = {
            let output = self.acoustic_stage(&ids, voice)?;
            self.features_stage(&output.mel)?
        };
        let spectrum = self.spectrum_stage(&features)?;
        drop(features);
        self.istft_stage(&spectrum)
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Hilfen für die Golden-Tests; nur mit dem Feature `test-util`.
#[cfg(feature = "test-util")]
impl Silero {
    /// Normalisierter Text und Satz, wie die Pipeline sie sieht.
    #[must_use]
    pub fn prepare(&self, text: &str) -> PreparedText {
        self.prepare_text(text)
    }

    /// Nur für die Golden-Tests (Feature `test-util`): alle Zwischenstände einer Stimme.
    /// Nutzt dieselben Stufen wie [`Silero::synthesize`], behält aber jede Zwischengröße.
    ///
    /// # Errors
    ///
    /// Gibt dieselben Fehler wie [`Silero::synthesize`] zurück.
    pub fn debug_stages(&self, text: &str, voice: &str) -> Result<Stages> {
        let prepared = self.prepare_text(text);
        if !prepared.has_text {
            return Err(Error::Input("text is empty".into()));
        }
        let voice = self.resolve_voice(voice)?;
        let Accented { homo, accented } = self.accent(&prepared.sentence, voice)?;
        let (homo, accented) = (homo.into_owned(), accented.into_owned());
        let ids = self.config.sequence(&accented)?;
        let output = self.acoustic_stage(&ids, voice)?;
        let features = self.features_stage(&output.mel)?;
        let spectrum = self.spectrum_stage(&features)?;
        let audio = self.istft_stage(&spectrum)?;
        // Die Stufen liefern `[1, 1201, T]`; die Transposition ist nur eine Sicht ohne Kopie.
        let stage_view = |tensor: &Tensor| tensor.transpose(1, 2).map_err(stage("vocoder-head"));
        let (mag, cos, sin) = (
            stage_view(&spectrum.mag)?,
            stage_view(&spectrum.cos)?,
            stage_view(&spectrum.sin)?,
        );
        Ok(Stages {
            prepared,
            homo,
            accented,
            ids,
            output,
            features,
            mag,
            cos,
            sin,
            audio,
        })
    }

    /// Nur für die Golden-Tests (Feature `test-util`): Homosolver-Logits pro Homograph.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Model`] oder [`Error::Pipeline`] zurück, wenn der Homosolver fehlschlägt.
    pub fn homosolver_logits(&self, text: &str) -> Result<Option<Tensor>> {
        let prepared = self.prepare_text(text);
        if !prepared.has_text {
            return Ok(None);
        }
        match &self.stress {
            Some(StressModel::Ru { homosolver, .. }) => homosolver.logits(&prepared.sentence),
            _ => Ok(None),
        }
    }
}

// `Silero` muss von mehreren Threads gleichzeitig nutzbar sein (parallele Chunks).
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Silero>();
};

/// Zwischenstände der Pipeline. Nur für die Golden-Tests (Feature `test-util`);
/// die Felder sind lesbar, damit die Tests jede Stufe gegen Python vergleichen können.
#[cfg(feature = "test-util")]
#[expect(
    missing_debug_implementations,
    reason = "Testhilfe; Tensoren der Stufen sind nicht von Interesse"
)]
pub struct Stages {
    pub prepared: PreparedText,
    pub homo: Vec<char>,
    pub accented: Vec<char>,
    pub ids: Vec<u32>,
    pub output: acoustic::forward::AcousticOutput,
    pub features: Tensor,
    pub mag: Tensor,
    pub cos: Tensor,
    pub sin: Tensor,
    pub audio: Vec<f32>,
}
