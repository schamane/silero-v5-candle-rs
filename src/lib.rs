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
use std::str::FromStr;
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

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
use vocoder::istft::{Head, Spectrum, SpectrumData};

pub use error::{Error, Result};
pub use text::accentor_uk::{StressOptions, UkrainianAccentor};
pub use text::normalize::{has_cyrillic, split_chunks};
pub use wav::write_wav;

#[cfg(all(feature = "accelerate", feature = "mkl"))]
compile_error!("accelerate and mkl are mutually exclusive; build one CPU backend at a time");

// Nur für die Golden-Tests (opt-in über das Feature `test-util`, nicht Teil der stabilen API).
#[cfg(feature = "test-util")]
pub use acoustic::forward::AcousticOutput;
#[cfg(feature = "test-util")]
pub use text::normalize::PreparedText;

pub const SAMPLE_RATE: u32 = 48_000;

/// Maximale Chunk-Länge in Codepoints; die Satzteilung folgt `SPEC-STE.md`
/// (`split_chunks(text, 400)`) und hält Text unter den Positionsgrenzen von BERT und Akustik.
pub const CHUNK_LIMIT: usize = 400;

/// Rechengerät für [`Silero::load`]: Metal, wenn ein Metal-Gerät existiert und ein
/// kleiner GEMM gelingt, sonst CPU. Ohne das Feature `metal` oder außerhalb von macOS
/// gibt es kein Metal.
///
/// Der Probelauf läuft einmal je Prozess und wird gemerkt: [`Silero::load`] und
/// ein Aufrufer, der das Gerät nur anzeigen will, nennen dasselbe Gerät, und der
/// zweite Aufruf erzeugt keine zweite Metal-Instanz.
pub fn default_device() -> Device {
    static DEVICE: LazyLock<Device> = LazyLock::new(probe_device);
    DEVICE.clone()
}

/// Führt den Probelauf; nur [`default_device`] ruft das.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn probe_device() -> Device {
    // candle indiziert `MTLCopyAllDevices()` und panikt ohne Metal-Gerät. Die
    // Vorprüfung über das Standardgerät vermeidet diesen Pfad.
    if objc2_metal::MTLCreateSystemDefaultDevice().is_none() {
        return Device::Cpu;
    }
    Device::new_metal(0)
        .ok()
        .filter(|device| metal_probe(device).is_ok())
        .unwrap_or(Device::Cpu)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn probe_device() -> Device {
    Device::Cpu
}

/// Prüft Rechnen und Rücktransfer auf `device`. `Device::new_metal` meldet auch dann
/// Erfolg, wenn der erste Kernel scheitert.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_probe(device: &Device) -> candle_core::Result<()> {
    let square = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?;
    square.matmul(&square)?.to_vec2::<f32>()?;
    Ok(())
}

/// Wahl des Rechengeräts für [`Silero::load_with_choice`].
///
/// Auf einem Basis-M1 gewinnt Metal bei einem einzelnen kurzen Chunk etwa 20-30 %;
/// bei mehrchunkigem Text ist die CPU dank paralleler Chunks schneller. Für lange
/// Texte hilft `SILERO_DEVICE=cpu`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceChoice {
    /// [`default_device`]: Metal, wenn nutzbar, sonst CPU.
    #[default]
    Auto,
    /// Immer die CPU.
    Cpu,
    /// Metal; Laden schlägt fehl, wenn Metal nicht verfügbar ist.
    Metal,
}

impl FromStr for DeviceChoice {
    type Err = Error;

    /// Liest `auto`, `cpu` oder `metal` ohne Rücksicht auf Groß- und Kleinschreibung.
    fn from_str(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("auto") {
            Ok(Self::Auto)
        } else if value.eq_ignore_ascii_case("cpu") {
            Ok(Self::Cpu)
        } else if value.eq_ignore_ascii_case("metal") {
            Ok(Self::Metal)
        } else {
            Err(Error::Input(format!(
                "unbekanntes Gerät {value:?}; erwartet auto, cpu oder metal"
            )))
        }
    }
}

impl DeviceChoice {
    /// Löst die Wahl in ein Gerät auf.
    fn resolve(self) -> Result<Device> {
        match self {
            Self::Auto => Ok(default_device()),
            Self::Cpu => Ok(Device::Cpu),
            Self::Metal => {
                let device = default_device();
                if device.is_metal() {
                    Ok(device)
                } else {
                    Err(Error::Input(
                        "Metal ist nicht verfügbar (kein Metal-Gerät, Feature `metal` \
                         aus oder Ziel nicht macOS)"
                            .into(),
                    ))
                }
            }
        }
    }
}

/// Prozessweite Sperre für alle Zugriffe auf ein Metal-Gerät.
///
/// candle 0.11.1 gibt Metal-Buffer in `metal_backend/device.rs` (`drop_unused_buffers`,
/// aufgerufen aus `MetalStorage::to_cpu`) ohne ausreichende Synchronisation frei:
/// Läuft das in mehreren Threads gleichzeitig, folgt ein `objc_release` auf einen
/// schon freigegebenen Buffer (SIGSEGV, gemessen in rund 10 % der Prozesse mit
/// 8 Threads). Die Sperre ist global und nicht je [`Silero`], weil [`default_device`]
/// allen Modellen dieselbe `MetalDevice` gibt. Sie schützt keine Daten, deshalb ist
/// eine vergiftete Sperre unbedenklich.
static METAL_LOCK: Mutex<()> = Mutex::new(());

/// Nimmt die Metal-Sperre, wenn `device` Metal ist; auf der CPU kostet das nichts.
fn metal_guard(device: &Device) -> Option<MutexGuard<'static, ()>> {
    device
        .is_metal()
        .then(|| METAL_LOCK.lock().unwrap_or_else(PoisonError::into_inner))
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

/// Obergrenze paralleler Chunks in [`Silero::synthesize_text`], unabhängig vom Backend.
///
/// Auf der CPU bringt der Sprung von 4 auf 8 Threads rund 13 % (M1, 34 Chunks); auf
/// Metal sättigt der Gewinn schon bei 4 Threads. Acht Chunks halten zusammen etwa
/// 550 MB RSS, deshalb bleibt der Wert bewusst moderat.
pub const MAX_PARALLEL_CHUNKS: usize = 8;

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
///
/// # Fehler auf Metal
///
/// Scheitert ein Metal-Aufruf mitten im Lauf, erscheint das als
/// `Err(`[`Error::Pipeline`]`)`; es gibt keinen Rückfall auf die CPU. candles
/// `command_encoder` nimmt seine Sperre mit `lock().unwrap()`: Panikt ein Thread
/// dabei, schlagen alle späteren Metal-Aufrufe fehl oder panikten ebenfalls.
///
/// # Nebenläufigkeit
///
/// Auf Metal serialisiert eine prozessweite Sperre die Geräte-Stufen (Akustik bis
/// Spektrum-Readback), weil candle 0.11.1 beim parallelen Freigeben von Buffern
/// abstürzen kann. Die ISTFT läuft außerhalb der Sperre; auf der CPU gibt es keine Sperre.
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
    /// verfügbare Gerät: Metal, wenn der Probelauf gelingt, sonst CPU. Die
    /// Umgebungsvariable `SILERO_DEVICE` (`auto`, `cpu` oder `metal`) überstimmt
    /// diese Wahl.
    ///
    /// Auf einem Basis-M1 gewinnt Metal bei einem einzelnen kurzen Chunk etwa 20-30 %;
    /// mehrchunkiger Text ist auf der CPU dank paralleler Chunks schneller
    /// (`SILERO_DEVICE=cpu`). Welches Gerät gewählt wurde, sagt [`Silero::device`].
    ///
    /// Mitten im Lauf scheiternde Metal-Aufrufe erscheinen als [`Error::Pipeline`],
    /// ohne Rückfall auf die CPU; siehe die Hinweise bei [`Silero`].
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Io`] zurück, wenn eine Datei nicht gelesen werden kann, und
    /// [`Error::Model`], wenn eine Datei fehlt, ungültig ist oder nicht zur Architektur
    /// passt. [`Error::Input`] meldet einen ungültigen Wert in `SILERO_DEVICE` oder
    /// `metal` ohne verfügbares Metal.
    pub fn load(dir: &Path) -> Result<Self> {
        let choice = match std::env::var("SILERO_DEVICE") {
            Ok(value) => value.parse()?,
            Err(_) => DeviceChoice::Auto,
        };
        Self::load_with_choice(dir, choice)
    }

    /// Lädt das Modell auf das Gerät, das `choice` benennt. Anders als
    /// [`Silero::load`] liest das nicht `SILERO_DEVICE`.
    ///
    /// # Errors
    ///
    /// Wie [`Silero::load`]; [`DeviceChoice::Metal`] gibt [`Error::Input`] zurück,
    /// wenn Metal nicht verfügbar ist.
    pub fn load_with_choice(dir: &Path, choice: DeviceChoice) -> Result<Self> {
        Self::load_impl(dir, &choice.resolve()?, true)
    }

    /// Lädt das Modell auf ein explizit gewähltes Gerät, etwa `&Device::Cpu` für einen
    /// Vergleich mit Metal oder für Umgebungen ohne Metal. Akzentor und
    /// Homograph-BERT laufen unabhängig davon auf der CPU.
    ///
    /// # Errors
    ///
    /// Gibt dieselben Fehler wie [`Silero::load`] zurück; ein fehlschlagender
    /// Geräte-Operator erscheint als [`Error::Model`] oder [`Error::Pipeline`].
    pub fn load_with_device(dir: &Path, device: &Device) -> Result<Self> {
        Self::load_impl(dir, device, true)
    }

    /// Wie [`Silero::load`], aber ohne `stress_dict.tsv`. Nur für die Golden-Tests:
    /// die Python-Referenz kennt das optionale Wörterbuch nicht.
    #[cfg(feature = "test-util")]
    pub fn load_reference(dir: &Path) -> Result<Self> {
        Self::load_impl(dir, &default_device(), false)
    }

    fn load_impl(dir: &Path, device: &Device, with_dictionary: bool) -> Result<Self> {
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
                accentor: Accentor::load(dir, with_dictionary)?,
                // Der Homograph-BERT ist klein und auf der CPU messbar schneller
                // (1,9 ms/Zeile vs. 2,4 ms/Zeile auf Metal). Auf Metal spart das
                // außerdem einen GPU-Sync vor der Akustik.
                homosolver: HomoSolver::load(dir, &Device::Cpu, bert, tokenizer)?,
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

    /// Betont einen Chunk ohne Synthese: Homograph-Löser, Ausnahmen und optionales
    /// `stress_dict.tsv` aus dem Modellverzeichnis, danach der Akzentor. Die Stimme
    /// entscheidet nur, ob ein paketgebundener ukrainischer Akzentor läuft.
    ///
    /// Für längeren Text zuerst mit [`split_chunks`] teilen; die Betonung ist wie bei
    /// [`Silero::synthesize_text`] chunkweise.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Input`] zurück, wenn der Text keine sprechbaren Zeichen enthält oder
    /// `voice` unbekannt ist, und [`Error::Pipeline`], wenn der Homograph-Löser fehlschlägt.
    pub fn accent_text(&self, text: &str, voice: &str) -> Result<String> {
        let prepared = self.prepare_text(text);
        if !prepared.has_text {
            return Err(Error::Input("text is empty".into()));
        }
        let voice = self.resolve_voice(voice)?;
        let accented = self.accent(&prepared.sentence, voice)?;
        Ok(accented.accented.iter().collect())
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
    /// Auf CPU und Metal laufen mehrere Chunks parallel (höchstens
    /// [`MAX_PARALLEL_CHUNKS`] Threads). Auf Metal sind die GPU-Kernel in der
    /// Command-Queue weiterhin FIFO; der Gewinn kommt aus überlappender Host-Arbeit
    /// und dem Füllen von Dispatch-Lücken. Das Ergebnis ist in beiden Fällen
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
        let parallel = chunks.len() > 1;
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

    fn readback_stage(&self, spectrum: &Spectrum) -> Result<SpectrumData> {
        self.head.readback(spectrum).map_err(stage("istft"))
    }

    fn overlap_add_stage(&self, data: &SpectrumData) -> Result<Vec<f32>> {
        self.head.overlap_add(data).map_err(stage("istft"))
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
        // Die Sperre deckt alle Geräte-Stufen bis zum Readback ab; die ISTFT danach
        // läuft ohne Sperre und überlappt mit dem nächsten Chunk.
        let data = {
            let _metal = metal_guard(&self.device);
            let features = {
                let output = self.acoustic_stage(&ids, voice)?;
                self.features_stage(&output.mel)?
            };
            let spectrum = self.spectrum_stage(&features)?;
            drop(features);
            self.readback_stage(&spectrum)?
        };
        self.overlap_add_stage(&data)
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
        // Der Rest greift auf das Gerät zu (auch die Tensor-Sichten unten): Sperre bis zum Ende.
        let _metal = metal_guard(&self.device);
        let output = self.acoustic_stage(&ids, voice)?;
        let features = self.features_stage(&output.mel)?;
        let spectrum = self.spectrum_stage(&features)?;
        let data = self.readback_stage(&spectrum)?;
        let audio = self.overlap_add_stage(&data)?;
        // Die Stufen liefern `[1, 1201, T]`; die Transposition ist nur eine Sicht ohne Kopie.
        // Der Metal-Spektrum-Graph spart das `minimum` und begrenzt erst in der ISTFT;
        // die Debug-Sicht muss aber den Modellvertrag `min(exp(x), 100)` zeigen.
        let mag = spectrum
            .mag
            .minimum(100f32)
            .map_err(stage("vocoder-head"))?;
        let stage_view = |tensor: &Tensor| tensor.transpose(1, 2).map_err(stage("vocoder-head"));
        let (mag, cos, sin) = (
            stage_view(&mag)?,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_choice_parses_case_insensitive() {
        for (text, want) in [
            ("auto", DeviceChoice::Auto),
            ("CPU", DeviceChoice::Cpu),
            (" Metal ", DeviceChoice::Metal),
        ] {
            assert_eq!(
                text.parse::<DeviceChoice>().ok(),
                Some(want),
                "Eingabe {text:?}"
            );
        }
    }

    #[test]
    fn device_choice_rejects_unknown() {
        assert!(matches!(
            "tpu".parse::<DeviceChoice>(),
            Err(Error::Input(_))
        ));
        assert!(matches!("".parse::<DeviceChoice>(), Err(Error::Input(_))));
        assert_eq!(DeviceChoice::default(), DeviceChoice::Auto);
    }
}
