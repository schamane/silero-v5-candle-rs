//! Silero v5_ru nativ in Rust: Akzentor, Homograph-Löser, FastPitch-Netze und
//! ISTFT-Vocoder in `candle`.

pub mod acoustic;
pub mod config;
mod error;
pub mod text;
pub mod vocoder;

use std::collections::HashSet;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use acoustic::forward::Acoustic;
use config::Config;
use error::stage;
use text::accentor::{AccentOptions, Accentor};
use text::homosolver::{HomoSolver, SolveOptions};
use text::normalize::{PreparedText, prepare_text_input, split_chunks};
use vocoder::convnext::Backbone;
use vocoder::istft::Head;

pub use error::{Error, Result};

pub const SAMPLE_RATE: u32 = 48_000;

/// Maximale Chunk-Länge in Codepoints; die Satzteilung folgt `SPEC-STE.md`
/// (`split_chunks(text, 400)`) und hält Text unter den Positionsgrenzen von BERT und Akustik.
pub const CHUNK_LIMIT: usize = 400;

pub struct Silero {
    config: Config,
    acoustic: Acoustic,
    accentor: Accentor,
    homosolver: HomoSolver,
    backbone: Backbone,
    head: Head,
    device: Device,
}

impl Silero {
    /// Lädt Konfiguration, Gewichte und Wörterbücher aus `dir`.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Io`] zurück, wenn eine Datei nicht gelesen werden kann, und
    /// [`Error::Model`], wenn eine Datei fehlt, ungültig ist oder nicht zur Architektur passt.
    pub fn load(dir: &Path) -> Result<Self> {
        let config = Config::load(&dir.join("config.json"))?;
        let device = Device::Cpu;
        let weights = dir.join("acoustic.safetensors");
        // SAFETY: Die Modelldateien in `dir` dürfen während des Mappings nicht geändert
        // oder ersetzt werden. Sonst ist der Inhalt des gemappten Speichers undefiniert.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &device)
                .map_err(|e| Error::Model(format!("{}: {e}", weights.display())))?
        };
        let acoustic = Acoustic::load(vb.clone(), &device)
            .map_err(|e| Error::Model(format!("acoustic: {e}")))?;
        let backbone = Backbone::load(vb.pp("vocoder.backbone"))
            .map_err(|e| Error::Model(format!("vocoder: {e}")))?;
        let window = candle_core::safetensors::load(dir.join("istft_window.safetensors"), &device)
            .map_err(|e| Error::Model(format!("istft_window.safetensors: {e}")))?
            .remove("window")
            .ok_or_else(|| Error::Model("istft_window.safetensors: window fehlt".into()))?
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| Error::Model(format!("istft_window.safetensors: {e}")))?;
        if window.len() != config.win_length {
            return Err(Error::Model(format!(
                "ISTFT-Fenster hat {} statt {} Werte",
                window.len(),
                config.win_length
            )));
        }
        let head = Head::load(
            vb.pp("vocoder.head"),
            window,
            config.n_fft,
            config.hop_length,
            config.win_length,
        )
        .map_err(|e| Error::Model(format!("vocoder.head: {e}")))?;
        let accentor = Accentor::load(dir, &device)?;
        let homosolver =
            HomoSolver::load(dir, &device, config.bert.clone(), config.tokenizer.clone())?;
        Ok(Self {
            config,
            acoustic,
            accentor,
            homosolver,
            backbone,
            head,
            device,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    pub fn prepare(&self, text: &str) -> PreparedText {
        prepare_text_input(&self.config, text)
    }

    /// `apply_tts` für einen Chunk; Satzteilung und Pausen liegen beim Aufrufer.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Input`] zurück, wenn der Text keine sprechbaren Zeichen enthält oder
    /// `voice` unbekannt ist, und [`Error::Pipeline`], wenn eine Stufe fehlschlägt.
    pub fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<f32>> {
        Ok(self.run_stages(text, voice)?.audio)
    }

    /// Synthetisiert beliebig langen Text: zerlegt ihn in Chunks von höchstens
    /// [`CHUNK_LIMIT`] Codepoints und hängt die Audiodaten ohne Pausen aneinander.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn ein Chunk nicht synthetisiert werden kann
    /// ([`Error::Input`] oder [`Error::Pipeline`]), oder wenn der Text keine sprechbaren
    /// Zeichen enthält ([`Error::Input`]).
    pub fn synthesize_text(&self, text: &str, voice: &str) -> Result<Vec<f32>> {
        let mut audio = Vec::new();
        for chunk in split_chunks(text, CHUNK_LIMIT) {
            if !self.prepare(&chunk).has_text {
                continue;
            }
            audio.extend(self.synthesize(&chunk, voice)?);
        }
        if audio.is_empty() {
            return Err(Error::Input("text is empty".into()));
        }
        Ok(audio)
    }

    /// Die gemeinsame Pipeline für `synthesize` und die Golden-Tests.
    fn run_stages(&self, text: &str, voice: &str) -> Result<Stages> {
        let prepared = self.prepare(text);
        if !prepared.has_text {
            return Err(Error::Input("text is empty".into()));
        }
        let homo = self
            .homosolver
            .solve(&prepared.sentence, SolveOptions::default())?;
        let accented = self.accentor.accent(
            &homo,
            AccentOptions::default(),
            &HashSet::new(),
            &HashSet::new(),
        );
        let ids = self.config.sequence(&accented)?;
        let speaker = self.config.speaker_id(voice)?;
        let output = self
            .acoustic
            .forward(&ids, speaker)
            .map_err(stage("akustik"))?;
        let features = self
            .backbone
            .forward(&output.mel)
            .map_err(stage("vocoder"))?;
        let (mag, cos, sin) = self
            .head
            .forward(&features)
            .map_err(stage("vocoder-head"))?;
        let audio = self.head.istft(&mag, &cos, &sin).map_err(stage("istft"))?;
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

    /// Nur für die Golden-Tests (nicht Teil der stabilen API): alle Zwischenstände einer Stimme.
    ///
    /// # Errors
    ///
    /// Gibt dieselben Fehler wie [`Silero::synthesize`] zurück.
    #[doc(hidden)]
    pub fn debug_stages(&self, text: &str, voice: &str) -> Result<Stages> {
        self.run_stages(text, voice)
    }

    /// Nur für die Golden-Tests (nicht Teil der stabilen API): Homosolver-Logits pro Homograph.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Model`] oder [`Error::Pipeline`] zurück, wenn der Homosolver fehlschlägt.
    #[doc(hidden)]
    pub fn homosolver_logits(&self, text: &str) -> Result<Option<Tensor>> {
        let prepared = self.prepare(text);
        if !prepared.has_text {
            return Ok(None);
        }
        self.homosolver.logits(&prepared.sentence)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Zwischenstände der Pipeline. Nur für die Golden-Tests (nicht Teil der stabilen API);
/// die Felder sind lesbar, damit die Tests jede Stufe gegen Python vergleichen können.
#[doc(hidden)]
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
