//! Silero v5_ru nativ in Rust: Akzentor, Homograph-Löser, FastPitch-Netze und
//! ISTFT-Vocoder in `candle`.

pub mod acoustic;
pub mod config;
pub mod text;
pub mod vocoder;

use std::collections::HashSet;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use acoustic::forward::Acoustic;
use config::Config;
use text::accentor::Accentor;
use text::homosolver::HomoSolver;
use text::normalize::{PreparedText, prepare_text_input};
use vocoder::convnext::Backbone;
use vocoder::istft::Head;

pub const SAMPLE_RATE: u32 = 48_000;

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
    pub fn load(dir: &Path) -> Result<Self, String> {
        let config = Config::load(&dir.join("config.json"))?;
        let device = Device::Cpu;
        let weights = dir.join("acoustic.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, &device)
                .map_err(|e| format!("{}: {e}", weights.display()))?
        };
        let acoustic = Acoustic::load(vb.clone(), &device).map_err(|e| format!("acoustic: {e}"))?;
        let backbone =
            Backbone::load(vb.pp("vocoder.backbone")).map_err(|e| format!("vocoder: {e}"))?;
        let window = candle_core::safetensors::load(dir.join("istft_window.safetensors"), &device)
            .map_err(|e| format!("istft_window.safetensors: {e}"))?
            .remove("window")
            .ok_or("istft_window.safetensors: window fehlt")?
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| format!("istft_window.safetensors: {e}"))?;
        if window.len() != config.win_length {
            return Err(format!(
                "ISTFT-Fenster hat {} statt {} Werte",
                window.len(),
                config.win_length
            ));
        }
        let head = Head::load(
            vb.pp("vocoder.head"),
            window,
            config.n_fft,
            config.hop_length,
            config.win_length,
        )
        .map_err(|e| format!("vocoder.head: {e}"))?;
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

    /// `apply_tts` für einen Chunk; der Harness teilt Sätze und fügt Pausen ein.
    pub fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<f32>, String> {
        Ok(self.run_stages(text, voice)?.audio)
    }

    /// Die gemeinsame Pipeline für `synthesize` und die Golden-Tests.
    fn run_stages(&self, text: &str, voice: &str) -> Result<Stages, String> {
        let prepared = self.prepare(text);
        if !prepared.has_text {
            return Err("Text is empty".into());
        }
        let homo = self
            .homosolver
            .solve(&prepared.sentence, true, true, true)?;
        let accented =
            self.accentor
                .accent(&homo, true, true, true, &HashSet::new(), &HashSet::new());
        let ids = self.config.sequence(&accented)?;
        let speaker = self.config.speaker_id(voice)?;
        let output = self
            .acoustic
            .forward(&ids, speaker)
            .map_err(|e| format!("Akustik: {e}"))?;
        let features = self
            .backbone
            .forward(&output.mel)
            .map_err(|e| format!("Vocoder: {e}"))?;
        let (mag, cos, sin) = self
            .head
            .forward(&features)
            .map_err(|e| format!("Vocoder-Head: {e}"))?;
        let audio = self
            .head
            .istft(&mag, &cos, &sin)
            .map_err(|e| format!("ISTFT: {e}"))?;
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

    /// Nur für Golden-Tests: alle Zwischenstände einer Stimme.
    pub fn debug_stages(&self, text: &str, voice: &str) -> Result<Stages, String> {
        self.run_stages(text, voice)
    }

    /// Nur für Golden-Tests: Homosolver-Logits pro Homograph.
    pub fn homosolver_logits(&self, text: &str) -> Result<Option<Tensor>, String> {
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
