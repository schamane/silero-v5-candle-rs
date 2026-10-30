//! Konfiguration aus `config.json` plus die daraus abgeleiteten Tabellen.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

/// Feste Architektur des Silero-v5_ru-Pakets; `config.json` muss dazu passen.
pub const SYMBOL_COUNT: usize = 47;
pub const SPEAKER_COUNT: usize = 5;
pub const MAX_POSITIONS: usize = 5000;

/// Werte, die `scripts/export_silero.py` aus dem TorchScript-Paket liest.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub symbols: String,
    pub symbol_to_id: HashMap<String, u32>,
    pub speakers: HashMap<String, u32>,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub max_positions: usize,
    pub bert: BertConfig,
    pub tokenizer: TokenizerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    pub position_embedding_type: String,
    pub use_cache: bool,
    pub classifier_dropout: Option<f64>,
    pub model_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerConfig {
    pub never_split: Vec<String>,
    pub unk_token_id: u32,
    pub sep_token_id: u32,
    pub pad_token_id: u32,
    pub cls_token_id: u32,
    pub homo_start_id: u32,
    pub homo_end_id: u32,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let config: Self =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        if config.symbols.chars().count() != SYMBOL_COUNT {
            return Err(format!(
                "{}: erwartet {SYMBOL_COUNT} Symbole",
                path.display()
            ));
        }
        if config.symbol_to_id.len() != SYMBOL_COUNT {
            return Err(format!(
                "{}: {} Symbol-IDs statt {SYMBOL_COUNT}",
                path.display(),
                config.symbol_to_id.len()
            ));
        }
        if config
            .symbols
            .chars()
            .any(|ch| !config.symbol_to_id.contains_key(&ch.to_string()))
        {
            return Err(format!("{}: Symbol ohne ID", path.display()));
        }
        if config
            .symbol_to_id
            .values()
            .any(|id| *id as usize >= SYMBOL_COUNT)
        {
            return Err(format!(
                "{}: Symbol-ID außerhalb 0..{SYMBOL_COUNT}",
                path.display()
            ));
        }
        if config.speakers.len() != SPEAKER_COUNT {
            return Err(format!(
                "{}: {} Sprecher statt {SPEAKER_COUNT}",
                path.display(),
                config.speakers.len()
            ));
        }
        if config
            .speakers
            .values()
            .any(|id| *id as usize >= SPEAKER_COUNT)
        {
            return Err(format!(
                "{}: Sprecher-ID außerhalb 0..{SPEAKER_COUNT}",
                path.display()
            ));
        }
        if config.sample_rate != crate::SAMPLE_RATE {
            return Err(format!(
                "{}: sample_rate {} statt {}",
                path.display(),
                config.sample_rate,
                crate::SAMPLE_RATE
            ));
        }
        if config.max_positions != MAX_POSITIONS {
            return Err(format!(
                "{}: max_positions {} statt {MAX_POSITIONS}",
                path.display(),
                config.max_positions
            ));
        }
        Ok(config)
    }

    /// IDs für `|text~`, wie `preprocess_tacotron`.
    pub fn sequence(&self, text: &[char]) -> Result<Vec<u32>, String> {
        let mut ids = Vec::with_capacity(text.len() + 2);
        ids.push(self.symbol_id('|')?);
        for ch in text {
            ids.push(self.symbol_id(*ch)?);
        }
        ids.push(self.symbol_id('~')?);
        Ok(ids)
    }

    pub fn symbol_id(&self, ch: char) -> Result<u32, String> {
        self.symbol_to_id
            .get(&ch.to_string())
            .copied()
            .ok_or_else(|| format!("unbekanntes Symbol {ch:?}"))
    }

    pub fn speaker_id(&self, voice: &str) -> Result<u32, String> {
        self.speakers
            .get(voice)
            .copied()
            .ok_or_else(|| format!("unbekannte Stimme {voice:?}"))
    }

    /// Zeichen, die `prepare_text_input` behält: `symbols[3:]` plus `^`.
    pub fn allowed_chars(&self) -> Vec<char> {
        let mut allowed: Vec<char> = self.symbols.chars().skip(3).collect();
        allowed.push('^');
        allowed
    }
}
