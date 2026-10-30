//! Konfiguration aus `config.json` plus die daraus abgeleiteten Tabellen.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::Deserialize;

use crate::error::io_at;
use crate::{Error, Result};

/// Feste Architektur des Silero-v5_ru-Pakets; `config.json` muss dazu passen.
pub const SYMBOL_COUNT: usize = 47;
pub const SPEAKER_COUNT: usize = 5;
pub const MAX_POSITIONS: usize = 5000;

/// Werte, die `scripts/export_silero.py` aus dem TorchScript-Paket liest.
///
/// Die Lookup-Tabellen entstehen in [`Config::load`]; ein direktes `serde`-Parsen
/// lässt sie leer.
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
    /// Einzelzeichen-Schlüssel aus `symbol_to_id`, für `symbol_id` ohne Allokation.
    #[serde(skip)]
    char_ids: HashMap<char, u32>,
    /// `symbols[3:]` plus `^`, für `prepare_text_input`.
    #[serde(skip)]
    allowed: HashSet<char>,
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
    /// Liest `config.json`, prüft es gegen die feste Architektur und baut die Lookup-Tabellen.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn die Datei nicht gelesen oder geparst werden kann,
    /// wenn Symbole oder Sprecher nicht genau einmal auf `0..COUNT` abgebildet sind
    /// oder wenn Abtastrate beziehungsweise `max_positions` nicht passen.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(io_at(path))?;
        let mut config: Self = serde_json::from_str(&text)
            .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        if config.symbols.chars().count() != SYMBOL_COUNT {
            return Err(Error::Model(format!(
                "{}: erwartet {SYMBOL_COUNT} Symbole",
                path.display()
            )));
        }
        if config.symbol_to_id.len() != SYMBOL_COUNT {
            return Err(Error::Model(format!(
                "{}: {} Symbol-IDs statt {SYMBOL_COUNT}",
                path.display(),
                config.symbol_to_id.len()
            )));
        }
        let char_ids: HashMap<char, u32> = config
            .symbol_to_id
            .iter()
            .filter_map(|(key, &id)| {
                let mut chars = key.chars();
                match (chars.next(), chars.next()) {
                    (Some(ch), None) => Some((ch, id)),
                    _ => None,
                }
            })
            .collect();
        if config.symbols.chars().any(|ch| !char_ids.contains_key(&ch)) {
            return Err(Error::Model(format!("{}: Symbol ohne ID", path.display())));
        }
        check_dense_ids(
            path,
            "Symbol",
            config.symbol_to_id.values().copied().collect(),
            SYMBOL_COUNT,
        )?;
        if config.speakers.len() != SPEAKER_COUNT {
            return Err(Error::Model(format!(
                "{}: {} Sprecher statt {SPEAKER_COUNT}",
                path.display(),
                config.speakers.len()
            )));
        }
        check_dense_ids(
            path,
            "Sprecher",
            config.speakers.values().copied().collect(),
            SPEAKER_COUNT,
        )?;
        if config.sample_rate != crate::SAMPLE_RATE {
            return Err(Error::Model(format!(
                "{}: sample_rate {} statt {}",
                path.display(),
                config.sample_rate,
                crate::SAMPLE_RATE
            )));
        }
        if config.max_positions != MAX_POSITIONS {
            return Err(Error::Model(format!(
                "{}: max_positions {} statt {MAX_POSITIONS}",
                path.display(),
                config.max_positions
            )));
        }
        config.char_ids = char_ids;
        config.allowed = config.symbols.chars().skip(3).chain(['^']).collect();
        Ok(config)
    }

    /// IDs für `|text~`, wie `preprocess_tacotron`.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn ein Zeichen kein Symbol aus `config.json` ist.
    pub fn sequence(&self, text: &[char]) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(text.len() + 2);
        ids.push(self.symbol_id('|')?);
        for ch in text {
            ids.push(self.symbol_id(*ch)?);
        }
        ids.push(self.symbol_id('~')?);
        Ok(ids)
    }

    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn `ch` kein Symbol aus `config.json` ist.
    pub fn symbol_id(&self, ch: char) -> Result<u32> {
        self.char_ids
            .get(&ch)
            .copied()
            .ok_or_else(|| Error::Input(format!("unbekanntes Symbol {ch:?}")))
    }

    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn `voice` kein Sprecher aus `config.json` ist.
    pub fn speaker_id(&self, voice: &str) -> Result<u32> {
        self.speakers
            .get(voice)
            .copied()
            .ok_or_else(|| Error::Input(format!("unbekannte Stimme {voice:?}")))
    }

    /// Zeichen, die `prepare_text_input` behält: `symbols[3:]` plus `^`.
    pub fn allowed_chars(&self) -> &HashSet<char> {
        &self.allowed
    }
}

/// Prüft, dass `ids` jede Zahl aus `0..count` genau einmal enthält.
///
/// Meldet den kleinsten doppelten Wert, sonst die gefundenen IDs, falls sie nicht
/// `0..count` ergeben.
///
/// # Errors
///
/// Gibt einen Fehler zurück, wenn eine ID doppelt vorkommt oder die Menge nicht `0..count` ist.
fn check_dense_ids(path: &Path, label: &str, mut ids: Vec<u32>, count: usize) -> Result<()> {
    ids.sort_unstable();
    let mut seen = HashSet::with_capacity(ids.len());
    for id in ids {
        if !seen.insert(id) {
            return Err(Error::Model(format!(
                "{}: doppelte {label}-ID {id}",
                path.display()
            )));
        }
    }
    let expected: HashSet<u32> = (0..count as u32).collect();
    if seen != expected {
        let mut found: Vec<u32> = seen.into_iter().collect();
        found.sort_unstable();
        return Err(Error::Model(format!(
            "{}: {label}-IDs {found:?} statt 0..{count}",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_ids_accept_any_order() {
        assert!(check_dense_ids(Path::new("c.json"), "Symbol", vec![2, 0, 1], 3).is_ok());
    }

    #[test]
    fn dense_ids_report_duplicate_value() {
        let err = check_dense_ids(Path::new("c.json"), "Symbol", vec![0, 2, 1, 1], 3)
            .expect_err("doppelte ID muss abgelehnt werden")
            .to_string();
        assert!(err.contains("doppelte Symbol-ID 1"), "Fehler: {err}");
    }

    #[test]
    fn dense_ids_report_gap() {
        let err = check_dense_ids(Path::new("c.json"), "Sprecher", vec![0, 1, 5], 3)
            .expect_err("ID außerhalb 0..3 muss abgelehnt werden")
            .to_string();
        assert!(err.contains("[0, 1, 5]"), "Fehler: {err}");
    }
}
