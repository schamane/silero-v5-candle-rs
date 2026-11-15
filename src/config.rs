//! Konfiguration aus `config.json` plus die daraus abgeleiteten Tabellen.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::Deserialize;

use crate::error::io_at;
use crate::{Error, Result};

/// Obergrenze der Positional-Encoding-Tabellen; gilt für alle v5-Pakete.
pub(crate) const MAX_POSITIONS: usize = 5000;

/// Werte, die `scripts/export_silero.py` aus dem TorchScript-Paket liest.
///
/// Die Lookup-Tabellen entstehen in [`Config::load`]; ein direktes `serde`-Parsen
/// lässt sie leer.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Config {
    pub symbols: String,
    pub symbol_to_id: HashMap<String, u32>,
    pub speakers: HashMap<String, u32>,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub max_positions: usize,
    /// Anzahl der Zeilen in den Speaker-Embeddings. Fehlt im Feld, wird sie aus
    /// der größten Sprecher-ID abgeleitet; `v5_cis_ext` hat eine Lücke in den IDs.
    #[serde(default)]
    pub speaker_rows: Option<usize>,
    /// Nur `v5_ru` hat Akzentor und Homograph-Löser. `v5_6_ru` wird nicht unterstützt.
    #[serde(default)]
    bert: Option<BertConfig>,
    #[serde(default)]
    tokenizer: Option<TokenizerConfig>,
    /// Ukrainischer Akzentor (`v5_cis_base_nostress`).
    #[serde(default)]
    accentor: Option<AccentorConfig>,
    /// Aus `bert`, `tokenizer` und `accentor` abgeleitet; [`Config::load`] verschiebt
    /// die drei Felder hierher.
    #[serde(skip)]
    stress: StressConfig,
    /// Einzelzeichen-Schlüssel aus `symbol_to_id`, für `symbol_id` ohne Allokation.
    #[serde(skip)]
    char_ids: HashMap<char, u32>,
    /// `symbols[3:]` plus `^`, für `prepare_text_input`.
    #[serde(skip)]
    allowed: HashSet<char>,
    /// Zeilen der Symbol-Embeddings (`max(symbol_to_id) + 1`).
    #[serde(skip)]
    symbol_count: usize,
    /// Zeilen der Speaker-Embeddings.
    #[serde(skip)]
    speaker_count: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct BertConfig {
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
    #[expect(
        dead_code,
        reason = "Pflichtfeld des config.json-Schemas, wird aber nicht gelesen"
    )]
    pub position_embedding_type: String,
    pub use_cache: bool,
    pub classifier_dropout: Option<f64>,
    pub model_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TokenizerConfig {
    pub never_split: Vec<String>,
    #[expect(
        dead_code,
        reason = "Pflichtfeld des config.json-Schemas, wird aber nicht gelesen"
    )]
    pub unk_token_id: u32,
    pub sep_token_id: u32,
    pub pad_token_id: u32,
    pub cls_token_id: u32,
    pub homo_start_id: u32,
    pub homo_end_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AccentorConfig {
    /// Sprachcode des Akzentors; derzeit nur `ukr`.
    pub lang: String,
}

/// Betonungskonfiguration eines Pakets; [`Config::load`] lässt nur gültige Kombinationen zu.
#[derive(Debug, Clone, Default)]
#[expect(
    clippy::large_enum_variant,
    reason = "eine Konfiguration je Paket; Boxen spart hier nichts"
)]
pub(crate) enum StressConfig {
    /// `v5_ru`: BERT-Homograph-Löser plus russischer Akzentor.
    Ru {
        bert: BertConfig,
        tokenizer: TokenizerConfig,
    },
    /// `v5_cis_base_nostress`: ukrainischer Akzentor.
    Uk(AccentorConfig),
    /// `v5_cis_ext`: der Aufrufer setzt die Betonung selbst.
    #[default]
    None,
}

impl Config {
    /// Liest `config.json`, prüft es gegen die feste Architektur und baut die Lookup-Tabellen.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn die Datei nicht gelesen oder geparst werden kann,
    /// wenn Symbole oder Sprecher nicht genau einmal auf `0..COUNT` abgebildet sind
    /// oder wenn Abtastrate beziehungsweise `max_positions` nicht passen.
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(io_at(path))?;
        let mut config: Self =
            serde_json::from_str(&text).map_err(|e| Error::model(path.display().to_string(), e))?;
        let (char_ids, symbol_count) = config.validate_symbols(path)?;
        let speaker_count = config.validate_speakers(path)?;
        config.stress = config.take_stress(path, &char_ids)?;
        config.validate_audio(path)?;
        config.char_ids = char_ids;
        // Die v5_cis-Symboltabelle beginnt mit `|!'`; ein blindes `skip(3)` würde
        // `!` und `'` verschlucken. `_`, `~` und `|` bleiben Steuerzeichen.
        config.allowed = config
            .symbols
            .chars()
            .filter(|ch| !matches!(ch, '_' | '~' | '|'))
            .chain(['^'])
            .collect();
        config.symbol_count = symbol_count;
        config.speaker_count = speaker_count;
        Ok(config)
    }

    /// Prüft, dass `symbols` und `symbol_to_id` zusammenpassen und die IDs `0..COUNT`
    /// genau einmal belegen. Liefert die Einzelzeichen-Tabelle und `COUNT`.
    fn validate_symbols(&self, path: &Path) -> Result<(HashMap<char, u32>, usize)> {
        if self.symbols.is_empty() {
            return Err(Error::model_msg(format!(
                "{}: keine Symbole",
                path.display()
            )));
        }
        let char_ids: HashMap<char, u32> = self
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
        if self.symbols.chars().any(|ch| !char_ids.contains_key(&ch)) {
            return Err(Error::model_msg(format!(
                "{}: Symbol ohne ID",
                path.display()
            )));
        }
        let symbol_count = self
            .symbol_to_id
            .values()
            .copied()
            .max()
            .map_or(0, |max| max as usize + 1);
        if symbol_count != self.symbol_to_id.len() {
            return Err(Error::model_msg(format!(
                "{}: {} Symbol-IDs, aber IDs bis {}",
                path.display(),
                self.symbol_to_id.len(),
                symbol_count.saturating_sub(1)
            )));
        }
        check_dense_ids(
            path,
            "Symbol",
            self.symbol_to_id.values().copied().collect(),
            symbol_count,
        )?;
        Ok((char_ids, symbol_count))
    }

    /// Prüft Sprecher-IDs auf Lücken-Deckung und Duplikate. Liefert die Zeilenzahl der
    /// Speaker-Embeddings.
    fn validate_speakers(&self, path: &Path) -> Result<usize> {
        let speaker_max = self
            .speakers
            .values()
            .copied()
            .max()
            .map_or(0, |max| max as usize + 1);
        let speaker_count = self.speaker_rows.unwrap_or(speaker_max);
        if speaker_count < speaker_max || speaker_count == 0 {
            return Err(Error::model_msg(format!(
                "{}: speaker_rows {speaker_count} deckt Sprecher-IDs bis {} nicht ab",
                path.display(),
                speaker_max.saturating_sub(1)
            )));
        }
        let mut seen = HashSet::with_capacity(self.speakers.len());
        if let Some(duplicate) = self.speakers.values().copied().find(|id| !seen.insert(*id)) {
            return Err(Error::model_msg(format!(
                "{}: doppelte Sprecher-ID {duplicate}",
                path.display()
            )));
        }
        Ok(speaker_count)
    }

    /// Verschiebt `bert`, `tokenizer` und `accentor` in eine gültige [`StressConfig`].
    fn take_stress(&mut self, path: &Path, char_ids: &HashMap<char, u32>) -> Result<StressConfig> {
        let russian = match (self.bert.take(), self.tokenizer.take()) {
            (Some(bert), Some(tokenizer)) => Some((bert, tokenizer)),
            (None, None) => None,
            _ => {
                return Err(Error::model_msg(format!(
                    "{}: bert und tokenizer müssen gemeinsam vorhanden sein",
                    path.display()
                )));
            }
        };
        let Some(accentor) = self.accentor.take() else {
            return Ok(
                russian.map_or(StressConfig::None, |(bert, tokenizer)| StressConfig::Ru {
                    bert,
                    tokenizer,
                }),
            );
        };
        if accentor.lang != "ukr" {
            return Err(Error::model_msg(format!(
                "{}: Akzentor-Sprache {:?} statt \"ukr\"",
                path.display(),
                accentor.lang
            )));
        }
        if russian.is_some() {
            return Err(Error::model_msg(format!(
                "{}: russischer und ukrainischer Akzentor schließen sich aus",
                path.display()
            )));
        }
        if !char_ids.contains_key(&'+') {
            return Err(Error::model_msg(format!(
                "{}: Akzentor braucht '+' im Symbolalphabet",
                path.display()
            )));
        }
        Ok(StressConfig::Uk(accentor))
    }

    /// Prüft Abtastrate und Positions-Obergrenze gegen die feste Architektur.
    fn validate_audio(&self, path: &Path) -> Result<()> {
        if self.sample_rate != crate::SAMPLE_RATE {
            return Err(Error::model_msg(format!(
                "{}: sample_rate {} statt {}",
                path.display(),
                self.sample_rate,
                crate::SAMPLE_RATE
            )));
        }
        if self.max_positions != MAX_POSITIONS {
            return Err(Error::model_msg(format!(
                "{}: max_positions {} statt {MAX_POSITIONS}",
                path.display(),
                self.max_positions
            )));
        }
        Ok(())
    }

    /// Betonungsmodell des Pakets, aus `bert`/`tokenizer`/`accentor` abgeleitet.
    pub(crate) fn stress(&self) -> &StressConfig {
        &self.stress
    }

    /// Zeilen der Symbol-Embeddings (`max(symbol_to_id) + 1`).
    pub(crate) fn symbol_count(&self) -> usize {
        self.symbol_count
    }

    /// Zeilen der Speaker-Embeddings.
    pub(crate) fn speaker_count(&self) -> usize {
        self.speaker_count
    }

    /// `true`, wenn das Paket die Betonung selbst setzt: russischer Akzentor
    /// (`v5_ru`) oder ukrainischer Akzentor (`v5_cis_base_nostress`, optional
    /// auch `v5_cis_ext`).
    pub(crate) fn has_accentor(&self) -> bool {
        !matches!(self.stress, StressConfig::None)
    }

    /// `true`, wenn der Akzentor auf die Sprache der Stimme passt. Die Sprache
    /// steht vor dem ersten `_` (`ukr_igor` → `ukr`). Der ukrainische Akzentor
    /// darf z. B. keine kaukasischen Stimmen im CIS-Paket verändern.
    pub(crate) fn accentor_matches(&self, voice: &str) -> bool {
        match &self.stress {
            StressConfig::Uk(accentor) => voice.split('_').next() == Some(accentor.lang.as_str()),
            StressConfig::Ru { .. } | StressConfig::None => false,
        }
    }

    /// `true`, wenn diese Stimme beim Laden die Betonung selbst setzt.
    pub(crate) fn has_accentor_for(&self, voice: &str) -> bool {
        matches!(self.stress, StressConfig::Ru { .. }) || self.accentor_matches(voice)
    }

    /// IDs für `|text~`, wie `preprocess_tacotron`.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn ein Zeichen kein Symbol aus `config.json` ist.
    pub(crate) fn sequence(&self, text: &[char]) -> Result<Vec<u32>> {
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
    pub(crate) fn symbol_id(&self, ch: char) -> Result<u32> {
        self.char_ids
            .get(&ch)
            .copied()
            .ok_or_else(|| Error::Input(format!("unbekanntes Symbol {ch:?}")))
    }

    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn `voice` kein Sprecher aus `config.json` ist.
    pub(crate) fn speaker_id(&self, voice: &str) -> Result<u32> {
        self.speakers
            .get(voice)
            .copied()
            .ok_or_else(|| Error::Input(format!("unbekannte Stimme {voice:?}")))
    }

    /// Zeichen, die `prepare_text_input` behält: `symbols[3:]` plus `^`.
    pub(crate) fn allowed_chars(&self) -> &HashSet<char> {
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
            return Err(Error::model_msg(format!(
                "{}: doppelte {label}-ID {id}",
                path.display()
            )));
        }
    }
    let expected: HashSet<u32> = (0..count as u32).collect();
    if seen != expected {
        let mut found: Vec<u32> = seen.into_iter().collect();
        found.sort_unstable();
        return Err(Error::model_msg(format!(
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
