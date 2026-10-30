//! `SimpleBertTokenizer` aus dem Silero-v5_ru-Paket.
//!
//! Vorbedingung: Der Text ist bereits vorbereitet, also durch `prepare_text_input`
//! gelaufen oder mit den Homograph-Markierungen versehen. Er enthält nur die Zeichen,
//! die das Modell kennt: kyrillische Buchstaben in Kleinschreibung, ASCII-Satzzeichen
//! und wenige Striche wie `–`. Der Tokenizer schreibt nicht klein, normalisiert keine
//! Unicode-Formen und trennt keine CJK-Zeichen ab. Bei anderem Eingabetext weicht er
//! vom Original ab.
//!
//! `model_max_length` aus `config.json` wird nicht angewendet, es gibt keine Kürzung.
//! Die Länge begrenzt der Aufrufer: `lib.rs` teilt Text in Chunks von höchstens
//! `CHUNK_LIMIT` Codepoints, damit die Sequenz unter den Positionsgrenzen bleibt.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::config::TokenizerConfig;
use crate::error::io_at;
use crate::{Error, Result};

/// Token für Wörter, die nicht zerlegt werden können, und für zu lange Wörter.
const UNK_TOKEN: &str = "[UNK]";

/// Wörter mit mehr Zeichen als dieser Grenze werden als `[UNK]` gezählt. Das ist der
/// Standardwert von `max_input_chars_per_word` im BERT-Tokenizer.
const MAX_WORD_CHARS: usize = 100;

pub struct BertTokenizer {
    vocab: HashMap<String, u32>,
    never_split: HashSet<String>,
    /// ID von [`UNK_TOKEN`] aus dem Vokabular, einmal beim Laden bestimmt.
    unk_id: u32,
    config: TokenizerConfig,
}

impl BertTokenizer {
    /// Lädt `bert_vocab.txt` aus `dir`. `vocab_size` ist `config.bert.vocab_size`.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn die Datei fehlt, ein Token doppelt vorkommt,
    /// die Anzahl der Tokens nicht zu `vocab_size` passt oder `[UNK]` fehlt.
    pub fn load(dir: &Path, config: TokenizerConfig, vocab_size: usize) -> Result<Self> {
        let path = dir.join("bert_vocab.txt");
        let text = std::fs::read_to_string(&path).map_err(io_at(path.clone()))?;
        let mut vocab = HashMap::with_capacity(vocab_size);
        for (id, token) in text.lines().enumerate() {
            if vocab.insert(token.to_string(), id as u32).is_some() {
                return Err(Error::Model(format!(
                    "{}: doppeltes Token {token:?} in Zeile {}",
                    path.display(),
                    id + 1
                )));
            }
        }
        if vocab.len() != vocab_size {
            return Err(Error::Model(format!(
                "{}: {} statt {vocab_size} Tokens",
                path.display(),
                vocab.len()
            )));
        }
        let unk_id = vocab.get(UNK_TOKEN).copied().ok_or_else(|| {
            Error::Model(format!(
                "{}: {UNK_TOKEN} fehlt im Vokabular",
                path.display()
            ))
        })?;
        Ok(Self {
            vocab,
            never_split: config.never_split.iter().cloned().collect(),
            unk_id,
            config,
        })
    }

    /// Konfiguration der Sondertoken, aus `config.json`.
    pub(crate) fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Kodiert vorbereiteten Text als `[CLS] ... [SEP]`-Sequenz von Vokabular-IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let tokens = self.basic_tokenize(text);
        let mut ids = vec![self.config.cls_token_id];
        ids.extend(self.wordpiece(&tokens));
        ids.push(self.config.sep_token_id);
        ids
    }

    fn basic_tokenize(&self, text: &str) -> Vec<String> {
        let mut cleaned = String::with_capacity(text.len());
        for ch in text.chars() {
            if ch == '\0' || ch == '\u{fffd}' {
                continue;
            }
            if ch.is_control() && !matches!(ch, '\t' | '\n' | '\r') {
                continue;
            }
            if ch.is_whitespace() {
                cleaned.push(' ');
            } else {
                cleaned.push(ch);
            }
        }
        let mut split_tokens = Vec::new();
        for token in cleaned.split_whitespace() {
            if self.never_split.contains(token) {
                split_tokens.push(token.to_string());
            } else {
                split_tokens.extend(run_split_on_punc(token));
            }
        }
        split_tokens
    }

    /// WordPiece-Zerlegung. Liefert direkt Vokabular-IDs; jede Ausgabe-ID stammt aus
    /// dem Vokabular, unbekannte Wörter ergeben `unk_id`.
    fn wordpiece(&self, tokens: &[String]) -> Vec<u32> {
        let mut output = Vec::with_capacity(tokens.len());
        // Ein Puffer für alle Teilwörter. `clear` behält die Kapazität, daher
        // wird pro Versuch nichts neu allokiert.
        let mut piece = String::new();
        for token in tokens {
            let chars: Vec<char> = token.chars().collect();
            if chars.len() > MAX_WORD_CHARS {
                output.push(self.unk_id);
                continue;
            }
            let mut start = 0;
            let mut sub_ids = Vec::new();
            let mut bad = false;
            while start < chars.len() {
                let mut end = chars.len();
                let mut found: Option<u32> = None;
                while start < end {
                    piece.clear();
                    if start > 0 {
                        piece.push_str("##");
                    }
                    piece.extend(&chars[start..end]);
                    if let Some(id) = self.vocab.get(piece.as_str()) {
                        found = Some(*id);
                        break;
                    }
                    end -= 1;
                }
                let Some(id) = found else {
                    bad = true;
                    break;
                };
                sub_ids.push(id);
                start = end;
            }
            if bad {
                output.push(self.unk_id);
            } else {
                output.extend(sub_ids);
            }
        }
        output
    }
}

fn run_split_on_punc(token: &str) -> Vec<String> {
    let mut output: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in token.chars() {
        if is_punctuation(ch) {
            if !current.is_empty() {
                output.push(std::mem::take(&mut current));
            }
            output.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        output.push(current);
    }
    output
}

fn is_punctuation(ch: char) -> bool {
    let cp = ch as u32;
    if (33..=47).contains(&cp)
        || (58..=64).contains(&cp)
        || (91..=96).contains(&cp)
        || (123..=126).contains(&cp)
    {
        return true;
    }
    // Unicode-Kategorie P nach dem Zeichenfilter: – (Pd) und … (Po). Alle
    // anderen P-Zeichen wurden vorher entfernt.
    matches!(ch, '\u{2013}' | '\u{2026}')
}
