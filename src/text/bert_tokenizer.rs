//! `SimpleBertTokenizer` aus `custom_tokenizers/bert_tokenizer.py`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::config::TokenizerConfig;

pub struct BertTokenizer {
    vocab: HashMap<String, u32>,
    never_split: HashSet<String>,
    pub config: TokenizerConfig,
}

impl BertTokenizer {
    pub fn load(dir: &Path, config: TokenizerConfig) -> Result<Self, String> {
        let path = dir.join("bert_vocab.txt");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut vocab = HashMap::new();
        for (id, token) in text.lines().enumerate() {
            vocab.insert(token.to_string(), id as u32);
        }
        if vocab.len() != 83_830 {
            return Err(format!(
                "{}: {} statt 83830 Tokens",
                path.display(),
                vocab.len()
            ));
        }
        Ok(Self {
            vocab,
            never_split: config.never_split.iter().cloned().collect(),
            config,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = self.basic_tokenize(text);
        tokens = self.wordpiece_tokenize(&tokens);
        let mut ids = Vec::with_capacity(tokens.len() + 2);
        ids.push(self.config.cls_token_id);
        for token in tokens {
            ids.push(
                self.vocab
                    .get(&token)
                    .copied()
                    .unwrap_or(self.config.unk_token_id),
            );
        }
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

    fn wordpiece_tokenize(&self, tokens: &[String]) -> Vec<String> {
        let mut output = Vec::new();
        for token in tokens {
            let chars: Vec<char> = token.chars().collect();
            if chars.len() > 100 {
                output.push("[UNK]".to_string());
                continue;
            }
            let mut start = 0;
            let mut sub_tokens = Vec::new();
            let mut bad = false;
            while start < chars.len() {
                let mut end = chars.len();
                let mut current: Option<String> = None;
                while start < end {
                    let mut substr: String = chars[start..end].iter().collect();
                    if start > 0 {
                        substr = format!("##{substr}");
                    }
                    if self.vocab.contains_key(&substr) {
                        current = Some(substr);
                        break;
                    }
                    end -= 1;
                }
                let Some(substr) = current else {
                    bad = true;
                    break;
                };
                sub_tokens.push(substr);
                start = end;
            }
            if bad {
                output.push("[UNK]".to_string());
            } else {
                output.extend(sub_tokens);
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
