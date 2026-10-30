//! Homograph-Löser: BERT (rubert-tiny2-Form) plus `homo_clf` auf
//! `[Marker-Embedding ; Wort-Mittel]`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertCandleConfig, HiddenAct};

use crate::config::{BertConfig, TokenizerConfig};
use crate::text::bert_tokenizer::BertTokenizer;

pub struct HomoSolver {
    bert: BertModel,
    head0: Linear,
    head1: Linear,
    tokenizer: BertTokenizer,
    homodict: HashMap<String, Vec<String>>,
    device: Device,
    bert_config: BertConfig,
}

impl HomoSolver {
    pub fn load(
        dir: &Path,
        device: &Device,
        bert_config: BertConfig,
        tokenizer_config: TokenizerConfig,
    ) -> Result<Self, String> {
        let config = BertCandleConfig {
            vocab_size: bert_config.vocab_size,
            hidden_size: bert_config.hidden_size,
            num_hidden_layers: bert_config.num_hidden_layers,
            num_attention_heads: bert_config.num_attention_heads,
            intermediate_size: bert_config.intermediate_size,
            hidden_act: match bert_config.hidden_act.as_str() {
                "gelu" => HiddenAct::Gelu,
                "gelu_approximate" => HiddenAct::GeluApproximate,
                "relu" => HiddenAct::Relu,
                other => return Err(format!("unbekannte BERT-Aktivierung {other}")),
            },
            hidden_dropout_prob: bert_config.hidden_dropout_prob,
            max_position_embeddings: bert_config.max_position_embeddings,
            type_vocab_size: bert_config.type_vocab_size,
            initializer_range: bert_config.initializer_range,
            layer_norm_eps: bert_config.layer_norm_eps,
            pad_token_id: bert_config.pad_token_id,
            position_embedding_type: Default::default(),
            use_cache: bert_config.use_cache,
            classifier_dropout: bert_config.classifier_dropout,
            model_type: bert_config.model_type.clone(),
        };
        let path = dir.join("homosolver.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&path), DType::F32, device)
                .map_err(|e| format!("{}: {e}", path.display()))?
        };
        let bert = BertModel::load(vb.pp("bert"), &config)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let head0 = candle_nn::linear(624, 256, vb.pp("homo_clf.0"))
            .map_err(|e| format!("{}: homo_clf.0: {e}", path.display()))?;
        let head1 = candle_nn::linear(256, 1, vb.pp("homo_clf.3"))
            .map_err(|e| format!("{}: homo_clf.3: {e}", path.display()))?;

        let homodict: HashMap<String, Vec<String>> = serde_json::from_str(
            &std::fs::read_to_string(dir.join("homodict.json"))
                .map_err(|e| format!("homodict.json: {e}"))?,
        )
        .map_err(|e| format!("homodict.json: {e}"))?;
        if homodict.len() != 1924 {
            return Err(format!(
                "homodict.json: {} statt 1924 Wörter",
                homodict.len()
            ));
        }
        let tokenizer = BertTokenizer::load(dir, tokenizer_config)?;
        Ok(Self {
            bert,
            head0,
            head1,
            tokenizer,
            homodict,
            device: device.clone(),
            bert_config,
        })
    }

    pub fn homodict(&self) -> &HashMap<String, Vec<String>> {
        &self.homodict
    }

    /// Wörter mit mindestens einem `ё` in einer Variante (für `skip_yo_words`).
    pub fn yo_words(&self) -> std::collections::HashSet<String> {
        self.homodict
            .iter()
            .filter(|(_, variants)| variants.iter().any(|v| v.contains('ё')))
            .map(|(word, _)| word.clone())
            .collect()
    }

    pub fn solve(
        &self,
        sentence: &[char],
        put_stress: bool,
        put_yo: bool,
        stress_single_vowel: bool,
    ) -> Result<Vec<char>, String> {
        if !(put_stress || put_yo) {
            return Ok(sentence.to_vec());
        }
        let matches = self.find_matches(sentence);
        if matches.is_empty() {
            return Ok(sentence.to_vec());
        }

        let (ids_batch, starts, ends) = self.batch(sentence, &matches)?;
        let logits = self.forward(&ids_batch, &starts, &ends)?;
        let probs = candle_nn::ops::sigmoid(&logits).map_err(|e| e.to_string())?;
        let preds: Vec<u32> = values(&probs)?
            .into_iter()
            .map(|value| value.round_ties_even() as u32)
            .collect();

        let mut result = sentence.to_vec();
        let mut offset: i32 = 0;
        for (found, pred) in matches.iter().zip(preds) {
            let start = found.start as i32 + offset;
            let end = found.end as i32 + offset;
            let key = found.word.iter().collect::<String>().to_lowercase();
            let Some(variants) = self.homodict.get(&key) else {
                continue;
            };
            let mut sorted = variants.clone();
            sorted.sort();
            let Some(variant) = sorted.get(pred as usize) else {
                continue;
            };
            let mut word_pred: Vec<char> = if put_yo {
                variant.chars().collect()
            } else {
                variant
                    .chars()
                    .map(|ch| if ch == 'ё' { 'е' } else { ch })
                    .collect()
            };
            let n_vowels = word_pred
                .iter()
                .filter(|ch| crate::text::accentor::is_vowel(**ch))
                .count();
            let Some(stress_idx) = word_pred.iter().position(|ch| *ch == '+') else {
                continue;
            };
            word_pred.retain(|ch| *ch != '+');
            let cased: Vec<char> = found
                .word
                .iter()
                .zip(&word_pred)
                .map(|(original, predicted)| {
                    if original.is_lowercase() {
                        predicted.to_lowercase().next().unwrap_or(*predicted)
                    } else {
                        predicted.to_uppercase().next().unwrap_or(*predicted)
                    }
                })
                .collect();
            let mut replacement = cased;
            if (n_vowels > 1 || stress_single_vowel)
                && put_stress
                && stress_idx <= replacement.len()
            {
                replacement.insert(stress_idx, '+');
                offset += 1;
            }
            if start >= 0 && (end as usize) <= result.len() {
                result.splice(start as usize..end as usize, replacement);
            }
        }
        Ok(result)
    }

    /// Logits für alle Homographen eines Satzes, `None` ohne Treffer.
    pub fn logits(&self, sentence: &[char]) -> Result<Option<Tensor>, String> {
        let matches = self.find_matches(sentence);
        if matches.is_empty() {
            return Ok(None);
        }
        let (ids_batch, starts, ends) = self.batch(sentence, &matches)?;
        self.forward(&ids_batch, &starts, &ends).map(Some)
    }

    fn batch(&self, sentence: &[char], matches: &[Found]) -> Result<Batch, String> {
        let mut ids_batch = Vec::new();
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for found in matches {
            let mark = build_marked(sentence, found);
            let ids = self.tokenizer.encode(&mark.iter().collect::<String>());
            let start = ids
                .iter()
                .position(|id| *id == self.tokenizer.config.homo_start_id)
                .ok_or("HOMO-Startmarker fehlt")?;
            let end = ids
                .iter()
                .position(|id| *id == self.tokenizer.config.homo_end_id)
                .ok_or("HOMO-Endmarker fehlt")?;
            ids_batch.push(ids);
            starts.push(start);
            ends.push(end);
        }
        Ok((ids_batch, starts, ends))
    }

    /// `(?=.*[а-яё])[а-яё+]+` mit IGNORECASE als Zeichenschleife.
    fn find_matches(&self, sentence: &[char]) -> Vec<Found> {
        let mut found = Vec::new();
        let mut i = 0;
        while i < sentence.len() {
            if !is_pattern_char(sentence[i]) {
                i += 1;
                continue;
            }
            let start = i;
            while i < sentence.len() && is_pattern_char(sentence[i]) {
                i += 1;
            }
            let end = i;
            if !sentence[start..].iter().any(|ch| is_lower_cyrillic(*ch)) {
                continue;
            }
            let word = sentence[start..end].to_vec();
            let key = word.iter().collect::<String>().to_lowercase();
            if self.homodict.contains_key(&key) {
                found.push(Found { start, end, word });
            }
        }
        found
    }

    fn forward(
        &self,
        ids_batch: &[Vec<u32>],
        starts: &[usize],
        ends: &[usize],
    ) -> Result<Tensor, String> {
        let batch = ids_batch.len();
        let max_len = ids_batch.iter().map(Vec::len).max().unwrap_or(0);
        let mut flat = vec![self.tokenizer.config.pad_token_id; batch * max_len];
        for (row, ids) in ids_batch.iter().enumerate() {
            flat[row * max_len..row * max_len + ids.len()].copy_from_slice(ids);
        }
        let device = &self.device;
        let input_ids =
            Tensor::from_vec(flat, (batch, max_len), device).map_err(|e| e.to_string())?;
        let token_types =
            Tensor::zeros((batch, max_len), DType::U32, device).map_err(|e| e.to_string())?;
        let hidden = self
            .bert
            .forward(&input_ids, &token_types, None)
            .map_err(|e| format!("BERT: {e}"))?;
        let mut logits = Vec::with_capacity(batch);
        for row in 0..batch {
            let row_hidden = hidden.narrow(0, row, 1).map_err(|e| e.to_string())?;
            let marker = row_hidden
                .narrow(1, starts[row], 1)
                .and_then(|t| t.squeeze(1))
                .map_err(|e| e.to_string())?;
            let count = ends[row].saturating_sub(starts[row] + 1);
            let word = if count == 0 {
                Tensor::zeros((1, self.bert_config.hidden_size), DType::F32, device)
                    .map_err(|e| e.to_string())?
            } else {
                row_hidden
                    .narrow(1, starts[row] + 1, count)
                    .and_then(|t| t.mean(1))
                    .map_err(|e| e.to_string())?
            };
            let features = Tensor::cat(&[&marker, &word], 1).map_err(|e| e.to_string())?;
            let value = self
                .head0
                .forward(&features)
                .and_then(|hidden| hidden.relu())
                .and_then(|hidden| self.head1.forward(&hidden))
                .map_err(|e| format!("homo_clf: {e}"))?
                .flatten_all()
                .map_err(|e| e.to_string())?
                .to_vec1::<f32>()
                .map_err(|e| e.to_string())?[0];
            logits.push(value);
        }
        Tensor::new(logits, &self.device)
            .and_then(|t| t.reshape((batch, 1)))
            .map_err(|e| e.to_string())
    }
}

type Batch = (Vec<Vec<u32>>, Vec<usize>, Vec<usize>);

struct Found {
    start: usize,
    end: usize,
    word: Vec<char>,
}

fn values(tensor: &Tensor) -> Result<Vec<f32>, String> {
    tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| error.to_string())
}

fn build_marked(sentence: &[char], found: &Found) -> Vec<char> {
    let mut mark = Vec::with_capacity(sentence.len() + 20);
    mark.extend_from_slice(&sentence[..found.start]);
    mark.extend(" [HOMO] ".chars());
    mark.extend_from_slice(&found.word);
    mark.extend(" [/HOMO] ".chars());
    mark.extend_from_slice(&sentence[found.end..]);
    mark
}

fn is_pattern_char(ch: char) -> bool {
    matches!(ch, 'а'..='я' | 'ё' | '+' | 'А'..='Я' | 'Ё')
}

fn is_lower_cyrillic(ch: char) -> bool {
    matches!(ch, 'а'..='я' | 'ё' | 'А'..='Я' | 'Ё')
}
