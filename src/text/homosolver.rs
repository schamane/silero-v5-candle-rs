//! Homograph-Löser: BERT (rubert-tiny2-Form) plus `homo_clf` auf
//! `[Marker-Embedding ; Wort-Mittel]`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertCandleConfig, HiddenAct};

use crate::config::{BertConfig, TokenizerConfig};
use crate::error::{io_at, stage};
use crate::text::accentor::AccentOptions;
use crate::text::bert_tokenizer::BertTokenizer;
use crate::{Error, Result};

/// Anzahl der Wörter in `homodict.json`. Das Homograph-Paket legt sie fest, `config.json`
/// enthält sie nicht.
const HOMODICT_WORD_COUNT: usize = 1924;
/// Breite der verborgenen Schicht `homo_clf.0`. Der Kopf nimmt `2 * hidden_size`
/// Eingaben auf (Marker-Embedding und Wort-Mittel).
const HOMO_HEAD_HIDDEN: usize = 256;

pub(crate) struct HomoSolver {
    bert: BertModel,
    head0: Linear,
    head1: Linear,
    tokenizer: BertTokenizer,
    homodict: HashMap<String, Vec<String>>,
    device: Device,
    /// `hidden_size` des BERT-Modells; der Kopf nimmt `2 * hidden_size` Merkmale.
    hidden_size: usize,
}

impl HomoSolver {
    /// Lädt BERT, Kopf, Vokabular und Homograph-Wörterbuch aus `dir`.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn eine Datei oder ein Gewicht fehlt, die BERT-
    /// Konfiguration unbekannt ist oder das Wörterbuch nicht `HOMODICT_WORD_COUNT`
    /// Wörter enthält.
    pub(crate) fn load(
        dir: &Path,
        device: &Device,
        bert_config: &BertConfig,
        tokenizer_config: &TokenizerConfig,
    ) -> Result<Self> {
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
                other => {
                    return Err(Error::model_msg(format!(
                        "unbekannte BERT-Aktivierung {other}"
                    )));
                }
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
        // SAFETY: Die Modelldateien in `dir` dürfen während des Mappings nicht geändert
        // oder ersetzt werden. Sonst ist der Inhalt des gemappten Speichers undefiniert.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&path), DType::F32, device)
                .map_err(|e| Error::model(path.display().to_string(), e))?
        };
        let bert = BertModel::load(vb.pp("bert"), &config)
            .map_err(|e| Error::model(path.display().to_string(), e))?;
        let head_in = 2 * bert_config.hidden_size;
        let head0 = candle_nn::linear(head_in, HOMO_HEAD_HIDDEN, vb.pp("homo_clf.0"))
            .map_err(|e| Error::model(format!("{}: homo_clf.0", path.display()), e))?;
        let head1 = candle_nn::linear(HOMO_HEAD_HIDDEN, 1, vb.pp("homo_clf.3"))
            .map_err(|e| Error::model(format!("{}: homo_clf.3", path.display()), e))?;

        let dict_path = dir.join("homodict.json");
        let dict_text = std::fs::read_to_string(&dict_path).map_err(io_at(dict_path))?;
        let mut homodict: HashMap<String, Vec<String>> =
            serde_json::from_str(&dict_text).map_err(|e| Error::model("homodict.json", e))?;
        if homodict.len() != HOMODICT_WORD_COUNT {
            return Err(Error::model_msg(format!(
                "homodict.json: {} statt {HOMODICT_WORD_COUNT} Wörter",
                homodict.len()
            )));
        }
        // Der Index aus dem Modell zeigt in die sortierte Reihenfolge der Varianten.
        // Sortieren wir einmal hier, nicht bei jedem Treffer.
        for variants in homodict.values_mut() {
            variants.sort();
        }
        let tokenizer = BertTokenizer::load(dir, tokenizer_config.clone(), bert_config.vocab_size)?;
        Ok(Self {
            bert,
            head0,
            head1,
            tokenizer,
            homodict,
            device: device.clone(),
            hidden_size: bert_config.hidden_size,
        })
    }

    /// Setzt Betonung und `ё` für Homographen im Satz.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn BERT oder der Kopf fehlschlägt ([`Error::Pipeline`])
    /// oder das Vokabular die Homograph-Marker nicht enthält ([`Error::Model`]).
    pub(crate) fn solve(&self, sentence: &[char], options: AccentOptions) -> Result<Vec<char>> {
        if !(options.put_stress || options.put_yo) {
            return Ok(sentence.to_vec());
        }
        let matches = self.find_matches(sentence);
        if matches.is_empty() {
            return Ok(sentence.to_vec());
        }

        let (ids_batch, starts, ends) = self.batch(sentence, &matches)?;
        let logits = self.forward(&ids_batch, &starts, &ends)?;
        let probs = candle_nn::ops::sigmoid(&logits).map_err(stage("homosolver"))?;
        let preds: Vec<u32> = values(&probs)?
            .into_iter()
            .map(|value| value.round_ties_even() as u32)
            .collect();

        let mut result = sentence.to_vec();
        let mut offset: i32 = 0;
        for (found, pred) in matches.iter().zip(preds) {
            let pred = self.context_override(sentence, found).unwrap_or(pred);
            let start = found.start as i32 + offset;
            let end = found.end as i32 + offset;
            let Some((replacement, stress_inserted)) =
                self.variant_replacement(found, pred, options)
            else {
                continue;
            };
            if stress_inserted {
                offset += 1;
            }
            if start >= 0 && (end as usize) <= result.len() {
                result.splice(start as usize..end as usize, replacement);
            }
        }
        Ok(result)
    }

    /// Kontextregel für das lokative `двери`: nach `в`/`на`/`о`/`при` ist die
    /// Endbetonung `двер+и` orthoepisch die Form des Präpositivs; `дв+ери` gilt in
    /// Genitiv/Dativ. Der Homograph-Klassifikator sieht das Präpositionswort nicht
    /// zuverlässig, deshalb entscheidet hier die Präposition.
    fn context_override(&self, sentence: &[char], found: &Found) -> Option<u32> {
        let key = found.word.iter().collect::<String>().to_lowercase();
        if key != "двери" {
            return None;
        }
        let previous = previous_word(sentence, found.start)?;
        let previous: String = previous
            .iter()
            .map(|ch| ch.to_lowercase().next().unwrap_or(*ch))
            .collect();
        if !matches!(
            previous.as_str(),
            "в" | "во" | "на" | "о" | "об" | "обо" | "при"
        ) {
            return None;
        }
        self.homodict
            .get("двери")?
            .iter()
            .position(|variant| variant == "двер+и")
            .and_then(|index| u32::try_from(index).ok())
    }

    /// Ersatz für `found` nach Variante `pred` seines Wörterbucheintrags. Das Flag
    /// meldet, ob eine Betonungsmarke `+` eingefügt wurde (verschiebt spätere Spannen).
    /// `None`, wenn die Variante oder ihre Betonungsmarke fehlt.
    fn variant_replacement(
        &self,
        found: &Found,
        pred: u32,
        options: AccentOptions,
    ) -> Option<(Vec<char>, bool)> {
        let AccentOptions {
            put_stress,
            put_yo,
            stress_single_vowel,
        } = options;
        let key = found.word.iter().collect::<String>().to_lowercase();
        let variant = self.homodict.get(&key)?.get(pred as usize)?;
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
        let stress_idx = word_pred.iter().position(|ch| *ch == '+')?;
        word_pred.retain(|ch| *ch != '+');
        let mut replacement: Vec<char> = found
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
        let insert_stress =
            (n_vowels > 1 || stress_single_vowel) && put_stress && stress_idx <= replacement.len();
        if insert_stress {
            replacement.insert(stress_idx, '+');
        }
        Some((replacement, insert_stress))
    }

    /// Logits für alle Homographen eines Satzes, `None` ohne Treffer.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn BERT oder der Kopf fehlschlägt ([`Error::Pipeline`])
    /// oder das Vokabular die Homograph-Marker nicht enthält ([`Error::Model`]).
    #[cfg(feature = "test-util")]
    pub(crate) fn logits(&self, sentence: &[char]) -> Result<Option<Tensor>> {
        let matches = self.find_matches(sentence);
        if matches.is_empty() {
            return Ok(None);
        }
        let (ids_batch, starts, ends) = self.batch(sentence, &matches)?;
        self.forward(&ids_batch, &starts, &ends).map(Some)
    }

    fn batch(&self, sentence: &[char], matches: &[Found]) -> Result<Batch> {
        let mut ids_batch = Vec::new();
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for found in matches {
            let mark = build_marked(sentence, found);
            let ids = self.tokenizer.encode(&mark.iter().collect::<String>());
            let start = ids
                .iter()
                .position(|id| *id == self.tokenizer.config().homo_start_id)
                .ok_or_else(|| Error::model_msg("HOMO-Startmarker fehlt"))?;
            let end = ids
                .iter()
                .position(|id| *id == self.tokenizer.config().homo_end_id)
                .ok_or_else(|| Error::model_msg("HOMO-Endmarker fehlt"))?;
            ids_batch.push(ids);
            starts.push(start);
            ends.push(end);
        }
        Ok((ids_batch, starts, ends))
    }

    /// `(?=.*[а-яё])[а-яё+]+` mit IGNORECASE als Zeichenschleife.
    fn find_matches(&self, sentence: &[char]) -> Vec<Found> {
        // Der Vorausblick `(?=.*[а-яё])` gilt, wenn ein kyrillisches Zeichen ab `start`
        // folgt. Das ist genau dann der Fall, wenn das letzte solche Zeichen ab `start`
        // liegt. Den Index berechnen wir einmal, statt den Rest des Satzes zu scannen.
        let last_cyrillic = sentence.iter().rposition(|ch| is_cyrillic(*ch));
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
            if !last_cyrillic.is_some_and(|last| last >= start) {
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

    /// Führt BERT und den kleinen Kopf für alle Zeilen des Batches in einem Durchlauf aus.
    fn forward(&self, ids_batch: &[Vec<u32>], starts: &[usize], ends: &[usize]) -> Result<Tensor> {
        let batch = ids_batch.len();
        let max_len = ids_batch.iter().map(Vec::len).max().unwrap_or(0);
        let pad_token_id = self.tokenizer.config().pad_token_id;
        let mut flat = vec![pad_token_id; batch * max_len];
        for (row, ids) in ids_batch.iter().enumerate() {
            flat[row * max_len..row * max_len + ids.len()].copy_from_slice(ids);
        }
        let device = &self.device;
        let input_ids =
            Tensor::from_vec(flat, (batch, max_len), device).map_err(stage("homosolver"))?;
        let token_types =
            Tensor::zeros((batch, max_len), DType::U32, device).map_err(stage("homosolver"))?;
        let hidden = self
            .bert
            .forward(&input_ids, &token_types, None)
            .map_err(stage("bert"))?;

        // Je Zeile eine Gewichtsmatrix (2, max_len): Zeile 0 wählt das Marker-Token,
        // Zeile 1 mittelt die Wortspanne. Ein Matmul ergibt `[marker ; wort]` für alle
        // Zeilen zugleich. Ohne Wortspanne bleibt die Mittelzeile null.
        let mut weights = vec![0f32; batch * 2 * max_len];
        for row in 0..batch {
            let base = row * 2 * max_len;
            let marker_slot = base + starts[row];
            *weights
                .get_mut(marker_slot)
                .ok_or_else(|| Error::Pipeline {
                    stage: "homosolver",
                    source: candle_core::Error::Msg(format!(
                        "HOMO-Startmarker {} außerhalb der Sequenz {max_len}",
                        starts[row]
                    )),
                })? = 1.0;
            let count = ends[row].saturating_sub(starts[row] + 1);
            if count > 0 {
                let share = 1.0 / count as f32;
                for position in starts[row] + 1..starts[row] + 1 + count {
                    let slot = base + max_len + position;
                    *weights.get_mut(slot).ok_or_else(|| Error::Pipeline {
                        stage: "homosolver",
                        source: candle_core::Error::Msg(format!(
                            "HOMO-Wortspanne {position} außerhalb der Sequenz {max_len}"
                        )),
                    })? = share;
                }
            }
        }
        let weights =
            Tensor::from_vec(weights, (batch, 2, max_len), device).map_err(stage("homosolver"))?;
        let pooled = weights.matmul(&hidden).map_err(stage("homosolver"))?;
        let features = pooled
            .reshape((batch, 2 * self.hidden_size))
            .map_err(stage("homosolver"))?;
        self.head0
            .forward(&features)
            .and_then(|hidden| hidden.relu())
            .and_then(|hidden| self.head1.forward(&hidden))
            .map_err(stage("homo_clf"))
    }
}

type Batch = (Vec<Vec<u32>>, Vec<usize>, Vec<usize>);

struct Found {
    start: usize,
    end: usize,
    word: Vec<char>,
}

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(stage("homosolver"))
}

/// Das Wort direkt vor `start`, ohne dazwischenliegende Leerzeichen; `None`, wenn
/// ein Satzzeichen oder der Satzanfang davor liegt.
fn previous_word(sentence: &[char], start: usize) -> Option<Vec<char>> {
    let mut end = start;
    while end > 0 && sentence[end - 1].is_whitespace() {
        end -= 1;
    }
    let mut begin = end;
    while begin > 0 && is_pattern_char(sentence[begin - 1]) {
        begin -= 1;
    }
    (begin < end).then(|| sentence[begin..end].to_vec())
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

/// Kyrillische Buchstaben in beiden Groß-/Kleinschreibungen.
fn is_cyrillic(ch: char) -> bool {
    matches!(ch, 'а'..='я' | 'ё' | 'А'..='Я' | 'Ё')
}
