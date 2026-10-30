//! Akzentor: fastText-artiges n-Gramm-Embedding plus zwei MLP-Köpfe und die
//! Regelkette aus `AccentorNgram.__call__`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use candle_core::{Device, Tensor};

use crate::config::Config;
use crate::error::io_at;
use crate::text::normalize::prepare_text_input;
use crate::{Error, Result};

pub fn is_vowel(ch: char) -> bool {
    VOWELS.contains(&ch)
}

const VOWELS: [char; 10] = ['а', 'о', 'у', 'ы', 'э', 'и', 'е', 'я', 'ё', 'ю'];
/// Wörter, die nicht betont werden. Der Vergleich erfolgt in Kleinschreibung.
const STOP_WORDS: [&str; 1] = ["то"];
/// Anzahl der Zeilen in `ngrams.txt`. Das Akzentor-Paket legt sie fest, `config.json`
/// enthält sie nicht.
const NGRAM_COUNT: usize = 126_523;
/// Anzahl der Zeilen in `exceptions.tsv`. Das Akzentor-Paket legt sie fest, `config.json`
/// enthält sie nicht.
const EXCEPTION_COUNT: usize = 16_969;
/// Embedding-Dimension des Akzentors, Form `embedding.weight` = `[NGRAM_COUNT, EMBEDDING_DIM]`.
const EMBEDDING_DIM: usize = 16;

/// Schalter für [`Accentor::accent`]. `Default` schaltet alle drei ein, wie die Pipeline sie
/// nutzt.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct AccentOptions {
    /// Betonung setzen.
    pub put_stress: bool,
    /// `ё` setzen.
    pub put_yo: bool,
    /// Auch Wörter mit nur einem Vokal betonen.
    pub stress_single_vowel: bool,
}

impl Default for AccentOptions {
    fn default() -> Self {
        Self {
            put_stress: true,
            put_yo: true,
            stress_single_vowel: true,
        }
    }
}

#[derive(Debug)]
struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    out: usize,
    in_dim: usize,
}

impl Linear {
    fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        out: usize,
        in_dim: usize,
    ) -> Result<Self> {
        let w = tensor_vec(tensors, &format!("{prefix}.weight"))?;
        let b = tensor_vec(tensors, &format!("{prefix}.bias"))?;
        if w.len() != out * in_dim {
            return Err(Error::Model(format!(
                "{prefix}.weight: {} Werte statt {out}x{in_dim}",
                w.len()
            )));
        }
        if b.len() != out {
            return Err(Error::Model(format!(
                "{prefix}.bias: {} Werte statt {out}",
                b.len()
            )));
        }
        Ok(Self { w, b, out, in_dim })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.in_dim,
            "Linear: Eingabe {} statt {}",
            x.len(),
            self.in_dim
        );
        let mut y = vec![0.0; self.out];
        for (o, slot) in y.iter_mut().enumerate() {
            let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            let mut sum = self.b[o];
            for (w, x) in row.iter().zip(x) {
                sum += w * x;
            }
            *slot = sum;
        }
        y
    }
}

fn tensor_vec(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Vec<f32>> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| Error::Model(format!("{name} fehlt in accentor.safetensors")))?;
    tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| Error::Model(format!("{name}: {error}")))
}

pub struct Accentor {
    ngrams: HashMap<String, u32>,
    embedding: Vec<f32>,
    dim: usize,
    stress: Vec<Linear>,
    yo: Vec<Linear>,
    exceptions: HashMap<String, (i32, i32)>,
}

impl Accentor {
    /// Lädt Gewichte, n-Gramm-Tabelle und Ausnahmen aus `dir`.
    ///
    /// # Errors
    ///
    /// Gibt einen Fehler zurück, wenn eine Datei fehlt, eine Form oder Anzahl nicht
    /// passt oder ein n-Gramm beziehungsweise Ausnahmewort doppelt vorkommt.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Io`] zurück, wenn eine Datei nicht gelesen werden kann, und sonst
    /// [`Error::Model`], wenn eine Datei fehlt, eine Form oder Anzahl nicht passt oder ein
    /// n-Gramm beziehungsweise Ausnahmewort doppelt vorkommt.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let path = dir.join("accentor.safetensors");
        let tensors = candle_core::safetensors::load(&path, device)
            .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        let embedding = tensor_vec(&tensors, "embedding.weight")?;
        let dim = EMBEDDING_DIM;
        if embedding.len() != NGRAM_COUNT * dim {
            return Err(Error::Model(
                "accentor.safetensors: unerwartete Embedding-Form".into(),
            ));
        }
        let mut ngrams = HashMap::with_capacity(NGRAM_COUNT);
        let ngrams_path = dir.join("ngrams.txt");
        let text = std::fs::read_to_string(&ngrams_path).map_err(io_at(ngrams_path))?;
        for (id, gram) in text.lines().enumerate() {
            if ngrams.insert(gram.to_string(), id as u32).is_some() {
                return Err(Error::Model(format!(
                    "ngrams.txt: doppeltes n-Gramm {gram:?} in Zeile {}",
                    id + 1
                )));
            }
        }
        if ngrams.len() != NGRAM_COUNT {
            return Err(Error::Model(format!(
                "ngrams.txt: {} statt {NGRAM_COUNT} Einträge",
                ngrams.len()
            )));
        }
        let stress = vec![
            Linear::load(&tensors, "stress_clf.0", 64, 16)?,
            Linear::load(&tensors, "stress_clf.2", 128, 64)?,
            Linear::load(&tensors, "stress_clf.4", 64, 128)?,
            Linear::load(&tensors, "stress_clf.6", 10, 64)?,
        ];
        let yo = vec![
            Linear::load(&tensors, "yo_clf.0", 64, 16)?,
            Linear::load(&tensors, "yo_clf.2", 128, 64)?,
            Linear::load(&tensors, "yo_clf.4", 64, 128)?,
            Linear::load(&tensors, "yo_clf.6", 7, 64)?,
        ];
        let mut exceptions = HashMap::with_capacity(EXCEPTION_COUNT);
        let exceptions_path = dir.join("exceptions.tsv");
        let text = std::fs::read_to_string(&exceptions_path).map_err(io_at(exceptions_path))?;
        for (line_no, line) in text.lines().enumerate() {
            let mut parts = line.split('\t');
            let word = parts.next().unwrap_or("");
            let stress = parts.next().and_then(|v| v.parse::<i32>().ok());
            let yo = parts.next().and_then(|v| v.parse::<i32>().ok());
            if let (false, Some(stress), Some(yo)) = (word.is_empty(), stress, yo)
                && exceptions.insert(word.to_string(), (stress, yo)).is_some()
            {
                return Err(Error::Model(format!(
                    "exceptions.tsv: doppeltes Wort {word:?} in Zeile {}",
                    line_no + 1
                )));
            }
        }
        if exceptions.len() != EXCEPTION_COUNT {
            return Err(Error::Model(format!(
                "exceptions.tsv: {} statt {EXCEPTION_COUNT} Einträge",
                exceptions.len()
            )));
        }
        Ok(Self {
            ngrams,
            embedding,
            dim,
            stress,
            yo,
            exceptions,
        })
    }

    /// `word_ngrams(word, 1, len+3)`, dann `embedding_bag(mean)`. `UNK` wird nur
    /// verwendet, wenn kein n-Gramm im Vokabular liegt.
    fn word_vector(&self, word: &[char]) -> Vec<f32> {
        let max_len = word.len() + 3;
        let mut wrapped = Vec::with_capacity(word.len() + 2);
        wrapped.push('<');
        wrapped.extend_from_slice(word);
        wrapped.push('>');
        let mut ids = Vec::new();
        // Ein Puffer für alle n-Gramme. `clear` behält die Kapazität.
        let mut gram = String::new();
        for n in 1..=max_len {
            if n > wrapped.len() {
                continue;
            }
            for start in 0..=wrapped.len() - n {
                gram.clear();
                gram.extend(&wrapped[start..start + n]);
                if let Some(id) = self.ngrams.get(gram.as_str()) {
                    ids.push(*id as usize);
                }
            }
        }
        if word.is_empty()
            && let Some(id) = self.ngrams.get("")
        {
            ids.push(*id as usize);
        }
        if ids.is_empty() {
            ids.push(*self.ngrams.get("UNK").unwrap_or(&0) as usize);
        }
        let mut mean = vec![0.0f32; self.dim];
        for id in &ids {
            let row = &self.embedding[id * self.dim..(id + 1) * self.dim];
            for (slot, value) in mean.iter_mut().zip(row) {
                *slot += value;
            }
        }
        let count = ids.len() as f32;
        for value in &mut mean {
            *value /= count;
        }
        mean
    }

    /// Logits beider Köpfe für die maskierten Wörter.
    pub fn logits(&self, words: &[Vec<char>]) -> Vec<(Vec<f32>, Vec<f32>)> {
        words
            .iter()
            .map(|word| {
                let vector = self.word_vector(word);
                (self.mlp(&self.stress, &vector), self.mlp(&self.yo, &vector))
            })
            .collect()
    }

    fn mlp(&self, layers: &[Linear], input: &[f32]) -> Vec<f32> {
        let mut x = input.to_vec();
        for (index, layer) in layers.iter().enumerate() {
            x = layer.forward(&x);
            if index + 1 < layers.len() {
                for value in &mut x {
                    *value = value.max(0.0);
                }
            }
        }
        x
    }

    /// Portierung von `AccentorNgram.__call__` für eine Zeile.
    pub fn accent(
        &self,
        sentence: &[char],
        options: AccentOptions,
        skip_stress_words: &HashSet<String>,
        skip_yo_words: &HashSet<String>,
    ) -> Vec<char> {
        if !(options.put_stress || options.put_yo) {
            return sentence.to_vec();
        }
        let tokens = self.tokenize(sentence);
        let masked: Vec<Vec<char>> = tokens
            .iter()
            .filter(|token| token.process)
            .map(|token| token.clean.clone())
            .collect();
        let logits = self.logits(&masked);
        let mut prediction = logits.into_iter();

        let mut output: Vec<Vec<char>> = Vec::with_capacity(tokens.len());
        for token in &tokens {
            if !token.process {
                output.push(token.raw.clone());
                continue;
            }
            let Some((stress_logits, yo_logits)) = prediction.next() else {
                output.push(token.raw.clone());
                continue;
            };
            output.push(self.apply_word(
                token,
                stress_logits,
                yo_logits,
                options,
                skip_stress_words,
                skip_yo_words,
            ));
        }
        output.concat()
    }

    fn apply_word(
        &self,
        token: &Token,
        stress_logits: Vec<f32>,
        yo_logits: Vec<f32>,
        options: AccentOptions,
        skip_stress_words: &HashSet<String>,
        skip_yo_words: &HashSet<String>,
    ) -> Vec<char> {
        let AccentOptions {
            put_stress,
            put_yo,
            stress_single_vowel,
        } = options;
        let (stress_probs, stress_pred) = softmax_argmax(&stress_logits);
        let (yo_probs, yo_pred) = softmax_argmax(&yo_logits);
        let clean_word: String = token.clean.iter().collect();
        let clean_yo = clean_word.replace('ё', "е");
        let mut raw_word = token.raw.clone();
        // Zeichenweise Kleinschreibung: genau ein Zeichen je Eingabezeichen, damit die
        // Indizes von `raw_lower` auch für `raw_word` gelten (`İ` wird sonst zu zwei Zeichen).
        let raw_lower: Vec<char> = raw_word
            .iter()
            .map(|ch| ch.to_lowercase().next().unwrap_or(*ch))
            .collect();
        let have_stress = raw_lower.contains(&'+');
        let have_yo = raw_lower.contains(&'ё');

        if have_stress && have_yo {
            return raw_word;
        }
        if !have_stress && have_yo && put_stress {
            let vowel_count = raw_lower.iter().filter(|ch| VOWELS.contains(ch)).count();
            if (vowel_count == 1 && !stress_single_vowel) || skip_stress_words.contains(&clean_yo) {
                return raw_word;
            }
            let positions: Vec<usize> = raw_lower
                .iter()
                .enumerate()
                .filter(|(_, ch)| **ch == 'ё')
                .map(|(index, _)| index)
                .collect();
            for (offset, position) in positions.into_iter().enumerate() {
                raw_word.insert(position + offset, '+');
            }
            return raw_word;
        }

        if let Some((exc_stress, exc_yo)) = self.exceptions.get(&clean_word) {
            return accentuate_exception(&raw_word, *exc_stress, *exc_yo, have_stress);
        }

        let mut stressed_vowel_ids = vec![stress_pred as i32];
        let stress_threshold = if put_stress { 0.5 } else { 1.0 };
        let set_stress = stress_probs[stress_pred] > stress_threshold
            && !have_stress
            && !skip_stress_words.contains(&clean_yo);
        let yo_vowel_ids = vec![yo_pred as i32];
        let yo_threshold = if put_yo { 0.5 } else { 1.0 };
        let set_yo = yo_probs[yo_pred] > yo_threshold && !skip_yo_words.contains(&clean_yo);

        if have_stress {
            stressed_vowel_ids = raw_lower
                .split(|ch| *ch == '+')
                .map(|part| part.iter().filter(|ch| VOWELS.contains(ch)).count() as i32)
                .collect();
        }

        let (mut stress_positions, yo_positions, num_vowels, first_vowel_pos) =
            get_positions(&raw_lower, &stressed_vowel_ids, &yo_vowel_ids);
        if num_vowels == 0 {
            return raw_word;
        }
        for yo_pos in &yo_positions {
            if stress_positions.contains(yo_pos) && set_yo && raw_lower[*yo_pos] == 'е' {
                let replacement = if raw_word[*yo_pos].is_lowercase() {
                    'ё'
                } else {
                    'Ё'
                };
                raw_word[*yo_pos] = replacement;
            }
        }
        let mut set_stress = set_stress;
        if num_vowels == 1 {
            stress_positions = vec![first_vowel_pos];
            set_stress = stress_single_vowel && put_stress;
        }
        if !have_stress && set_stress {
            for (offset, position) in stress_positions.into_iter().enumerate() {
                raw_word.insert(position + offset, '+');
            }
        }
        raw_word
    }

    fn tokenize(&self, sentence: &[char]) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < sentence.len() {
            if sentence[i].is_whitespace() {
                let start = i;
                while i < sentence.len() && sentence[i].is_whitespace() {
                    i += 1;
                }
                tokens.push(Token {
                    raw: sentence[start..i].to_vec(),
                    clean: Vec::new(),
                    process: false,
                });
                continue;
            }
            let start = i;
            while i < sentence.len() && !sentence[i].is_whitespace() {
                i += 1;
            }
            let word = &sentence[start..i];
            let parts: Vec<Vec<char>> = split_on(word, '-');
            let last = parts.len() - 1;
            for (index, part) in parts.iter().enumerate() {
                let mut raw = part.clone();
                if index < last {
                    raw.push('-');
                }
                let clean: Vec<char> = raw
                    .iter()
                    .flat_map(|ch| ch.to_lowercase())
                    .filter(|ch| matches!(ch, 'а'..='я' | 'ё'))
                    .collect();
                // Vergleich in Kleinschreibung. `prepare_text_input` schreibt die Eingabe
                // bereits klein, daher ändert das für die Pipeline nichts.
                let stop = index == last && {
                    let lowered: String = part.iter().flat_map(|ch| ch.to_lowercase()).collect();
                    STOP_WORDS.contains(&lowered.as_str())
                };
                tokens.push(Token {
                    raw,
                    process: !clean.is_empty() && !stop,
                    clean,
                });
            }
        }
        tokens
    }
}

struct Token {
    raw: Vec<char>,
    clean: Vec<char>,
    process: bool,
}

fn split_on(word: &[char], sep: char) -> Vec<Vec<char>> {
    let mut parts = Vec::new();
    let mut current = Vec::new();
    for ch in word {
        if *ch == sep {
            parts.push(current);
            current = Vec::new();
        } else {
            current.push(*ch);
        }
    }
    parts.push(current);
    parts
}

fn softmax_argmax(logits: &[f32]) -> (Vec<f32>, usize) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for value in &mut probs {
        *value /= sum;
    }
    let mut argmax = 0;
    for (index, value) in probs.iter().enumerate() {
        if *value > probs[argmax] {
            argmax = index;
        }
    }
    (probs, argmax)
}

fn get_positions(
    word: &[char],
    stressed_vowel_ids: &[i32],
    yo_vowel_ids: &[i32],
) -> (Vec<usize>, Vec<usize>, usize, usize) {
    let vowel_ids: Vec<usize> = word
        .iter()
        .enumerate()
        .filter(|(_, ch)| VOWELS.contains(ch))
        .map(|(index, _)| index)
        .collect();
    let ye_ids: Vec<usize> = word
        .iter()
        .enumerate()
        .filter(|(_, ch)| **ch == 'е')
        .map(|(index, _)| index)
        .collect();
    let stress_positions = stressed_vowel_ids
        .iter()
        .filter(|idx| **idx >= 0 && (**idx as usize) < vowel_ids.len() && !vowel_ids.is_empty())
        .map(|idx| vowel_ids[*idx as usize])
        .collect();
    let yo_positions = yo_vowel_ids
        .iter()
        .filter(|idx| **idx > 0 && ((**idx - 1) as usize) < ye_ids.len() && !ye_ids.is_empty())
        .map(|idx| ye_ids[(*idx - 1) as usize])
        .collect();
    let num_vowels = vowel_ids.len();
    let first_vowel_pos = vowel_ids.first().copied().unwrap_or(0);
    (stress_positions, yo_positions, num_vowels, first_vowel_pos)
}

fn accentuate_exception(
    word: &[char],
    exc_stress: i32,
    exc_yo: i32,
    have_stress: bool,
) -> Vec<char> {
    if have_stress {
        let stress_positions: Vec<usize> = word
            .iter()
            .enumerate()
            .filter(|(_, ch)| **ch == '+')
            .map(|(index, _)| index)
            .collect();
        let mut accentuated: Vec<char> = word.iter().copied().filter(|ch| *ch != '+').collect();
        if exc_yo >= 0 && stress_positions.contains(&((exc_yo + 1) as usize)) {
            let pos = exc_yo as usize;
            if pos < accentuated.len() {
                accentuated[pos] = if accentuated[pos].is_lowercase() {
                    'ё'
                } else {
                    'Ё'
                };
            }
        }
        for position in stress_positions {
            if position <= accentuated.len() {
                accentuated.insert(position, '+');
            }
        }
        accentuated
    } else {
        let mut raw = word.to_vec();
        if exc_yo >= 0 && (exc_yo as usize) < raw.len() {
            let pos = exc_yo as usize;
            raw[pos] = if raw[pos].is_lowercase() { 'ё' } else { 'Ё' };
        }
        if exc_stress >= 0 && (exc_stress as usize) <= raw.len() {
            raw.insert(exc_stress as usize, '+');
        }
        raw
    }
}

/// Hilfsfunktion für Tests: derselbe Pfad wie `synthesize`, nur Text.
///
/// # Errors
///
/// Gibt [`Error::Input`] zurück, wenn der Text keine sprechbaren Zeichen enthält.
pub fn accent_text(
    config: &Config,
    accentor: &Accentor,
    text: &str,
) -> Result<(Vec<char>, Vec<char>)> {
    let prepared = prepare_text_input(config, text);
    if !prepared.has_text {
        return Err(Error::Input("text is empty".into()));
    }
    let accented = accentor.accent(
        &prepared.sentence,
        AccentOptions::default(),
        &HashSet::new(),
        &HashSet::new(),
    );
    Ok((prepared.sentence, accented))
}
