//! Akzentor: fastText-artiges n-Gramm-Embedding plus zwei MLP-Köpfe und die
//! Regelkette aus `AccentorNgram.__call__`.

use std::collections::HashMap;
use std::path::Path;

use crate::error::io_at;
use crate::text::ngram::{Linear, NgramEmbedding, run_mlp, softmax_argmax};
use crate::text::stress_dict::StressDictionary;
use crate::text::tokens::{Token, hyphen_parts, lower_chars, ordinal_positions};
use crate::{Error, Result};

pub(crate) fn is_vowel(ch: char) -> bool {
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
pub(crate) struct AccentOptions {
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

pub(crate) struct Accentor {
    embedding: NgramEmbedding,
    stress: Vec<Linear>,
    yo: Vec<Linear>,
    exceptions: HashMap<String, (i32, i32)>,
    /// Optionales Wörterbuch aus `stress_dict.tsv`; fehlt es, bleibt das Modell allein.
    dictionary: Option<StressDictionary>,
}

impl Accentor {
    /// Lädt Gewichte, n-Gramm-Tabelle und Ausnahmen aus `dir`.
    ///
    /// Die Gewichte bleiben auf der CPU: der Akzentor rechnet in reinem Rust mit
    /// `Vec<f32>`; ein Laden auf die GPU wäre nur ein zusätzlicher Kopierweg.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Io`] zurück, wenn eine Datei nicht gelesen werden kann, und sonst
    /// [`Error::Model`], wenn eine Datei fehlt, eine Form oder Anzahl nicht passt oder ein
    /// n-Gramm beziehungsweise Ausnahmewort doppelt vorkommt.
    pub(crate) fn load(dir: &Path, with_dictionary: bool) -> Result<Self> {
        let path = dir.join("accentor.safetensors");
        let tensors = candle_core::safetensors::load(&path, &candle_core::Device::Cpu)
            .map_err(|e| Error::model(path.display().to_string(), e))?;
        let ngrams_path = dir.join("ngrams.txt");
        let embedding = NgramEmbedding::load(
            &ngrams_path,
            &tensors,
            "embedding",
            NGRAM_COUNT,
            EMBEDDING_DIM,
        )?;
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
                return Err(Error::model_msg(format!(
                    "exceptions.tsv: doppeltes Wort {word:?} in Zeile {}",
                    line_no + 1
                )));
            }
        }
        if exceptions.len() != EXCEPTION_COUNT {
            return Err(Error::model_msg(format!(
                "exceptions.tsv: {} statt {EXCEPTION_COUNT} Einträge",
                exceptions.len()
            )));
        }
        Ok(Self {
            embedding,
            stress,
            yo,
            exceptions,
            dictionary: if with_dictionary {
                StressDictionary::load(dir)?
            } else {
                None
            },
        })
    }

    /// Logits beider Köpfe für die maskierten Wörter.
    pub(crate) fn logits(&self, words: &[Vec<char>]) -> Vec<(Vec<f32>, Vec<f32>)> {
        words
            .iter()
            .map(|word| {
                let vector = self.embedding.word_vector(word);
                (run_mlp(&self.stress, &vector), run_mlp(&self.yo, &vector))
            })
            .collect()
    }

    /// Portierung von `AccentorNgram.__call__` für eine Zeile.
    pub(crate) fn accent(&self, sentence: &[char], options: AccentOptions) -> Vec<char> {
        if !(options.put_stress || options.put_yo) {
            return sentence.to_vec();
        }
        let tokens = Self::tokenize(sentence);
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
            output.push(self.apply_word(token, (&stress_logits, &yo_logits), options));
        }
        output.concat()
    }

    fn apply_word(
        &self,
        token: &Token,
        (stress_logits, yo_logits): (&[f32], &[f32]),
        options: AccentOptions,
    ) -> Vec<char> {
        let word = WordCase::new(token);
        if word.have_stress() && word.have_yo() {
            return word.raw;
        }
        if !word.have_stress() && word.have_yo() && options.put_stress {
            return word.stress_yo_words(options);
        }
        let clean_word: String = token.clean.iter().collect();
        if let Some((exc_stress, exc_yo)) = self.exceptions.get(&clean_word) {
            let have_stress = word.have_stress();
            return accentuate_exception(&word.raw, *exc_stress, *exc_yo, have_stress);
        }
        if !word.have_stress()
            && let Some(stress) = self
                .dictionary
                .as_ref()
                .and_then(|dictionary| dictionary.stress(&clean_word))
        {
            let stress = i32::try_from(stress).unwrap_or(i32::MAX);
            return accentuate_exception(&word.raw, stress, -1, false);
        }
        word.predicted(stress_logits, yo_logits, options)
    }

    fn tokenize(sentence: &[char]) -> Vec<Token> {
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
            for (raw, is_last) in hyphen_parts(word) {
                let clean: Vec<char> = raw
                    .iter()
                    .flat_map(|ch| ch.to_lowercase())
                    .filter(|ch| matches!(ch, 'а'..='я' | 'ё'))
                    .collect();
                // Vergleich in Kleinschreibung. `prepare_text_input` schreibt die Eingabe
                // bereits klein, daher ändert das für die Pipeline nichts.
                let stop = is_last && {
                    let lowered: String = raw.iter().flat_map(|ch| ch.to_lowercase()).collect();
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

/// Ein Wort in Originalschreibung und in zeichenweiser Kleinschreibung.
struct WordCase {
    raw: Vec<char>,
    /// Genau ein Zeichen je Eingabezeichen, damit die Indizes auch für `raw` gelten
    /// (`İ` wird sonst zu zwei Zeichen).
    lower: Vec<char>,
}

impl WordCase {
    fn new(token: &Token) -> Self {
        let raw = token.raw.clone();
        let lower = lower_chars(&raw);
        Self { raw, lower }
    }

    fn have_stress(&self) -> bool {
        self.lower.contains(&'+')
    }

    fn have_yo(&self) -> bool {
        self.lower.contains(&'ё')
    }

    /// Wort mit `ё` ohne Betonungsmarke: die Marke kommt vor jedes `ё`.
    fn stress_yo_words(mut self, options: AccentOptions) -> Vec<char> {
        let vowel_count = self.lower.iter().filter(|ch| VOWELS.contains(ch)).count();
        if vowel_count == 1 && !options.stress_single_vowel {
            return self.raw;
        }
        let positions: Vec<usize> = self
            .lower
            .iter()
            .enumerate()
            .filter(|(_, ch)| **ch == 'ё')
            .map(|(index, _)| index)
            .collect();
        for (offset, position) in positions.into_iter().enumerate() {
            self.raw.insert(position + offset, '+');
        }
        self.raw
    }

    /// Setzt `ё` und Betonung nach den Vorhersagen beider Köpfe.
    fn predicted(
        mut self,
        stress_logits: &[f32],
        yo_logits: &[f32],
        options: AccentOptions,
    ) -> Vec<char> {
        let AccentOptions {
            put_stress,
            put_yo,
            stress_single_vowel,
        } = options;
        let have_stress = self.have_stress();
        let (stress_probs, stress_pred) = softmax_argmax(stress_logits);
        let (yo_probs, yo_pred) = softmax_argmax(yo_logits);
        let mut stressed_vowel_ids = vec![stress_pred as i32];
        let stress_threshold = if put_stress { 0.5 } else { 1.0 };
        let set_stress = stress_probs[stress_pred] > stress_threshold && !have_stress;
        let yo_vowel_ids = vec![yo_pred as i32];
        let yo_threshold = if put_yo { 0.5 } else { 1.0 };
        let set_yo = yo_probs[yo_pred] > yo_threshold;

        if have_stress {
            stressed_vowel_ids = self
                .lower
                .split(|ch| *ch == '+')
                .map(|part| part.iter().filter(|ch| VOWELS.contains(ch)).count() as i32)
                .collect();
        }

        let (mut stress_positions, yo_positions, num_vowels, first_vowel_pos) =
            get_positions(&self.lower, &stressed_vowel_ids, &yo_vowel_ids);
        if num_vowels == 0 {
            return self.raw;
        }
        for yo_pos in &yo_positions {
            if stress_positions.contains(yo_pos) && set_yo && self.lower[*yo_pos] == 'е' {
                let replacement = if self.raw[*yo_pos].is_lowercase() {
                    'ё'
                } else {
                    'Ё'
                };
                self.raw[*yo_pos] = replacement;
            }
        }
        let mut set_stress = set_stress;
        if num_vowels == 1 {
            stress_positions = vec![first_vowel_pos];
            set_stress = stress_single_vowel && put_stress;
        }
        if !have_stress && set_stress {
            for (offset, position) in stress_positions.into_iter().enumerate() {
                self.raw.insert(position + offset, '+');
            }
        }
        self.raw
    }
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
    let stress_positions = ordinal_positions(&vowel_ids, stressed_vowel_ids);
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
