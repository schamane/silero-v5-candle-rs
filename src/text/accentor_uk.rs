//! Ukrainischer Akzentor aus `silero-stress` (`AccentorNgramSimple`).
//!
//! Anders als der russische Akzentor hat er keinen Homograph-Löser, keinen
//! `ё`-Kopf und keine Ausnahmeliste: ein n-Gramm-Embedding, ein MLP mit zehn
//! Stress-Klassen und die Regelkette aus `AccentorNgramSimple.__call__`.

use std::path::Path;

use candle_core::Device;

use crate::text::ngram::{Linear, NgramEmbedding, run_mlp, softmax_argmax};
use crate::{Error, Result};

/// Ukrainische Vokale in der Reihenfolge, in der das Modell sie zählt.
pub const VOWELS: &str = "аеєиіїоуюя";
/// Zeichen vor dem betonten Vokal, wie in `silero-stress`.
pub const STRESS_TOKEN: char = '+';

/// Anzahl der Zeilen in `ngrams.txt`; das Akzentor-Paket legt sie fest.
const NGRAM_COUNT: usize = 130_694;
/// Embedding-Dimension des Akzentors (`embedding.weight` = `[NGRAM_COUNT, 16]`).
const EMBEDDING_DIM: usize = 16;
/// Anzahl der Stress-Klassen; das Modell kann höchstens den zehnten Vokal betonen.
const MAX_VOWELS: usize = 10;
/// Schwellwert aus `AccentorNgramSimple.__call__`.
const STRESS_THRESHOLD: f32 = 0.5;

/// Schalter für [`UkrainianAccentor::accent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StressOptions {
    /// Auch Wörter mit genau einem Vokal betonen. `Default` ist `true`, wie im
    /// Python-Akzentor.
    pub stress_single_vowel: bool,
}

impl Default for StressOptions {
    fn default() -> Self {
        Self {
            stress_single_vowel: true,
        }
    }
}

/// Akzentor für ukrainischen Text.
#[derive(Debug)]
pub struct UkrainianAccentor {
    embedding: NgramEmbedding,
    stress: Vec<Linear>,
}

impl UkrainianAccentor {
    /// Lädt `accentor.safetensors` und `ngrams.txt` aus `dir`.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Io`] zurück, wenn eine Datei nicht gelesen werden kann, und
    /// [`Error::Model`], wenn Formen, Anzahl oder Schichtkette nicht passen.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let path = dir.join("accentor.safetensors");
        let tensors = candle_core::safetensors::load(&path, device)
            .map_err(|error| Error::Model(format!("{}: {error}", path.display())))?;
        let ngrams_path = dir.join("ngrams.txt");
        let embedding = NgramEmbedding::load(
            &ngrams_path,
            &tensors,
            "embedding",
            NGRAM_COUNT,
            EMBEDDING_DIM,
        )?;
        let stress = vec![
            Linear::load_auto(&tensors, "stress_clf.0")?,
            Linear::load_auto(&tensors, "stress_clf.2")?,
            Linear::load_auto(&tensors, "stress_clf.4")?,
            Linear::load_auto(&tensors, "stress_clf.6")?,
        ];
        if stress[0].in_dim() != EMBEDDING_DIM {
            return Err(Error::Model(format!(
                "stress_clf.0: Eingabe {} statt {EMBEDDING_DIM}",
                stress[0].in_dim()
            )));
        }
        for (index, pair) in stress.windows(2).enumerate() {
            if pair[0].out() != pair[1].in_dim() {
                return Err(Error::Model(format!(
                    "stress_clf.{}: Ausgabe {} passt nicht zu stress_clf.{} Eingabe {}",
                    index * 2,
                    pair[0].out(),
                    index * 2 + 2,
                    pair[1].in_dim()
                )));
            }
        }
        if stress[3].out() != MAX_VOWELS {
            return Err(Error::Model(format!(
                "stress_clf.6: Ausgabe {} statt {MAX_VOWELS}",
                stress[3].out()
            )));
        }
        Ok(Self { embedding, stress })
    }

    /// Logits der Stress-Klassen für jedes Wort. Nur für die Golden-Tests.
    #[doc(hidden)]
    pub fn logits(&self, words: &[Vec<char>]) -> Vec<Vec<f32>> {
        words
            .iter()
            .map(|word| {
                let vector = self.embedding.word_vector(word);
                run_mlp(&self.stress, &vector)
            })
            .collect()
    }

    /// Setzt `+` vor den betonten Vokal; vorhandene `+` bleiben unangetastet.
    pub fn accent(&self, sentence: &[char], options: StressOptions) -> Vec<char> {
        let tokens = tokenize(sentence);
        let mut output = Vec::with_capacity(tokens.len());
        for token in &tokens {
            if !token.process {
                output.push(token.raw.clone());
                continue;
            }
            let raw_lower = lower_chars(&token.raw);
            if raw_lower.contains(&STRESS_TOKEN) {
                output.push(token.raw.clone());
                continue;
            }
            let vector = self.embedding.word_vector(&token.clean);
            let logits = run_mlp(&self.stress, &vector);
            output.push(apply_prediction(&token.raw, &logits, options));
        }
        output.concat()
    }
}

/// Ein Wortstück oder ein Trenner; `raw` bleibt für die Ausgabe, `clean` geht ins Modell.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Token {
    raw: Vec<char>,
    clean: Vec<char>,
    process: bool,
}

/// Portierung von `re.split(r'([\s.,!?;:<>=()/\\]+)', sentence)` und der
/// Bindestrich-Aufteilung. Leere Trenner werden nicht erzeugt; sie ändern die
/// Ausgabe nicht.
fn tokenize(sentence: &[char]) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut start = 0;
    let mut index = 0;
    while index < sentence.len() {
        if is_delimiter(sentence[index]) {
            if start < index {
                push_word(&sentence[start..index], &mut tokens);
            }
            let delimiter_start = index;
            while index < sentence.len() && is_delimiter(sentence[index]) {
                index += 1;
            }
            tokens.push(Token {
                raw: sentence[delimiter_start..index].to_vec(),
                clean: Vec::new(),
                process: false,
            });
            start = index;
        } else {
            index += 1;
        }
    }
    if start < sentence.len() {
        push_word(&sentence[start..], &mut tokens);
    }
    tokens
}

/// `word.split('-')`; alle Teile außer dem letzten behalten ihren Bindestrich.
fn push_word(word: &[char], tokens: &mut Vec<Token>) {
    let parts = split_on(word, '-');
    let last = parts.len() - 1;
    for (index, part) in parts.iter().enumerate() {
        let mut raw = part.to_vec();
        if index < last {
            raw.push('-');
        }
        // `silero-stress` baut `re_cond` aus einer Liste, nicht aus einem String;
        // das resultierende Muster matcht keine normalen Zeichen. Der Modellinput
        // ist deshalb nur die Kleinschreibung: Apostrophe und Bindestriche bleiben.
        let clean = lower_chars(&raw);
        let process = !clean.is_empty();
        tokens.push(Token {
            raw,
            clean,
            process,
        });
    }
}

fn split_on(word: &[char], separator: char) -> Vec<Vec<char>> {
    let mut parts = Vec::new();
    let mut current = Vec::new();
    for ch in word {
        if *ch == separator {
            parts.push(current);
            current = Vec::new();
        } else {
            current.push(*ch);
        }
    }
    parts.push(current);
    parts
}

/// Pythons `re` behandelt zusätzlich die Steuerzeichen U+001C..U+001F als `\s`.
fn is_delimiter(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '.' | ',' | '!' | '?' | ';' | ':' | '<' | '>' | '=' | '(' | ')' | '/' | '\\' | '\u{1c}'
                ..='\u{1f}'
        )
}

/// Kleinschreibung mit genau einem Zeichen je Eingabezeichen, damit die
/// Positionsindizes zu `raw` passen. Ukrainische Buchstaben sind 1:1; `İ` würde
/// bei Pythons `str.lower()` expandieren und die Indizes verschieben.
fn lower_chars(chars: &[char]) -> Vec<char> {
    chars
        .iter()
        .map(|ch| ch.to_lowercase().next().unwrap_or(*ch))
        .collect()
}

/// Setzt die Betonung für ein bereits vorbereitetes Wort.
fn apply_prediction(raw: &[char], logits: &[f32], options: StressOptions) -> Vec<char> {
    let raw_lower = lower_chars(raw);
    let (probs, prediction) = softmax_argmax(logits);
    let passed_threshold = probs[prediction] > STRESS_THRESHOLD;
    let (mut positions, num_vowels, first_vowel) = get_positions(&raw_lower, &[prediction as i32]);
    if num_vowels == 0 {
        return raw.to_vec();
    }
    let mut set_stress = passed_threshold;
    if num_vowels == 1 {
        positions = vec![first_vowel];
        set_stress = options.stress_single_vowel;
    }
    let mut output = raw.to_vec();
    if set_stress {
        for (offset, position) in positions.into_iter().enumerate() {
            let at = (position + offset).min(output.len());
            output.insert(at, STRESS_TOKEN);
        }
    }
    output
}

/// Char-Positionen der betonten Vokale, Anzahl der Vokale und erster Vokal.
fn get_positions(word: &[char], stressed_vowel_ids: &[i32]) -> (Vec<usize>, usize, usize) {
    let vowel_positions: Vec<usize> = word
        .iter()
        .enumerate()
        .filter(|(_, ch)| VOWELS.contains(**ch))
        .map(|(index, _)| index)
        .collect();
    let positions = stressed_vowel_ids
        .iter()
        .filter(|id| **id >= 0 && (**id as usize) < vowel_positions.len())
        .map(|id| vowel_positions[*id as usize])
        .collect();
    let first = vowel_positions.first().copied().unwrap_or(0);
    (positions, vowel_positions.len(), first)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(text: &str) -> Vec<char> {
        text.chars().collect()
    }

    fn logits_for(vowel: usize) -> Vec<f32> {
        let mut logits = vec![-100.0; MAX_VOWELS];
        logits[vowel] = 100.0;
        logits
    }

    #[test]
    fn tokenizer_keeps_raw_text_and_marks_words() {
        let text = "Привіт,  як справи?";
        let tokens = tokenize(&chars(text));
        assert_eq!(
            tokens.iter().map(|t| t.raw.len()).sum::<usize>(),
            text.chars().count()
        );
        let words: Vec<String> = tokens
            .iter()
            .filter(|token| token.process)
            .map(|token| token.clean.iter().collect())
            .collect();
        assert_eq!(words, ["привіт", "як", "справи"]);
        let raw: String = tokens.iter().flat_map(|token| token.raw.clone()).collect();
        assert_eq!(raw, text);
    }

    #[test]
    fn tokenizer_splits_hyphens_and_keeps_apostrophes() {
        let tokens = tokenize(&chars("пуху-пір'я"));
        let processed: Vec<(String, String)> = tokens
            .iter()
            .filter(|token| token.process)
            .map(|token| {
                (
                    token.raw.iter().collect(),
                    token.clean.iter().collect::<String>(),
                )
            })
            .collect();
        assert_eq!(
            processed,
            [
                ("пуху-".to_string(), "пуху-".to_string()),
                ("пір'я".to_string(), "пір'я".to_string()),
            ]
        );
    }

    #[test]
    fn tokenizer_keeps_letters_outside_the_ukrainian_alphabet() {
        // `silero-stress` filtert faktisch nichts; das Modell sieht `ъыэ` unverändert.
        let tokens = tokenize(&chars("ъыэ"));
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].process);
        assert_eq!(tokens[0].clean, chars("ъыэ"));
        let stressed = apply_prediction(&tokens[0].raw, &logits_for(0), StressOptions::default());
        assert_eq!(stressed, tokens[0].raw);
    }

    #[test]
    fn prediction_picks_vowel_ordinal() {
        let word = chars("молоко");
        let stressed = apply_prediction(&word, &logits_for(1), StressOptions::default());
        assert_eq!(stressed.iter().collect::<String>(), "мол+око");
    }

    #[test]
    fn one_vowel_word_is_stressed_only_with_option() {
        let word = chars("дім");
        let flat = vec![0.0; MAX_VOWELS];
        let with = apply_prediction(&word, &flat, StressOptions::default());
        assert_eq!(with.iter().collect::<String>(), "д+ім");
        let without = apply_prediction(
            &word,
            &flat,
            StressOptions {
                stress_single_vowel: false,
            },
        );
        assert_eq!(without.iter().collect::<String>(), "дім");
    }

    #[test]
    fn word_without_vowel_is_untouched() {
        let word = chars("рт");
        let stressed = apply_prediction(&word, &logits_for(0), StressOptions::default());
        assert_eq!(stressed, word);
    }

    #[test]
    fn prediction_beyond_word_vowels_does_not_insert() {
        let word = chars("молоко");
        let stressed = apply_prediction(&word, &logits_for(9), StressOptions::default());
        assert_eq!(stressed, word);
    }

    #[test]
    fn uppercase_is_preserved() {
        let word = chars("Молоко");
        let stressed = apply_prediction(&word, &logits_for(1), StressOptions::default());
        assert_eq!(stressed.iter().collect::<String>(), "Мол+око");
    }
}
