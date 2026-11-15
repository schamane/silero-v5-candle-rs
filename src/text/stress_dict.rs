//! Optionales Betonungswörterbuch aus `stress_dict.tsv`.
//!
//! Die Datei ist eine sortierte, UTF-8-kodierte Liste `wort<TAB>position`, wobei
//! `position` die 1-basierte Zeichenposition des betonten Vokals ist. Sie wird nicht
//! mitgeliefert: `scripts/export_stress_dict.py` erzeugt sie aus dem öffentlichen
//! Wiktionary-Stressdatensatz. Fehlt die Datei, bleibt das Modell unverändert.

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use crate::Result;
use crate::error::{Error, io_at};
use crate::text::accentor::is_vowel;

/// Sortierte Wörterbuchzeilen als Byte-Offsets in `data`; kein `String` je Eintrag.
pub(crate) struct StressDictionary {
    data: String,
    lines: Vec<u32>,
}

impl StressDictionary {
    /// Lädt `stress_dict.tsv` aus `dir`; `None`, wenn die Datei fehlt.
    ///
    /// # Errors
    ///
    /// Gibt [`Error::Io`] zurück, wenn die Datei nicht gelesen werden kann, und
    /// [`Error::Model`], wenn sie leer, unsortiert, doppelt oder syntaktisch ungültig ist.
    pub(crate) fn load(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join("stress_dict.tsv");
        let data = match fs::read_to_string(&path) {
            Ok(data) => data,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io_at(path)(error)),
        };
        Self::parse(data)
            .map(Some)
            .map_err(|error| Error::model_msg(format!("{}: {error}", path.display())))
    }

    /// Parst die Datei und validiert Sortierung, Eindeutigkeit und Vokalposition.
    fn parse(data: String) -> Result<Self> {
        let mut lines = Vec::new();
        let mut offset = 0usize;
        let mut previous: Option<&str> = None;
        for (line_no, chunk) in data.split_inclusive('\n').enumerate() {
            let line = chunk.strip_suffix('\n').unwrap_or(chunk);
            let line = line.strip_suffix('\r').unwrap_or(line);
            let Some((word, position)) = line.split_once('\t') else {
                return Err(Error::model_msg(format!(
                    "stress_dict.tsv: Zeile {} hat kein Tab",
                    line_no + 1
                )));
            };
            if word.is_empty() {
                return Err(Error::model_msg(format!(
                    "stress_dict.tsv: leeres Wort in Zeile {}",
                    line_no + 1
                )));
            }
            let position: usize = position.parse().map_err(|_| {
                Error::model_msg(format!(
                    "stress_dict.tsv: ungültige Position {position:?} in Zeile {}",
                    line_no + 1
                ))
            })?;
            if position == 0 {
                return Err(Error::model_msg(format!(
                    "stress_dict.tsv: Position 0 in Zeile {}",
                    line_no + 1
                )));
            }
            let Some(stressed) = word.chars().nth(position - 1) else {
                return Err(Error::model_msg(format!(
                    "stress_dict.tsv: Position {position} außerhalb von {word:?} in Zeile {}",
                    line_no + 1
                )));
            };
            if !is_vowel(stressed) {
                return Err(Error::model_msg(format!(
                    "stress_dict.tsv: Position {position} zeigt in {word:?} auf {stressed:?}, keinen Vokal (Zeile {})",
                    line_no + 1
                )));
            }
            if previous.is_some_and(|prev| prev >= word) {
                return Err(Error::model_msg(format!(
                    "stress_dict.tsv: {word:?} ist doppelt oder unsortiert (Zeile {})",
                    line_no + 1
                )));
            }
            previous = Some(word);
            lines
                .push(u32::try_from(offset).map_err(|_| {
                    Error::model_msg("stress_dict.tsv: Datei ist größer als 4 GiB")
                })?);
            offset += chunk.len();
        }
        Ok(Self { data, lines })
    }

    /// 0-basierte Zeichenposition des betonten Vokals, falls `word` enthalten ist.
    pub(crate) fn stress(&self, word: &str) -> Option<usize> {
        let index = self
            .lines
            .binary_search_by(|offset| {
                let (candidate, _) = self.entry(*offset as usize);
                candidate.cmp(word)
            })
            .ok()?;
        let (_, position) = self.entry(self.lines[index] as usize);
        position.parse::<usize>().ok()?.checked_sub(1)
    }

    fn entry(&self, offset: usize) -> (&str, &str) {
        let end = self.data[offset..]
            .find('\n')
            .map_or(self.data.len(), |newline| offset + newline);
        let line = self.data[offset..end]
            .strip_suffix('\r')
            .unwrap_or(&self.data[offset..end]);
        line.split_once('\t')
            .expect("StressDictionary validiert beim Laden jedes Tab")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(data: &str) -> StressDictionary {
        StressDictionary::parse(data.to_owned()).expect("gültiges Wörterbuch")
    }

    #[test]
    fn finds_stress_and_converts_to_zero_based() {
        let dictionary = parsed("столе\t5\nтикали\t2\n");
        assert_eq!(dictionary.stress("столе"), Some(4));
        assert_eq!(dictionary.stress("тикали"), Some(1));
        assert_eq!(dictionary.stress("нет"), None);
    }

    #[test]
    fn rejects_unsorted_or_duplicate_words() {
        for data in ["тикали\t2\nстоле\t5\n", "столе\t5\nстоле\t5\n"] {
            assert!(
                StressDictionary::parse(data.to_owned()).is_err(),
                "{data:?}"
            );
        }
    }

    #[test]
    fn rejects_malformed_positions() {
        for data in ["столе\n", "столе\t0\n", "столе\t6\n", "столе\t2\n"] {
            assert!(
                StressDictionary::parse(data.to_owned()).is_err(),
                "{data:?}"
            );
        }
    }

    #[test]
    fn missing_file_is_not_an_error() {
        let directory = std::env::temp_dir().join("silero-stress-dict-missing");
        let _ = std::fs::create_dir_all(&directory);
        assert!(
            StressDictionary::load(&directory)
                .expect("optional")
                .is_none()
        );
    }
}
