//! Textvorbereitung, Satzaufteilung und Symbol-IDs.
//!
//! Portiert `PartTTSModelMultiAcc_v3.prepare_text_input` und
//! `clean_star_text`; gearbeitet wird ausschließlich auf `Vec<char>`, damit
//! Indizes wie in Pythons Codepoint-Strings bleiben.

use crate::config::Config;

/// Ergebnis von `prepare_text_input`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedText {
    /// Gefilterter Text mit `^`-Sternsyntax.
    pub sentence: Vec<char>,
    /// Nur `[а-я\- ]`.
    pub clean: Vec<char>,
    pub has_text: bool,
}

/// `prepare_text_input` für Klartext (kein Phonem-Modus).
pub fn prepare_text_input(config: &Config, text: &str) -> PreparedText {
    let allowed = config.allowed_chars();
    // Kleinschreibung, Zeichenersetzung und Filter wirken jeweils auf ein Zeichen
    // ohne Bezug zu den Nachbarn; ein Durchlauf liefert daher dieselbe Folge.
    let mut filtered = String::with_capacity(text.len());
    for ch in text.chars() {
        // Python `str.lower`; für die gefilterten Zeichen genügt die
        // zeichenweise Abbildung.
        for lower in ch.to_lowercase() {
            let lower = match lower {
                '\u{2014}' => '\u{2013}', // — -> –
                '\u{2011}' => '-',        // ‑ -> -
                other => other,
            };
            if allowed.contains(&lower) {
                filtered.push(lower);
            }
        }
    }
    let sentence = clean_star_text(&filtered);
    let clean: Vec<char> = sentence
        .iter()
        .copied()
        .filter(|ch| matches!(ch, 'а'..='я' | '-' | ' '))
        .collect();
    let has_text = sentence.iter().any(|ch| is_text_char(*ch));
    PreparedText {
        sentence,
        clean,
        has_text,
    }
}

/// `clean_star_text`: kollabiert Whitespace, entfernt Leerzeichen um `^` und
/// fasst `^^` zusammen.
pub fn clean_star_text(text: &str) -> Vec<char> {
    let mut work = String::with_capacity(text.len() + 1);
    for ch in text.chars() {
        if ch == '^' {
            work.push_str(" ^");
        } else {
            work.push(ch);
        }
    }
    let collapsed = collapse_whitespace(&work);
    let without_space = collapse_carets(&collapsed.replace("^ ", "^"));
    let mut trimmed = without_space.trim();
    if let Some(rest) = trimmed.strip_suffix('^') {
        trimmed = rest;
    }
    trimmed.trim().chars().collect()
}

fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !space {
                out.push(' ');
                space = true;
            }
        } else {
            out.push(ch);
            space = false;
        }
    }
    out.trim().to_string()
}

fn collapse_carets(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut caret = false;
    for ch in text.chars() {
        if ch == '^' {
            if !caret {
                out.push('^');
                caret = true;
            }
        } else {
            out.push(ch);
            caret = false;
        }
    }
    out
}

/// Satzaufteilung in Chunks (400 Zeichen, Codepoints); `scripts/export_silero.py`
/// spiegelt sie für die Goldens.
pub fn split_chunks(text: &str, limit: usize) -> Vec<String> {
    let limit = limit.max(3);
    let mut chunks = Vec::new();
    for sentence in split_sentences(text) {
        let mut sentence: Vec<char> = sentence.trim().chars().collect();
        while sentence.len() > limit {
            let mut cut = last_substring(&sentence, ", ", limit)
                .max(last_substring(&sentence, "; ", limit))
                .max(last_substring(&sentence, " — ", limit));
            if cut < limit / 3 {
                cut = last_substring(&sentence, " ", limit);
            }
            if cut < limit / 3 {
                cut = limit;
            }
            let head: String = sentence[..cut].iter().collect();
            let head = head.trim();
            if !head.is_empty() {
                chunks.push(head.to_string());
            }
            sentence = sentence[cut..].to_vec();
            while matches!(sentence.first(), Some(' ' | ',' | ';')) {
                sentence.remove(0);
            }
        }
        let tail: String = sentence.iter().collect();
        let tail = tail.trim();
        if !tail.is_empty() {
            chunks.push(tail.to_string());
        }
    }
    chunks
}

/// `SENTENCE_SPLIT = (?<=[.!?…])\s+|\n+`.
fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        let boundary = if matches!(chars[i], '.' | '!' | '?' | '…') {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            (j > i + 1).then_some((i + 1, j))
        } else if chars[i] == '\n' {
            let mut j = i;
            while j < chars.len() && chars[j] == '\n' {
                j += 1;
            }
            Some((i, j))
        } else {
            None
        };
        if let Some((end, next)) = boundary {
            pieces.push(chars[start..end].iter().collect());
            start = next;
            i = next;
        } else {
            i += 1;
        }
    }
    if start < chars.len() {
        pieces.push(chars[start..].iter().collect());
    }
    pieces
}

/// Letzter Startindex von `needle` mit `start + needle.len() <= limit`.
fn last_substring(chars: &[char], needle: &str, limit: usize) -> usize {
    let needle: Vec<char> = needle.chars().collect();
    let mut found = 0;
    if needle.is_empty() || chars.len() < needle.len() {
        return found;
    }
    for start in 0..=chars.len() - needle.len() {
        if start + needle.len() > limit {
            break;
        }
        if chars[start..start + needle.len()] == needle[..] {
            found = start;
        }
    }
    found
}

/// Enthält `text` ein Zeichen, das `has_text` zählt (`а-я` oder `ё` nach `lower()`)?
pub fn has_cyrillic(text: &str) -> bool {
    text.chars().any(|ch| ch.to_lowercase().any(is_text_char))
}

/// Buchstaben, die als Sprache zählen: `а-я` und `ё` (nach `lower()`).
fn is_text_char(ch: char) -> bool {
    matches!(ch, 'а'..='я' | 'ё')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_at_sentence_boundaries() {
        assert_eq!(
            split_chunks("Привет! Как дела? Хорошо.", 400),
            vec!["Привет!", "Как дела?", "Хорошо."]
        );
        assert_eq!(
            split_chunks("Первая строка\nВторая", 400),
            vec!["Первая строка", "Вторая"]
        );
        assert_eq!(split_chunks("Привет… и что", 400), vec!["Привет…", "и что"]);
    }

    #[test]
    fn breaks_overlong_chunks_at_commas() {
        let text = format!("{}, и ещё немного текста.", "а".repeat(300));
        let chunks = split_chunks(&text, 400);
        assert_eq!(chunks.len(), 1);
        let text = format!(
            "{}, {}, и ещё немного текста.",
            "а".repeat(250),
            "б".repeat(250)
        );
        let chunks = split_chunks(&text, 400);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 400));
    }

    #[test]
    fn tiny_limits_still_make_progress() {
        assert_eq!(
            split_chunks("abcdef", 0),
            vec!["abc".to_string(), "def".to_string()]
        );
    }

    #[test]
    fn cyrillic_filter_counts_yo() {
        assert!(has_cyrillic("Привет"));
        assert!(has_cyrillic("Ёж"));
        assert!(has_cyrillic("ё"));
        assert!(has_cyrillic("Ё!"));
        assert!(!has_cyrillic("hello 123"));
        assert!(!has_cyrillic("!?."));
        assert!(!has_cyrillic(""));
    }

    /// Liest `models/silero-v5-ru/config.json` über `Config::load`, weil nur dort die
    /// Lookup-Tabellen entstehen. Fehlt die Datei, wird übersprungen, mit
    /// `SILERO_REQUIRE_GOLDEN` panikt der Test stattdessen.
    fn model_config() -> Option<Config> {
        use std::io::Write;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models/silero-v5-ru/config.json");
        if !path.is_file() {
            if std::env::var_os("SILERO_REQUIRE_GOLDEN").is_some() {
                panic!(
                    "SILERO_REQUIRE_GOLDEN gesetzt, aber {} fehlt",
                    path.display()
                );
            }
            let _ = writeln!(std::io::stderr(), "SKIP: {} fehlt", path.display());
            return None;
        }
        Some(Config::load(&path).unwrap_or_else(|err| panic!("config.json lädt nicht: {err}")))
    }

    fn as_string(chars: &[char]) -> String {
        chars.iter().collect()
    }

    #[test]
    fn empty_input_stays_empty() {
        let Some(config) = model_config() else { return };
        let prepared = prepare_text_input(&config, "");
        assert_eq!(as_string(&prepared.sentence), "");
        assert_eq!(as_string(&prepared.clean), "");
        assert!(!prepared.has_text);
        assert!(clean_star_text("").is_empty());
        assert!(split_chunks("", 400).is_empty());
        // Nur Whitespace ergibt keinen Chunk.
        assert!(split_chunks("  \n  ", 400).is_empty());
        assert!(!has_cyrillic(""));
    }

    #[test]
    fn tabs_and_newlines_collapse_in_clean_star_text() {
        assert_eq!(as_string(&clean_star_text("a\t\tb\n\nc")), "a b c");
        assert_eq!(as_string(&clean_star_text("\n\tx\t")), "x");
    }

    /// Pinnt Ist-Verhalten: `prepare_text_input` filtert Tab und Zeilenumbruch
    /// (sie stehen nicht in `symbols`), bevor `clean_star_text` sie zu Leerzeichen
    /// machen könnte. Ein Tab mitten im Satz verschwindet also ersatzlos.
    #[test]
    fn tab_inside_sentence_is_dropped_by_prepare() {
        let Some(config) = model_config() else { return };
        let prepared = prepare_text_input(&config, "привет\tмир");
        assert_eq!(as_string(&prepared.sentence), "приветмир");
    }

    /// Pinnt Ist-Verhalten der `^`-Syntax: Vor jedem `^` steht genau ein Leerzeichen,
    /// das Leerzeichen danach fällt weg, `^^` wird zu `^`, und ein nachgestelltes
    /// `^` entfällt.
    #[test]
    fn caret_syntax_through_clean_star_text() {
        assert_eq!(as_string(&clean_star_text("a^b")), "a ^b");
        assert_eq!(as_string(&clean_star_text("a ^ b")), "a ^b");
        assert_eq!(as_string(&clean_star_text("a^^b")), "a ^b");
        assert_eq!(as_string(&clean_star_text("abc^")), "abc");
        assert_eq!(as_string(&clean_star_text("^")), "");
        assert_eq!(as_string(&clean_star_text("^abc")), "^abc");
    }

    #[test]
    fn uppercase_yo_is_lowercased_and_counts_as_text() {
        let Some(config) = model_config() else { return };
        let prepared = prepare_text_input(&config, "Ёж");
        assert_eq!(as_string(&prepared.sentence), "ёж");
        // `clean` führt nur `а-я`, `-` und Leerzeichen; `ё` fällt dort heraus.
        assert_eq!(as_string(&prepared.clean), "ж");
        assert!(prepared.has_text);
        assert_eq!(as_string(&prepare_text_input(&config, "ЁЖ").sentence), "ёж");
    }

    /// Pinnt Ist-Verhalten: Kombinierende Zeichen fehlen in `symbols` und fallen
    /// ersatzlos weg, ohne vorherige Unicode-Normalisierung. Das zerlegte `ё`
    /// (`е` + U+0308) wird damit zu `е`.
    #[test]
    fn combining_marks_are_dropped() {
        let Some(config) = model_config() else { return };
        let prepared = prepare_text_input(&config, "е\u{0308}ж");
        assert_eq!(as_string(&prepared.sentence), "еж");
        assert_eq!(as_string(&prepared.clean), "еж");
        assert!(prepared.has_text);
        let only_mark = prepare_text_input(&config, "\u{0301}");
        assert!(only_mark.sentence.is_empty());
        assert!(!only_mark.has_text);
    }

    #[test]
    fn punctuation_only_sentence_has_no_text() {
        let Some(config) = model_config() else { return };
        let prepared = prepare_text_input(&config, "!?.,;:");
        assert_eq!(as_string(&prepared.sentence), "!?.,;:");
        assert!(prepared.clean.is_empty());
        assert!(!prepared.has_text);
        assert!(!has_cyrillic("!?.,;:"));
        assert_eq!(split_chunks("!?.", 400), vec!["!?."]);
    }

    /// Ein Chunk aus nur `Ё!` passt zum Chunk-Filter und zu `has_text`.
    #[test]
    fn yo_only_chunk_passes_chunk_filter() {
        let Some(config) = model_config() else { return };
        let chunks = split_chunks("Ё!", 400);
        assert_eq!(chunks, vec!["Ё!"]);
        for chunk in &chunks {
            assert!(has_cyrillic(chunk), "Chunk {chunk:?} fehlt im Filter");
            assert!(prepare_text_input(&config, chunk).has_text);
        }
    }

    /// Chunk-Filter und `has_text` stimmen für alle Texte überein.
    #[test]
    fn chunk_filter_agrees_with_has_text() {
        let Some(config) = model_config() else { return };
        for text in ["Ё!", "ё", "Привет", "hello", "-", "- — !", "!?.", "1, 2"] {
            assert_eq!(
                has_cyrillic(text),
                prepare_text_input(&config, text).has_text,
                "Text {text:?}"
            );
        }
    }

    #[test]
    fn prepare_filters_latin_and_digits() {
        let Some(config) = model_config() else { return };
        let prepared = prepare_text_input(&config, "Привет, world 2026 — да!");
        assert_eq!(
            prepared.sentence.iter().collect::<String>(),
            "привет, – да!"
        );
        assert_eq!(prepared.clean.iter().collect::<String>(), "привет  да");
        assert!(prepared.has_text);
    }

    #[test]
    fn hyphen_only_is_empty_text() {
        let Some(config) = model_config() else { return };
        assert!(!prepare_text_input(&config, "-").has_text);
        assert!(!prepare_text_input(&config, "- — !").has_text);
        assert!(prepare_text_input(&config, "Привет").has_text);
    }
}
