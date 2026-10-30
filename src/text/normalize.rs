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
    let mut lowered = String::with_capacity(text.len());
    for ch in text.chars() {
        // Python `str.lower`; für die gefilterten Zeichen genügt die
        // zeichenweise Abbildung.
        for lower in ch.to_lowercase() {
            lowered.push(lower);
        }
    }
    let mut filtered = String::with_capacity(lowered.len());
    for ch in lowered.chars() {
        let ch = match ch {
            '\u{2014}' => '\u{2013}', // — -> –
            '\u{2011}' => '-',        // ‑ -> -
            other => other,
        };
        if allowed.contains(&ch) {
            filtered.push(ch);
        }
    }
    let sentence = clean_star_text(&filtered);
    let clean: Vec<char> = sentence
        .iter()
        .copied()
        .filter(|ch| matches!(ch, 'а'..='я' | '-' | ' '))
        .collect();
    let has_text = sentence.iter().any(|ch| matches!(ch, 'а'..='я' | 'ё'));
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
    let mut without_space: String = collapsed.replace("^ ", "^");
    without_space = collapse_carets(&without_space);
    let mut chars: Vec<char> = without_space.trim().chars().collect();
    if chars.last() == Some(&'^') {
        chars.pop();
    }
    let trimmed: String = chars.iter().collect();
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

/// Satzaufteilung wie `scripts/say.py::split_chunks` (400 Zeichen, Codepoints).
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

/// `CYRILLIC = [а-я]` nach `lower()`; `ё` zählt nicht.
pub fn has_cyrillic(text: &str) -> bool {
    text.chars()
        .any(|ch| ch.to_lowercase().any(|lower| matches!(lower, 'а'..='я')))
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
    fn cyrillic_filter_ignores_yo() {
        assert!(has_cyrillic("Привет"));
        assert!(has_cyrillic("Ёж"));
        assert!(!has_cyrillic("ё"));
        assert!(!has_cyrillic("hello 123"));
    }

    #[test]
    fn prepare_filters_latin_and_digits() {
        let config_json = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("models/silero-v5-ru/config.json"),
        );
        let Ok(config_json) = config_json else { return };
        let config: crate::config::Config = serde_json::from_str(&config_json).unwrap();
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
        let config_json = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("models/silero-v5-ru/config.json"),
        );
        let Ok(config_json) = config_json else { return };
        let config: crate::config::Config = serde_json::from_str(&config_json).unwrap();
        assert!(!prepare_text_input(&config, "-").has_text);
        assert!(!prepare_text_input(&config, "- — !").has_text);
        assert!(prepare_text_input(&config, "Привет").has_text);
    }
}
