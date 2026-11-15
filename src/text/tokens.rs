//! Gemeinsame Bausteine der beiden Akzentoren: Wortstücke, Bindestrich-Aufteilung,
//! zeichenweise Kleinschreibung und die Abbildung von Vokal-Ordnungszahlen auf Positionen.

/// Ein Wortstück oder ein Trenner; `raw` bleibt für die Ausgabe, `clean` geht ins Modell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Token {
    pub(crate) raw: Vec<char>,
    pub(crate) clean: Vec<char>,
    pub(crate) process: bool,
}

/// Teilt `word` an `separator`; ein leerer Teil bleibt erhalten, es gibt immer mindestens einen.
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

/// `word.split('-')`; alle Teile außer dem letzten behalten ihren Bindestrich.
/// Das zweite Element ist `true` für den letzten Teil.
pub(crate) fn hyphen_parts(word: &[char]) -> impl Iterator<Item = (Vec<char>, bool)> {
    let parts = split_on(word, '-');
    let last = parts.len() - 1;
    parts.into_iter().enumerate().map(move |(index, mut raw)| {
        if index < last {
            raw.push('-');
        }
        (raw, index == last)
    })
}

/// Kleinschreibung mit genau einem Zeichen je Eingabezeichen, damit die
/// Positionsindizes zu `raw` passen (`İ` würde sonst zu zwei Zeichen).
pub(crate) fn lower_chars(chars: &[char]) -> Vec<char> {
    chars
        .iter()
        .map(|ch| ch.to_lowercase().next().unwrap_or(*ch))
        .collect()
}

/// Bildet Vokal-Ordnungszahlen auf Positionen ab; Zahlen außerhalb von `0..positions.len()`
/// entfallen.
pub(crate) fn ordinal_positions(positions: &[usize], ordinals: &[i32]) -> Vec<usize> {
    ordinals
        .iter()
        .filter_map(|ordinal| usize::try_from(*ordinal).ok())
        .filter_map(|ordinal| positions.get(ordinal).copied())
        .collect()
}
