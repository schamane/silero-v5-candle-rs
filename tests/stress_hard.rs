//! Pins the hard-case stress-suite scores from `tests/data/`.
//!
//! RU requires `models/silero-v5-ru` with the generated `stress_dict.tsv`;
//! UK requires `models/silero-v5-cis-base-nostress`. Missing inputs skip the
//! test unless `SILERO_REQUIRE_GOLDEN` is set.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use candle_core::Device;
use serde::Deserialize;
use silero_v5_candle::Silero;

#[derive(Deserialize)]
struct Case {
    id: String,
    text: String,
    target: String,
    gold: String,
    #[serde(default)]
    accept: Vec<String>,
}

fn suite(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn skip(reason: &str) {
    if std::env::var_os("SILERO_REQUIRE_GOLDEN").is_some() {
        panic!("SILERO_REQUIRE_GOLDEN gesetzt: {reason}");
    }
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

fn form(token: &str) -> String {
    token
        .trim_matches(|c: char| !(c.is_alphanumeric() || c == '+'))
        .to_lowercase()
}

fn bare(token: &str) -> String {
    form(token).replace('+', "")
}

/// Same rule as `scripts/stress_bench_hard.py`: one researched target per sentence.
fn score(cases: &[Case], lines: &[String]) -> (usize, Vec<String>) {
    assert_eq!(
        cases.len(),
        lines.len(),
        "eine Ausgabezeile pro Fall erwartet"
    );
    let mut passed = 0;
    let mut errors = Vec::new();
    for (case, line) in cases.iter().zip(lines) {
        let target = case.target.to_lowercase();
        let expected: Vec<String> = std::iter::once(&case.gold)
            .chain(case.accept.iter())
            .map(|gold| gold.to_lowercase())
            .collect();
        let tokens: Vec<String> = line.split_whitespace().map(form).collect();
        let bare_tokens: Vec<String> = line.split_whitespace().map(bare).collect();
        if !bare_tokens.contains(&target) {
            errors.push(format!("{}: Ziel {target:?} fehlt in {line:?}", case.id));
        } else if tokens.iter().any(|token| expected.contains(token)) {
            passed += 1;
        } else {
            let got: Vec<&String> = tokens
                .iter()
                .filter(|token| {
                    let naked = bare(token);
                    naked.contains(&target) || target.contains(naked.as_str())
                })
                .collect();
            errors.push(format!(
                "{}: {target:?} erwartet {expected:?}, gelesen {got:?} in {line:?}",
                case.id
            ));
        }
    }
    (passed, errors)
}

fn run(dir: &str, name: &str, voice: &str, minimum: usize, needs_dict: bool) {
    if cfg!(debug_assertions) {
        skip("Test braucht --release (Modelllast)");
        return;
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir);
    if !dir.is_dir() {
        skip(&format!("{} fehlt", dir.display()));
        return;
    }
    if needs_dict && !dir.join("stress_dict.tsv").is_file() {
        skip(&format!(
            "{} fehlt; scripts/export_stress_dict.py erzeugt sie",
            dir.join("stress_dict.tsv").display()
        ));
        return;
    }
    let cases: Vec<Case> =
        serde_json::from_str(&fs::read_to_string(suite(name)).expect("Suite ist lesbar"))
            .expect("Suite ist gültiges JSON");
    let model = Silero::load_with_device(&dir, &Device::Cpu).expect("Modell lädt");
    let lines: Vec<String> = cases
        .iter()
        .map(|case| {
            model
                .accent_text(&case.text, voice)
                .unwrap_or_else(|error| panic!("{}: {error}", case.id))
        })
        .collect();
    let (passed, errors) = score(&cases, &lines);
    println!("{name}: {passed}/{} (mindestens {minimum})", cases.len());
    assert!(
        passed >= minimum,
        "{name}: nur {passed}/{} betont, mindestens {minimum} erwartet; Fehler:\n{}",
        cases.len(),
        errors.join("\n")
    );
}

#[test]
fn ru_hard_suite_keeps_58_of_60() {
    run(
        "models/silero-v5-ru",
        "stress_hard60.json",
        "baya",
        58,
        true,
    );
}

#[test]
fn uk_hard_suite_keeps_45_of_60() {
    run(
        "models/silero-v5-cis-base-nostress",
        "stress_hard60_uk.json",
        "ukr_igor",
        45,
        false,
    );
}
