//! Golden für den ukrainischen Akzentor aus `silero-stress`.
//!
//! Der Export liegt unter `models/silero-v5-cis-base-nostress/` und ist wie alle
//! Modellgewichte nicht eingecheckt. Ohne Verzeichnis wird der Test übersprungen.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use silero_v5_candle::text::accentor_uk::{StressOptions, UkrainianAccentor};

fn model_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models/silero-v5-cis-base-nostress")
}

fn skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

fn require_golden() -> bool {
    std::env::var_os("SILERO_REQUIRE_GOLDEN").is_some()
}

fn load_accentor() -> Option<UkrainianAccentor> {
    let dir = model_dir();
    let golden = dir.join("golden/accentor.json");
    if !golden.is_file() {
        if require_golden() {
            panic!(
                "SILERO_REQUIRE_GOLDEN gesetzt, aber {} fehlt",
                golden.display()
            );
        }
        skip(&format!(
            "{} fehlt; Akzentor-Golden übersprungen",
            golden.display()
        ));
        return None;
    }
    Some(UkrainianAccentor::load(&dir, &Device::Cpu).expect("Akzentor lädt"))
}

fn values(tensor: &candle_core::Tensor) -> Vec<f32> {
    tensor
        .flatten_all()
        .expect("Tensor lässt sich flachen")
        .to_vec1::<f32>()
        .expect("Tensor ist als f32 lesbar")
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for value in &mut probs {
        *value /= sum;
    }
    probs
}

#[test]
fn accentor_matches_python() {
    if cfg!(debug_assertions) {
        if require_golden() {
            panic!("SILERO_REQUIRE_GOLDEN gesetzt, aber Golden-Tests brauchen --release");
        }
        skip("Golden-Tests brauchen --release");
        return;
    }
    let Some(accentor) = load_accentor() else {
        return;
    };
    let dir = model_dir();
    let text = std::fs::read_to_string(dir.join("golden/accentor.json"))
        .expect("accentor.json ist lesbar");
    let golden: serde_json::Value =
        serde_json::from_str(&text).expect("accentor.json ist gültiges JSON");
    let words: Vec<String> = golden["words"]
        .as_array()
        .expect("words ist ein Array")
        .iter()
        .map(|word| word.as_str().expect("word ist ein String").to_string())
        .collect();
    let clean: Vec<Vec<char>> = words.iter().map(|word| word.chars().collect()).collect();
    let logits = accentor.logits(&clean);

    let expected =
        candle_core::safetensors::load(dir.join("golden/accentor.safetensors"), &Device::Cpu)
            .expect("Akzentor-Golden lädt");
    let expected = expected["logits"]
        .to_dtype(DType::F32)
        .expect("logits sind f32");
    assert_eq!(expected.dims(), [words.len(), 10], "logits: Shape");
    let expected = values(&expected);
    assert_eq!(logits.len(), words.len(), "logits: Anzahl");
    let mut max = 0.0f32;
    for (index, row) in logits.iter().enumerate() {
        assert_eq!(row.len(), 10, "logits[{index}]: Länge");
        for (class, value) in row.iter().enumerate() {
            let wanted = expected[index * 10 + class];
            let diff = (value - wanted).abs();
            max = max.max(diff);
            // Große negative Logits streuen durch die f32-Summierung relativ;
            // entscheidend sind Argmax und Softmax-Wahrscheinlichkeit.
            let tolerance = 1e-4 + 1e-5 * wanted.abs();
            assert!(
                diff <= tolerance,
                "logits[{index}][{class}] ({:?}): {value} statt {wanted} (Δ {diff} > {tolerance})",
                words[index]
            );
            let actual_probs = softmax(row);
            let wanted_probs = softmax(&expected[index * 10..(index + 1) * 10]);
            assert!(
                (actual_probs[class] - wanted_probs[class]).abs() <= 1e-5,
                "probs[{index}][{class}] ({:?})",
                words[index]
            );
        }
    }
    eprintln!("accentor logits: max {max:.3e}");

    for sentence in golden["sentences"]
        .as_array()
        .expect("sentences ist ein Array")
    {
        let raw = sentence["text"].as_str().expect("text ist ein String");
        let wanted = sentence["accented"]
            .as_str()
            .expect("accented ist ein String");
        let actual: String = accentor
            .accent(&raw.chars().collect::<Vec<_>>(), StressOptions::default())
            .iter()
            .collect();
        assert_eq!(actual, wanted, "Satz {raw:?}: Akzentor");
    }
}
