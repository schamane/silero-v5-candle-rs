//! Vergleich gegen die Goldens aus `scripts/export_silero.py`.
//!
//! Läuft ohne Modellverzeichnis als übersprungener Test, damit
//! `cargo test --workspace` ohne die (nicht eingecheckten) Gewichte baut.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use silero_v5_candle::Silero;
use silero_v5_candle::acoustic::forward::AcousticOutput;

/// Meldet einen Skip direkt auf stderr: libtest fängt `eprintln!` bei bestandenen Tests ab.
fn skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

/// Mit `SILERO_REQUIRE_GOLDEN` (beliebiger Wert) werden fehlende Goldens zum Fehler statt zum Skip.
fn require_golden() -> bool {
    std::env::var_os("SILERO_REQUIRE_GOLDEN").is_some()
}

fn model_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models/silero-v5-ru")
}

fn load_model() -> Option<Silero> {
    let dir = model_dir();
    if !dir.is_dir() {
        if require_golden() {
            panic!(
                "SILERO_REQUIRE_GOLDEN gesetzt, aber {} fehlt",
                dir.display()
            );
        }
        skip(&format!(
            "{} fehlt; Golden-Tests übersprungen",
            dir.display()
        ));
        return None;
    }
    Some(Silero::load(&dir).expect("Modell lädt"))
}

fn golden(path: &Path) -> HashMap<String, Tensor> {
    candle_core::safetensors::load(path, &Device::Cpu).expect("Golden lädt")
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .flatten_all()
        .expect("Tensor lässt sich flachen")
        .to_vec1::<f32>()
        .expect("Tensor ist als f32 lesbar")
}

fn compare(name: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) {
    assert_eq!(
        actual.dims(),
        expected.dims(),
        "{name}: Shape {:?} statt {:?}",
        actual.dims(),
        expected.dims()
    );
    let actual = values(actual);
    let expected = values(expected);
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: Länge {} statt {}",
        actual.len(),
        expected.len()
    );
    let mut max = 0.0f32;
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        let diff = (a - e).abs();
        if diff > max {
            max = diff;
        }
        assert!(
            diff <= tolerance,
            "{name}: Abweichung {diff} bei Index {index} ({a} statt {e}) > {tolerance}"
        );
    }
    eprintln!("{name}: max {max:.3e}");
}

fn compare_audio(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: Länge {} statt {}",
        actual.len(),
        expected.len()
    );
    let mut max = 0.0f32;
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        if diff > max {
            max = diff;
        }
        assert!(
            diff <= tolerance,
            "{name}: Abweichung {diff} bei Sample {index} ({a} statt {e}) > {tolerance}"
        );
    }
    eprintln!("{name}: max {max:.3e}");
}

fn check_stages(
    name: &str,
    output: &AcousticOutput,
    tensors: &HashMap<String, Tensor>,
    full: bool,
) {
    let expected_dur: Vec<u32> = values(&tensors["dur"])
        .into_iter()
        .map(|value| value as u32)
        .collect();
    assert_eq!(output.dur, expected_dur, "{name}: dur_hat");
    compare(
        &format!("{name} dur_raw"),
        &output.dur_raw,
        &tensors["dur_raw"],
        1e-4,
    );
    compare(
        &format!("{name} pitch_raw"),
        &output.pitch_raw,
        &tensors["pitch_raw"],
        1e-3,
    );
    compare(
        &format!("{name} pitch"),
        &output.pitch,
        &tensors["pitch"],
        1e-4,
    );
    compare(&format!("{name} mel"), &output.mel, &tensors["mel"], 1e-3);
    if full {
        compare(
            &format!("{name} encoder"),
            &output.encoder,
            &tensors["encoder"],
            1e-3,
        );
        compare(
            &format!("{name} cond"),
            &output.cond,
            &tensors["cond"],
            1e-3,
        );
        compare(
            &format!("{name} expanded"),
            &output.expanded,
            &tensors["expanded"],
            1e-3,
        );
        compare(
            &format!("{name} decoder"),
            &output.decoder,
            &tensors["decoder"],
            1e-3,
        );
    }
}

#[test]
fn goldens_match_python() {
    if cfg!(debug_assertions) {
        if require_golden() {
            panic!(
                "SILERO_REQUIRE_GOLDEN gesetzt, aber Golden-Tests brauchen --release (candle im Debug-Build zu langsam)"
            );
        }
        skip("Golden-Tests brauchen --release (candle im Debug-Build zu langsam)");
        return;
    }
    let Some(model) = load_model() else { return };
    let dir = model_dir();
    let golden_dir = dir.join("golden");
    let texts =
        std::fs::read_to_string(golden_dir.join("text.jsonl")).expect("golden/text.jsonl lesbar");
    let entries: Vec<serde_json::Value> = texts
        .lines()
        .map(|line| serde_json::from_str(line).expect("text.jsonl Zeile ist gültiges JSON"))
        .collect();
    for entry in &entries {
        let id = entry["id"].as_u64().expect("Feld id ist eine Zahl");
        let text = entry["text"].as_str().expect("Feld text ist ein String");
        let prepared = model.prepare(text);
        assert_eq!(
            prepared.sentence.iter().collect::<String>(),
            entry["prepared"]
                .as_str()
                .expect("Feld prepared ist ein String"),
            "Satz {id}: prepared"
        );
        assert_eq!(
            prepared.clean.iter().collect::<String>(),
            entry["clean"].as_str().expect("Feld clean ist ein String")
        );
        assert_eq!(
            prepared.has_text,
            entry["has_text"]
                .as_bool()
                .expect("Feld has_text ist ein Bool")
        );
        if !prepared.has_text {
            continue;
        }
        let stages = model.debug_stages(text, "baya").expect("Pipeline");
        assert_eq!(
            stages.homo.iter().collect::<String>(),
            entry["homo"].as_str().expect("Feld homo ist ein String"),
            "Satz {id}: homosolver"
        );
        assert_eq!(
            stages.accented.iter().collect::<String>(),
            entry["accented"]
                .as_str()
                .expect("Feld accented ist ein String"),
            "Satz {id}: accentor"
        );
        let sequence: Vec<u32> = entry["sequence"]
            .as_array()
            .expect("Feld sequence ist ein Array")
            .iter()
            .map(|value| value.as_u64().expect("sequence enthält nur Zahlen") as u32)
            .collect();
        assert_eq!(stages.ids, sequence, "Satz {id}: sequence");

        // Homosolver-Logits (nur Sätze mit Homographen).
        let homo_path = golden_dir.join(format!("homo-{id}.safetensors"));
        if homo_path.is_file() {
            let expected = golden(&homo_path);
            let logits = model
                .homosolver_logits(text)
                .expect("Homosolver-Logits")
                .expect("Golden vorhanden, aber keine Homosolver-Logits");
            compare(
                &format!("Satz {id} homo_logits"),
                &logits,
                &expected["logits"],
                1e-4,
            );
        } else {
            assert!(
                model
                    .homosolver_logits(text)
                    .expect("Homosolver-Logits")
                    .is_none(),
                "Satz {id}: Homosolver-Logits ohne Golden"
            );
        }

        let voices = if id < 5 {
            vec!["baya", "kseniya", "xenia", "aidar", "eugene"]
        } else {
            vec!["baya"]
        };
        for voice in voices {
            let path = golden_dir.join(format!("{id}-{voice}.safetensors"));
            let tensors = golden(&path);
            let stages = model.debug_stages(text, voice).expect("Pipeline");
            check_stages(
                &format!("Satz {id} {voice}"),
                &stages.output,
                &tensors,
                tensors.contains_key("encoder"),
            );
            if let Some(backbone) = tensors.get("backbone") {
                compare(
                    &format!("Satz {id} {voice} backbone"),
                    &stages.features,
                    backbone,
                    1e-3,
                );
            }
            if let Some(mag) = tensors.get("mag") {
                compare(&format!("Satz {id} {voice} mag"), &stages.mag, mag, 5e-3);
                compare(
                    &format!("Satz {id} {voice} cos"),
                    &stages.cos,
                    &tensors["cos"],
                    5e-3,
                );
                compare(
                    &format!("Satz {id} {voice} sin"),
                    &stages.sin,
                    &tensors["sin"],
                    5e-3,
                );
            }
            let expected_audio = values(&tensors["audio"]);
            compare_audio(
                &format!("Satz {id} {voice} audio"),
                &stages.audio,
                &expected_audio,
                5e-3,
            );
        }
    }
}

/// Regressions-Snapshot von `split_chunks`, kein Vergleich mit der Python-Referenz.
///
/// `golden/chunks.json` ist die Ausgabe einer handgeschriebenen Spiegelung von `split_chunks`
/// im Export-Skript (`scripts/export_silero.py`). Der Test hält die Rust-Chunklogik gegen
/// diesen Snapshot fest; ob sie der Python-Referenz entspricht, prüft er nicht.
#[test]
fn chunks_match_python() {
    let path = model_dir().join("golden/chunks.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        skip(&format!(
            "{} fehlt; Chunk-Test übersprungen",
            path.display()
        ));
        return;
    };
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&text).expect("chunks.json ist gültiges JSON");
    let corpus = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus.txt"),
    )
    .expect("tests/corpus.txt lesbar");
    let lines: Vec<&str> = corpus
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert_eq!(
        entries.len(),
        lines.len(),
        "chunks.json hat {} Einträge, tests/corpus.txt hat {} Zeilen",
        entries.len(),
        lines.len()
    );
    for (index, text) in lines.into_iter().enumerate() {
        let expected_chunks: Vec<String> = entries[index]["chunks"]
            .as_array()
            .expect("Feld chunks ist ein Array")
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .expect("chunks enthält nur Strings")
                    .to_string()
            })
            .collect();
        let expected_kept: Vec<String> = entries[index]["kept"]
            .as_array()
            .expect("Feld kept ist ein Array")
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .expect("kept enthält nur Strings")
                    .to_string()
            })
            .collect();
        let chunks = silero_v5_candle::text::normalize::split_chunks(text, 400);
        assert_eq!(chunks, expected_chunks, "Zeile {index}: chunks");
        let kept: Vec<String> = chunks
            .into_iter()
            .filter(|chunk| silero_v5_candle::text::normalize::has_cyrillic(chunk))
            .collect();
        assert_eq!(kept, expected_kept, "Zeile {index}: kept");
    }
}
