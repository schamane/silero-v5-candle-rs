//! Vergleich gegen die Goldens aus `scripts/export_silero.py`.
//!
//! Läuft ohne Modellverzeichnis als übersprungener Test, damit
//! `cargo test --workspace` ohne die (nicht eingecheckten) Gewichte baut.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use silero_v5_ru_candle::Silero;
use silero_v5_ru_candle::acoustic::forward::AcousticOutput;

/// Meldet einen Skip direkt auf stderr: libtest fängt `eprintln!` bei bestandenen Tests ab.
fn skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "SKIP: {reason}");
}

fn model_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("models/silero-v5-ru")
}

fn load_model() -> Option<Silero> {
    let dir = model_dir();
    if !dir.is_dir() {
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
    tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn compare(name: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) {
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
        skip("Golden-Tests brauchen --release (candle im Debug-Build zu langsam)");
        return;
    }
    let Some(model) = load_model() else { return };
    let dir = model_dir();
    let golden_dir = dir.join("golden");
    let texts = std::fs::read_to_string(golden_dir.join("text.jsonl")).expect("text.jsonl");
    let entries: Vec<serde_json::Value> = texts
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for entry in &entries {
        let id = entry["id"].as_u64().unwrap();
        let text = entry["text"].as_str().unwrap();
        let prepared = model.prepare(text);
        assert_eq!(
            prepared.sentence.iter().collect::<String>(),
            entry["prepared"].as_str().unwrap(),
            "Satz {id}: prepared"
        );
        assert_eq!(
            prepared.clean.iter().collect::<String>(),
            entry["clean"].as_str().unwrap()
        );
        assert_eq!(prepared.has_text, entry["has_text"].as_bool().unwrap());
        if !prepared.has_text {
            continue;
        }
        let stages = model.debug_stages(text, "baya").expect("Pipeline");
        assert_eq!(
            stages.homo.iter().collect::<String>(),
            entry["homo"].as_str().unwrap(),
            "Satz {id}: homosolver"
        );
        assert_eq!(
            stages.accented.iter().collect::<String>(),
            entry["accented"].as_str().unwrap(),
            "Satz {id}: accentor"
        );
        let sequence: Vec<u32> = entry["sequence"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
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
    let entries: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    let corpus = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus.txt"),
    )
    .unwrap();
    for (index, text) in corpus
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let expected_chunks: Vec<String> = entries[index]["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect();
        let expected_kept: Vec<String> = entries[index]["kept"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect();
        let chunks = silero_v5_ru_candle::text::normalize::split_chunks(text, 400);
        assert_eq!(chunks, expected_chunks, "Zeile {index}: chunks");
        let kept: Vec<String> = chunks
            .into_iter()
            .filter(|chunk| silero_v5_ru_candle::text::normalize::has_cyrillic(chunk))
            .collect();
        assert_eq!(kept, expected_kept, "Zeile {index}: kept");
    }
}
